from pymilvus import model
from pymilvus import MilvusClient
import pandas as pd
from tqdm import tqdm
import logging
from dotenv import load_dotenv
load_dotenv()
import torch    
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
from neo4j import GraphDatabase
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j")  
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))


try:
    with neo4j_driver.session() as session:

        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()["count"]
        logging.info(f"Successfully connected to Neo4j. Total nodes in database: {count}")
        
        result = session.run("MATCH (c:ObjectConcept) RETURN count(c) as count")
        concept_count = result.single()["count"]
        logging.info(f"Total ObjectConcept nodes: {concept_count}")
        
        result = session.run("""
            MATCH (c:ObjectConcept)
            RETURN keys(c) as properties
            LIMIT 1
        """)
        properties = result.single()["properties"]
        logging.info(f"ObjectConcept node properties: {properties}")
        
        result = session.run("MATCH (d:Description) RETURN count(d) as count")
        desc_count = result.single()["count"]
        logging.info(f"Total Description nodes: {desc_count}")
        
        result = session.run("MATCH ()-[r:HAS_DESCRIPTION]->() RETURN count(r) as count")
        rel_count = result.single()["count"]
        logging.info(f"Total HAS_DESCRIPTION relationships: {rel_count}")
        
        test_concept = "267036007"  # Dyspnea
        result = session.run("""
            MATCH (c:ObjectConcept {id: $id})-[:HAS_DESCRIPTION]->(d:Description)
            RETURN c.id as concept_id, c.FSN as fsn, d.term as term
        """, id=test_concept)
        test_results = list(result)
        logging.info(f"Test query results for concept {test_concept}:")
        for record in test_results:
            logging.info(f"  Concept: {record['concept_id']}, FSN: {record['fsn']}, Term: {record['term']}")
        
except Exception as e:
    logging.error(f"Failed to connect to Neo4j: {e}")
    raise

def get_concept_descriptions(concept_id, concept_code):
    with neo4j_driver.session() as session:

        concept_check = session.run("""
            MATCH (c:ObjectConcept {id: $concept_code})
            RETURN c.id as id, c.FSN as fsn
        """, concept_code=concept_code)
        concept = concept_check.single()
        
        if not concept:
            logging.warning(f"Concept {concept_id} (code: {concept_code}) not found in Neo4j")
            return []
            
        logging.info(f"Found concept: {concept['id']} with FSN: {concept['fsn']}")
        
        result = session.run("""
            MATCH (c:ObjectConcept {id: $concept_code})-[:HAS_DESCRIPTION]->(d:Description)
            RETURN d.term as term, d.descriptionType as type, d.active as active
            ORDER BY d.descriptionType
        """, concept_code=concept_code)
        
        descriptions = [(record["term"], record["type"], record["active"]) for record in result]
        
        if descriptions:
            logging.info(f"Found {len(descriptions)} descriptions for concept {concept['id']}:")
            for term, type_, active in descriptions:
                logging.info(f"  - Term: {term}, Type: {type_}, Active: {active}")
        else:
            logging.warning(f"No descriptions found for concept {concept['id']}")
            
        return [desc[0] for desc in descriptions]

embedding_function = model.dense.SentenceTransformerEmbeddingFunction(
            # model_name='nvidia/NV-Embed-v2', 
            # model_name='dunzhang/stella_en_1.5B_v5',
            # model_name='all-mpnet-base-v2',
            # model_name='intfloat/multilingual-e5-large-instruct',
            # model_name='Alibaba-NLP/gte-Qwen2-1.5B-instruct',
            model_name='BAAI/bge-m3',
            # model_name='jinaai/jina-embeddings-v3',
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            trust_remote_code=True
        )
# embedding_function = model.dense.OpenAIEmbeddingFunction(model_name='text-embedding-3-large')

file_path = "backend/data/SNOMED_3.csv"
db_path = "backend/db/snomed_bge_m3.db"

client = MilvusClient(db_path)

collection_name = "concepts_with_synonym"

if client.has_collection(collection_name):
    logging.info(f"Dropping existing collection: {collection_name}")
    client.drop_collection(collection_name)

logging.info("Loading data from CSV")
df = pd.read_csv(file_path, 
                 dtype=str, 
                low_memory=False,
                 ).fillna("NA")

sample_doc = "Sample Text"
sample_embedding = embedding_function([sample_doc])[0]
vector_dim = len(sample_embedding)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim), 
    FieldSchema(name="concept_id", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="concept_name", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="domain_id", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="vocabulary_id", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="concept_class_id", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="standard_concept", dtype=DataType.VARCHAR, max_length=1),
    FieldSchema(name="concept_code", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="valid_start_date", dtype=DataType.VARCHAR, max_length=10),
    FieldSchema(name="valid_end_date", dtype=DataType.VARCHAR, max_length=10),
    # FieldSchema(name="full_name", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="synonyms", dtype=DataType.VARCHAR, max_length=1000),
    # FieldSchema(name="definitions", dtype=DataType.VARCHAR, max_length=1000), 
    FieldSchema(name="input_file", dtype=DataType.VARCHAR, max_length=500),
]
schema = CollectionSchema(fields, 
                          "SNOMED-CT Concepts", 
                          enable_dynamic_field=True)

if not client.has_collection(collection_name):
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        # dimension=vector_dim
    )
    logging.info(f"Created new collection: {collection_name}")

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",  
    index_type="AUTOINDEX", 
    metric_type="COSINE",  
    params={"nlist": 1024}  
)

client.create_index(
    collection_name=collection_name,
    index_params=index_params
)


batch_size = 1024

for start_idx in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
    end_idx = min(start_idx + batch_size, len(df))
    batch_df = df.iloc[start_idx:end_idx]

    docs = []
    for _, row in batch_df.iterrows():
        concept_id = row['concept_id']
        concept_code = row['concept_code']
        concept_name = row['concept_name']
        
        synonyms = get_concept_descriptions(concept_id, concept_code)
        synonyms_text = " ".join(synonyms) if synonyms else ""

        doc_parts = [concept_name]
        if synonyms_text:
            doc_parts.append(synonyms_text)
            
        docs.append(" ".join(doc_parts))

    try:
        embeddings = embedding_function(docs)
        logging.info(f"Generated embeddings for batch {start_idx // batch_size + 1}")
    except Exception as e:
        logging.error(f"Error generating embeddings for batch {start_idx // batch_size + 1}: {e}")
        continue

    data = []
    for idx, (_, row) in enumerate(batch_df.iterrows()):
        concept_id = row['concept_id']
        concept_code = row['concept_code']
        synonyms = get_concept_descriptions(concept_id, concept_code)
        synonyms_text = " ".join(synonyms) if synonyms else ""
        
        data.append({
            "vector": embeddings[idx],
            "concept_id": str(row['concept_id']),
            "concept_name": str(row['concept_name']),
            "domain_id": str(row['domain_id']),
            "vocabulary_id": str(row['vocabulary_id']),
            "concept_class_id": str(row['concept_class_id']),
            "standard_concept": str(row['standard_concept']),
            "concept_code": str(row['concept_code']),
            "valid_start_date": str(row['valid_start_date']),
            "valid_end_date": str(row['valid_end_date']),
            "synonyms": synonyms_text,
            "input_file": file_path
        })


    try:
        res = client.insert(
            collection_name=collection_name,
            data=data
        )
        logging.info(f"Inserted batch {start_idx // batch_size + 1}, result: {res}")
    except Exception as e:
        logging.error(f"Error inserting batch {start_idx // batch_size + 1}: {e}")

logging.info("Insert process completed.")


neo4j_driver.close()


# query = "somatic hallucination"
query = "SOB"
query_embeddings = embedding_function([query])



search_result = client.search(
    collection_name=collection_name,
    data=[query_embeddings[0].tolist()],
    limit=5,
    output_fields=["concept_name", "synonyms", "concept_class_id"]
)
logging.info(f"Search result for '{query}': {search_result}")


query_result = client.query(
    collection_name=collection_name,
    filter="concept_name == 'Dyspnea'",
    output_fields=["concept_name", "synonyms", "concept_class_id"],
    limit=5
)
logging.info(f"Query result for concept_name == 'Dyspnea': {query_result}")
