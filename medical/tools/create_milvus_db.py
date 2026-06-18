from pymilvus import model
from pymilvus import MilvusClient
import pandas as pd
from tqdm import tqdm
import logging
from dotenv import load_dotenv
load_dotenv()
import torch    
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

file_path = "backend/data/SNOMED_5000.csv"
db_path = "backend/db/snomed_bge_m3.db"

client = MilvusClient(db_path)

collection_name = "concepts_only_name"
# collection_name = "concepts_with_synonym"

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
    # FieldSchema(name="synonyms", dtype=DataType.VARCHAR, max_length=1000), 
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

    # docs = [f"Term: {row['concept_name']}; Synonyms: {row['Synonyms']}" for _, row in batch_df.iterrows()]
    docs = []
    for _, row in batch_df.iterrows():
        doc_parts = [row['concept_name']]

        # if row['Full Name'] != "NA" and row['Full Name'] != row['concept_name']:
        #     doc_parts.append(",Full Name: " + row['Full Name'])

        # if row['Synonyms'] != "NA" and row['Synonyms'] != row['concept_name']:
        #     doc_parts.append(", Synonyms: " + row['Synonyms'])

        # if row['Definitions'] != "NA" and row['Definitions'] not in [row['concept_name'], row.get('Full Name', '')]:
        #     doc_parts.append(", Definitions: " + row['Definitions'])

        docs.append(" ".join(doc_parts))

    try:
        embeddings = embedding_function(docs)
        logging.info(f"Generated embeddings for batch {start_idx // batch_size + 1}")
    except Exception as e:
        logging.error(f"Error generating embeddings for batch {start_idx // batch_size + 1}: {e}")
        continue

    data = [
        {
            # "id": idx + start_idx,
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
            # "invalid_reason": str(row['invalid_reason']),
            # "full_name": str(row['Full Name']),
            # "synonyms": str(row['Synonyms']),
            # "definitions": str(row['Definitions']),
            "input_file": file_path
        } for idx, (_, row) in enumerate(batch_df.iterrows())
    ]

    try:
        res = client.insert(
            collection_name=collection_name,
            data=data
        )
        logging.info(f"Inserted batch {start_idx // batch_size + 1}, result: {res}")
    except Exception as e:
        logging.error(f"Error inserting batch {start_idx // batch_size + 1}: {e}")

logging.info("Insert process completed.")

# query = "somatic hallucination"
query = "SOB"
query_embeddings = embedding_function([query])


search_result = client.search(
    collection_name=collection_name,
    data=[query_embeddings[0].tolist()],
    limit=5,
    output_fields=["concept_name", 
                #    "synonyms", 
                   "concept_class_id", 
                   ]
)
logging.info(f"Search result for 'Somatic hallucination': {search_result}")

query_result = client.query(
    collection_name=collection_name,
    filter="concept_name == 'Dyspnea'",
    output_fields=["concept_name", 
                #    "synonyms", 
                   "concept_class_id", 
                   ],
    limit=5
)
logging.info(f"Query result for concept_name == 'Dyspnea': {query_result}")
