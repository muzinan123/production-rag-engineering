from pymilvus import MilvusClient
from dotenv import load_dotenv
from utils.embedding_factory import EmbeddingFactory
from utils.embedding_config import EmbeddingProvider, EmbeddingConfig
import os
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class StdService:

    def __init__(self, 
                 provider="huggingface",
                 model="BAAI/bge-m3",
                 db_path="db/snomed_bge_m3.db",
                 collection_name="concepts_only_name"):

        provider_mapping = {
            'openai': EmbeddingProvider.OPENAI,
            'bedrock': EmbeddingProvider.BEDROCK,
            'huggingface': EmbeddingProvider.HUGGINGFACE
        }
        

        embedding_provider = provider_mapping.get(provider.lower())
        if embedding_provider is None:
            raise ValueError(f"Unsupported provider: {provider}")
            
        config = EmbeddingConfig(
            provider=embedding_provider,
            model_name=model
        )
        self.embedding_func = EmbeddingFactory.create_embedding_function(config)
        
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        self.client.load_collection(self.collection_name)

    def search_similar_terms(self, query: str, limit: int = 5) -> List[Dict]:
        query_embedding = self.embedding_func.embed_query(query)
        

        search_params = {
            "collection_name": self.collection_name,
            "data": [query_embedding],
            "limit": limit,
            "output_fields": [
                "concept_id", "concept_name", "domain_id", 
                "vocabulary_id", "concept_class_id", "standard_concept",
                "concept_code", "synonyms"
            ],
            # "filter": "domain_id == 'Condition'"
        }
        

        search_result = self.client.search(**search_params)

        results = []
        for hit in search_result[0]:
            results.append({
                "concept_id": hit['entity'].get('concept_id'),
                "concept_name": hit['entity'].get('concept_name'),
                "domain_id": hit['entity'].get('domain_id'),
                "vocabulary_id": hit['entity'].get('vocabulary_id'),
                "concept_class_id": hit['entity'].get('concept_class_id'),
                "standard_concept": hit['entity'].get('standard_concept'),
                "concept_code": hit['entity'].get('concept_code'),
                "synonyms": hit['entity'].get('synonyms'),
                "distance": float(hit['distance'])
            })

        return results

    def __del__(self):

        if hasattr(self, 'client') and hasattr(self, 'collection_name'):
            self.client.release_collection(self.collection_name)