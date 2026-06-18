from langchain_community.llms import Ollama
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Dict
from services.std_service import StdService
import os
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AbbrService:

    def __init__(self):
        self.std_service = None  
        
    def _get_std_service(self, embedding_options: dict) -> StdService:

        try:
            return StdService(
                provider=embedding_options.get("provider", "huggingface"),
                model=embedding_options.get("model", "BAAI/bge-m3"),
                db_path=f"db/{embedding_options.get('dbName', 'snomed_bge_m3')}.db",
                collection_name=embedding_options.get("collectionName", "concepts_only_name")
            )
        except Exception as e:
            logger.error(f"Failed to initialize StdService: {str(e)}")
            raise ValueError(f"Failed to initialize standardization service: {str(e)}")

    def _get_llm(self, llm_options: dict):
 
        provider = llm_options.get("provider", "ollama")
        model = llm_options.get("model", "llama3.1:8b")
        
        if provider == "ollama":
            return Ollama(model=model)
        elif provider == "openai":
            return ChatOpenAI(
                model=model,
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
    def simple_ollama_expansion(self, text: str, llm_options: dict) -> Dict:

        llm = self._get_llm(llm_options)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You job is to simply return the input with ALL abbreviations in medical domain replaced with their expanded forms."),
            ("system", "Input consist of clinical notes. Keep all occurrences of ___ in the output."),
            ("system", "Do NOT include supplementary messages like -> Here are the expanded abbreviations: I only want the output as a string."),
            ("system", "Do NOT spell out numbers, leave them as digits."),
            ("human", "{input}"),
        ])
        
        chain = prompt | llm
        result = chain.invoke({"input": text})
        
        expanded_text = result.content if hasattr(result, 'content') else str(result)
        
        return {
            "input": text,
            "expanded_text": expanded_text,
            "method": "simple_llm"
        }

    def llm_rank_query_db(self, text: str, context: str, llm_options: dict, embedding_options: dict) -> Dict:

        try:
            self.std_service = self._get_std_service(embedding_options)

            llm = self._get_llm(llm_options)
            expand_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given the medical abbreviation and its context, provide the most likely expansion based on common medical usage."),
                ("human", f"Abbreviation: {text}\nContext: {context}")
            ])
            
            chain = expand_prompt | llm
            expansion_result = chain.invoke({})
            
            expansion_text = expansion_result.content if hasattr(expansion_result, 'content') else str(expansion_result)
            
            std_terms = self.std_service.search_similar_terms(expansion_text)
            
            return {
                "input": text,
                "context": context,
                "expansion": expansion_text,
                "standardized_terms": std_terms,
                "method": "llm_db"
            }
        except Exception as e:
            logger.error(f"Error in llm_rank_query_db: {str(e)}")
            raise ValueError(f"Failed to process abbreviation expansion: {str(e)}") 