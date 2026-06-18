import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging

class EvaluationService:
    def __init__(self):
        self.output_dir = Path("06-evaluation-result")
        self.output_dir.mkdir(exist_ok=True)

    async def process_evaluation(
        self,
        file_content: bytes,
        collection_id: str,
        top_k: int,
        threshold: float
    ) -> Dict[str, Any]:
        try:
            df = pd.read_csv(pd.io.common.BytesIO(file_content))
            
            df['combined_text'] = df.apply(
                lambda row: f"{row['ID']} {row['Disclosure Requirement']} {row['Corresponding Text']}", 
                axis=1
            )
            
            from services.search_service import SearchService
            search_service = SearchService()
            
            results = []
            total_score_hit = 0
            total_score_find = 0
            valid_queries = 0
            
            for _, row in df.iterrows():
                try:
                    page_numbers = str(row['Page Number'])
                    expected_pages = [
                        int(x.strip()) 
                        for x in page_numbers.split(',') 
                        if x.strip().isdigit()
                    ]
                    
                    if not expected_pages:
                        continue
                    
                    search_response = await search_service.search(
                        query=row['combined_text'],
                        collection_id=collection_id,
                        top_k=top_k,
                        threshold=threshold
                    )
                    
                    search_results = search_response["results"]
                    found_pages = [int(result['metadata']['page']) for result in search_results]
                    
                    hits = sum(1 for page in found_pages if page in expected_pages)
                    score_hit = hits / len(found_pages) if found_pages else 0
                    score_find = len(set(found_pages) & set(expected_pages)) / len(expected_pages)
                    
                    result_entry = {
                        "id": row['ID'],
                        "requirement": row['Disclosure Requirement'],
                        "expected_pages": expected_pages,
                        "found_pages": found_pages,
                        "score_hit": score_hit,
                        "score_find": score_find,
                        "compliance_status": row['Compliance Status']
                    }
                    
                    results.append(result_entry)
                    total_score_hit += score_hit
                    total_score_find += score_find
                    valid_queries += 1
                    
                except Exception as e:
                    print(f"Error processing row: {str(e)}")
                    continue
            
            if valid_queries == 0:
                raise ValueError("No valid queries found in the CSV file")
            
            average_scores = {
                "score_hit": total_score_hit / valid_queries,
                "score_find": total_score_find / valid_queries
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            evaluation_results = {
                "results": results,
                "average_scores": average_scores,
                "total_queries": valid_queries
            }
            
            return evaluation_results
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise 