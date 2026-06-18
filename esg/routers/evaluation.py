from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from services.evaluation_service import EvaluationService
from typing import Dict, Any

router = APIRouter()

@router.post("/evaluate")
async def evaluate_search(
    file: UploadFile = File(...),
    collection_id: str = Form(...),
    top_k: int = Form(10),
    threshold: float = Form(0.7)
) -> Dict[str, Any]:
    try:
        file_content = await file.read()
        evaluation_service = EvaluationService()
        
        results = await evaluation_service.process_evaluation(
            file_content=file_content,
            collection_id=collection_id,
            top_k=top_k,
            threshold=threshold
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 