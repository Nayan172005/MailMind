"""Ask route — POST /ask"""

from fastapi import APIRouter, HTTPException

from rag.schemas import AskRequest, AskResponse

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Question answering with grounded generation."""
    from api.main import pipeline

    if not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Index not loaded. Build the index first.")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        return pipeline.ask(request.query, top_k=request.top_k)
    except RuntimeError as e:
        if "GEMINI_API_KEY" in str(e):
            raise HTTPException(status_code=500, detail="Gemini API key not configured.")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
