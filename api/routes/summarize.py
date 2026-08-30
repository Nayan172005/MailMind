"""Summarize route — POST /summarize"""

from fastapi import APIRouter, HTTPException

from rag.schemas import SummarizeRequest, SummarizeResponse

router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_emails(request: SummarizeRequest):
    """Summarize a single email or a group of emails matching a query."""
    from api.main import pipeline

    if not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Index not loaded. Build the index first.")

    if not request.message_id and not request.query:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'message_id' or 'query'.",
        )

    try:
        return pipeline.summarize(
            message_id=request.message_id,
            query=request.query,
            top_k=request.top_k,
        )
    except RuntimeError as e:
        if "GEMINI_API_KEY" in str(e):
            raise HTTPException(status_code=500, detail="Gemini API key not configured.")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
