"""Email detail route — GET /email/{message_id}"""

from fastapi import APIRouter, HTTPException

from rag.schemas import EmailDetailResponse

router = APIRouter()


@router.get("/email/{message_id}", response_model=EmailDetailResponse)
async def get_email(message_id: str):
    """Retrieve full email details by message_id."""
    from api.main import pipeline

    if not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Index not loaded.")

    result = pipeline.get_email(message_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Email not found: {message_id}")

    return result
