"""Search route — POST /search"""

from fastapi import APIRouter, HTTPException

from rag.schemas import SearchRequest, SearchResponse

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_emails(request: SearchRequest):
    """Semantic email search."""
    from api.main import pipeline

    if not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Index not loaded. Build the index first.")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        return pipeline.search(request.query, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
