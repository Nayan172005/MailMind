"""Index management routes — POST /index/build, POST /index/update, GET /index/status"""

from fastapi import APIRouter, HTTPException

from rag.schemas import IndexBuildRequest, IndexStatusResponse

router = APIRouter()


@router.post("/index/build", response_model=IndexStatusResponse)
async def build_index(request: IndexBuildRequest = IndexBuildRequest()):
    """Build the FAISS index from the email dataset."""
    from api.main import pipeline

    try:
        return pipeline.build_index(force_rebuild=request.force_rebuild)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/update", response_model=IndexStatusResponse)
async def update_index():
    """Update the index with newly added emails."""
    from api.main import pipeline

    try:
        return pipeline.update_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/index/status", response_model=IndexStatusResponse)
async def index_status():
    """Get the current index status."""
    from api.main import pipeline

    return pipeline.get_index_status()
