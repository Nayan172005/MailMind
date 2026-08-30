"""Conversation route — POST /conversation/ask"""

from fastapi import APIRouter, HTTPException

from rag.schemas import ConversationRequest, ConversationResponse

router = APIRouter()


@router.post("/conversation/ask", response_model=ConversationResponse)
async def conversation_ask(request: ConversationRequest):
    """Conversational question answering with session history."""
    from api.main import pipeline

    if not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Index not loaded. Build the index first.")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        return pipeline.conversation_ask(
            session_id=request.session_id,
            query=request.query,
            top_k=request.top_k,
        )
    except RuntimeError as e:
        if "GEMINI_API_KEY" in str(e):
            raise HTTPException(status_code=500, detail="Gemini API key not configured.")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Clear a conversation session."""
    from api.main import pipeline

    pipeline.clear_conversation(session_id)
    return {"status": "cleared", "session_id": session_id}
