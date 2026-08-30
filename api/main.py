"""
MailMind API — FastAPI Application

Main application with CORS, startup event, and route registration.

Usage:
    uvicorn api.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.pipeline import RAGPipeline
from rag.utils import get_logger

logger = get_logger("api")

# Global pipeline instance
pipeline = RAGPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FAISS index on startup."""
    logger.info("Starting MailMind API...")
    try:
        pipeline.load_index()
        logger.info("FAISS index loaded successfully.")
    except FileNotFoundError:
        logger.warning(
            "No FAISS index found. Use POST /index/build to create one."
        )
    yield
    logger.info("Shutting down MailMind API.")


app = FastAPI(
    title="MailMind API",
    description="Intelligent Email Summarization and Information Retrieval",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Register routes ─────────────────────────────────────────────────

from api.routes.search import router as search_router
from api.routes.ask import router as ask_router
from api.routes.summarize import router as summarize_router
from api.routes.conversation import router as conversation_router
from api.routes.index import router as index_router
from api.routes.email import router as email_router

app.include_router(search_router, tags=["Search"])
app.include_router(ask_router, tags=["Question Answering"])
app.include_router(summarize_router, tags=["Summarization"])
app.include_router(conversation_router, tags=["Conversation"])
app.include_router(index_router, tags=["Index Management"])
app.include_router(email_router, tags=["Email"])


@app.get("/", tags=["Health"])
async def root():
    """API health check."""
    return {
        "service": "MailMind API",
        "status": "running",
        "index_loaded": pipeline.is_ready,
    }
