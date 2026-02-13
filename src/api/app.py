"""FastAPI application for the Hey Seven Property Q&A Agent.

Uses lifespan context manager, SSE streaming via sse-starlette,
and pure ASGI middleware (no BaseHTTPMiddleware).
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from .middleware import ErrorHandlingMiddleware, RequestLoggingMiddleware
from .models import ChatRequest, HealthResponse, PropertyInfoResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize agent and property data on startup, cleanup on shutdown."""
    logger.info("Initializing Property Q&A agent...")
    try:
        from src.agent.graph import create_agent

        app.state.agent = create_agent()
        logger.info("Agent initialized successfully.")
    except Exception:
        logger.exception("Failed to initialize agent. /chat will return 503.")
        app.state.agent = None

    # Run RAG ingestion if ChromaDB index doesn't exist yet
    chroma_dir = Path("data/chroma")
    if not chroma_dir.exists():
        try:
            from src.rag.pipeline import ingest_property

            ingest_property("data/mohegan_sun.json")
            logger.info("Property data ingested into ChromaDB.")
        except Exception:
            logger.exception("Failed to ingest property data.")

    # Load property metadata for /property endpoint
    property_path = Path("data/mohegan_sun.json")
    if property_path.exists():
        with open(property_path, encoding="utf-8") as f:
            app.state.property_data = json.load(f)
        logger.info("Property metadata loaded.")
    else:
        logger.warning("Property data file not found at %s", property_path)
        app.state.property_data = {}

    app.state.ready = True
    yield
    app.state.ready = False
    app.state.agent = None
    logger.info("Application shutdown complete.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Hey Seven Property Q&A Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — wide open for demo
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pure ASGI middleware (added in reverse execution order)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # ------------------------------------------------------------------
    # POST /chat — SSE streaming response
    # ------------------------------------------------------------------
    @app.post("/chat")
    async def chat_endpoint(request: Request, body: ChatRequest):
        agent = getattr(request.app.state, "agent", None)
        if agent is None:
            return EventSourceResponse(
                _error_event("Agent not initialized"), status_code=503
            )

        from src.agent.graph import chat

        result = await chat(agent, body.message, body.thread_id)

        async def event_generator():
            yield {"data": json.dumps({**result, "done": True})}

        return EventSourceResponse(event_generator())

    # ------------------------------------------------------------------
    # GET /health — Health check
    # ------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request):
        ready = getattr(request.app.state, "ready", False)
        agent_ready = getattr(request.app.state, "agent", None) is not None
        property_loaded = bool(getattr(request.app.state, "property_data", None))
        all_healthy = ready and agent_ready and property_loaded
        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            version="0.1.0",
            agent_ready=agent_ready,
            property_loaded=property_loaded,
        )

    # ------------------------------------------------------------------
    # GET /property — Property metadata
    # ------------------------------------------------------------------
    @app.get("/property", response_model=PropertyInfoResponse)
    async def property_info(request: Request):
        data = getattr(request.app.state, "property_data", {})
        prop = data.get("property", {})
        categories = [k for k in data if k != "property"]
        doc_count = sum(
            len(v) if isinstance(v, list) else 1
            for k, v in data.items()
            if k != "property"
        )
        return PropertyInfoResponse(
            name=prop.get("name", "Unknown"),
            location=prop.get("location", "Unknown"),
            categories=categories,
            document_count=doc_count,
        )

    # ------------------------------------------------------------------
    # Static files for frontend (MUST be last)
    # ------------------------------------------------------------------
    static_dir = Path(__file__).resolve().parent.parent.parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


async def _error_event(message: str):
    """Yield a single SSE error event."""
    yield {"data": json.dumps({"error": message, "done": True})}


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=True,
    )
