"""FastAPI application for the Hey Seven Property Q&A Agent.

Uses lifespan context manager, SSE streaming via sse-starlette,
and pure ASGI middleware (no BaseHTTPMiddleware).
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from src.config import get_settings

from .middleware import (
    ApiKeyMiddleware,
    ErrorHandlingMiddleware,
    RateLimitMiddleware,
    RequestBodyLimitMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
)
from .models import ChatRequest, HealthResponse, PropertyInfoResponse

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize agent and property data on startup, cleanup on shutdown."""
    logger.info("Initializing Property Q&A agent...")
    try:
        from src.agent.graph import build_graph

        app.state.agent = build_graph()
        logger.info("Agent initialized successfully.")
    except Exception:
        logger.exception("Failed to initialize agent. /chat will return 503.")
        app.state.agent = None

    # Run RAG ingestion if ChromaDB index doesn't exist yet
    chroma_dir = Path(settings.CHROMA_PERSIST_DIR)
    if not chroma_dir.exists():
        try:
            from src.rag.pipeline import ingest_property

            ingest_property()
            logger.info("Property data ingested into ChromaDB.")
        except Exception:
            logger.exception("Failed to ingest property data.")

    # Load property metadata for /property endpoint
    property_path = Path(settings.PROPERTY_DATA_PATH)
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
        version=settings.VERSION,
        lifespan=lifespan,
    )

    # CORS — configured per environment
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID", "X-API-Key"],
    )

    # Pure ASGI middleware (added in reverse execution order)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestBodyLimitMiddleware)
    app.add_middleware(ApiKeyMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # ------------------------------------------------------------------
    # POST /chat — SSE streaming response (real token streaming)
    # ------------------------------------------------------------------
    @app.post("/chat")
    async def chat_endpoint(request: Request, body: ChatRequest):
        agent = getattr(request.app.state, "agent", None)
        if agent is None:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=503,
                content={"error": "Agent not initialized. Try again later."},
                headers={"Retry-After": "30"},
            )

        from src.agent.graph import chat_stream

        async def event_generator():
            try:
                async with asyncio.timeout(settings.SSE_TIMEOUT_SECONDS):
                    async for event in chat_stream(
                        agent, body.message, body.thread_id
                    ):
                        if await request.is_disconnected():
                            logger.info("Client disconnected, cancelling stream")
                            return
                        yield event
            except TimeoutError:
                logger.warning("SSE stream timed out after %ds", settings.SSE_TIMEOUT_SECONDS)
                yield {
                    "event": "error",
                    "data": json.dumps({"error": "Response timed out. Please try again."}),
                }
                yield {"event": "done", "data": json.dumps({"done": True})}
            except Exception:
                logger.exception("Error during SSE stream")
                yield {
                    "event": "error",
                    "data": json.dumps({"error": "An error occurred while generating the response."}),
                }
                yield {"event": "done", "data": json.dumps({"done": True})}

        return EventSourceResponse(event_generator())

    # ------------------------------------------------------------------
    # GET /health — Health check
    # ------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request):
        from fastapi.responses import JSONResponse

        ready = getattr(request.app.state, "ready", False)
        agent_ready = getattr(request.app.state, "agent", None) is not None
        property_loaded = bool(getattr(request.app.state, "property_data", None))
        all_healthy = ready and agent_ready and property_loaded
        body = HealthResponse(
            status="healthy" if all_healthy else "degraded",
            version=settings.VERSION,
            agent_ready=agent_ready,
            property_loaded=property_loaded,
        )
        # Return 503 for degraded state so Cloud Run / k8s don't route
        # traffic to unhealthy containers.
        status_code = 200 if all_healthy else 503
        return JSONResponse(content=body.model_dump(), status_code=status_code)

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


app = create_app()

if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=True,
    )
