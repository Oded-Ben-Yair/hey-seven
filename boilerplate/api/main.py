"""FastAPI application entry point for the Casino Host Agent.

Creates the FastAPI app, configures middleware, mounts routes, and
initializes the LangGraph agent on startup.
"""

import asyncio
import json
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.middleware.base import BaseHTTPMiddleware

from .middleware import (
    configure_cors,
    error_handling_middleware,
    rate_limiting_middleware,
    request_logging_middleware,
)
from .routes import router, set_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A fully configured FastAPI app ready to serve.
    """
    app = FastAPI(
        title="Hey Seven Casino Host Agent",
        description=(
            "AI-powered casino host that handles player lookups, comp "
            "calculations, reservations, personalized messaging, and "
            "regulatory compliance."
        ),
        version=os.environ.get("APP_VERSION", "0.1.0"),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Middleware (applied in reverse order — last added = first executed)
    app.add_middleware(BaseHTTPMiddleware, dispatch=rate_limiting_middleware)
    app.add_middleware(BaseHTTPMiddleware, dispatch=error_handling_middleware)
    app.add_middleware(BaseHTTPMiddleware, dispatch=request_logging_middleware)
    configure_cors(app)

    # Routes
    app.include_router(router)

    # Startup event: initialize agent
    @app.on_event("startup")
    async def startup_event() -> None:
        """Initialize the LangGraph agent on application startup."""
        logger.info("Initializing Casino Host agent...")
        try:
            from langgraph_agent.agent import create_agent

            agent = create_agent()
            set_agent(agent)
            logger.info("Casino Host agent initialized successfully.")
        except Exception:
            logger.exception(
                "Failed to initialize agent. Chat endpoints will be unavailable."
            )

    # Health check at root (for Cloud Run)
    @app.get("/health")
    async def root_health() -> dict:
        """Root-level health check for Cloud Run startup probes."""
        return {"status": "healthy"}

    # WebSocket endpoint for streaming
    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket) -> None:
        """WebSocket endpoint for streaming chat responses.

        Accepts JSON messages with the format:
            {"message": "...", "thread_id": "..."}

        Streams the agent's response back as it's generated.
        """
        await websocket.accept()
        logger.info("WebSocket connection established.")

        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                payload = json.loads(data)

                message = payload.get("message", "")
                thread_id = payload.get("thread_id")

                if not message:
                    await websocket.send_json(
                        {"error": "Empty message", "type": "error"}
                    )
                    continue

                # Send acknowledgment
                await websocket.send_json(
                    {"type": "ack", "thread_id": thread_id}
                )

                try:
                    from langgraph_agent.agent import chat

                    from .routes import _get_agent

                    agent = _get_agent()
                    result = await chat(
                        agent=agent,
                        message=message,
                        thread_id=thread_id,
                    )

                    # Send complete response
                    await websocket.send_json(
                        {
                            "type": "response",
                            "response": result["response"],
                            "thread_id": result["thread_id"],
                            "player_id": result.get("player_id"),
                            "escalation": result.get("escalation", False),
                            "compliance_flags": result.get("compliance_flags", []),
                        }
                    )
                except Exception:
                    logger.exception("WebSocket chat error")
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": "Failed to process message.",
                        }
                    )

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected.")
        except Exception:
            logger.exception("WebSocket error")

    return app


# Create the app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
