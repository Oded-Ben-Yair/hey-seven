"""FastAPI application entry point for the Casino Host Agent.

Creates the FastAPI app with a ``lifespan`` context manager (not the
deprecated ``@app.on_event``), mounts pure ASGI middleware, and exposes
the WebSocket chat endpoint with authentication and safety limits.
"""

import asyncio
import hmac
import json
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .middleware import (
    ErrorHandlingMiddleware,
    RequestLoggingMiddleware,
    configure_cors,
)
from .routes import HealthResponse, router

# ---------------------------------------------------------------------------
# Structured logging (Cloud Logging compatible)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize agent on startup, cleanup on shutdown.

    Stores the agent on ``app.state.agent`` so all routes can access it
    via ``request.app.state.agent`` -- no global mutable state needed.
    """
    logger.info("Initializing Casino Host agent...")
    try:
        from langgraph_agent.agent import create_agent

        agent = create_agent()
        app.state.agent = agent
        logger.info("Casino Host agent initialized successfully.")
    except Exception:
        logger.exception(
            "Failed to initialize agent. Chat endpoints will return 503."
        )
        app.state.agent = None

    yield  # Application runs here

    # Shutdown cleanup
    logger.info("Shutting down Casino Host agent.")
    app.state.agent = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

# WebSocket constants
_WS_MAX_MESSAGE_BYTES = 8_192  # 8 KiB per message
_WS_RATE_LIMIT_PER_SEC = 5
_WS_IDLE_TIMEOUT_SEC = 300  # 5 minutes


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
        version=os.getenv("APP_VERSION", "0.1.0"),
        docs_url="/docs" if os.getenv("ENVIRONMENT", "dev") != "production" else None,
        redoc_url="/redoc" if os.getenv("ENVIRONMENT", "dev") != "production" else None,
        lifespan=lifespan,
    )

    # Pure ASGI middleware (applied in reverse order -- last added = first executed)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    configure_cors(app)

    # Versioned routes
    app.include_router(router)

    # ------------------------------------------------------------------
    # Root health check (for Cloud Run startup probes)
    # ------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    async def root_health() -> HealthResponse | JSONResponse:
        """Root-level health check for Cloud Run startup probes.

        Returns 503 when the agent is not ready.
        """
        version = os.getenv("APP_VERSION", "0.1.0")
        agent = getattr(app.state, "agent", None)
        if agent is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "version": version,
                    "agent_ready": False,
                },
            )
        return HealthResponse(status="healthy", version=version, agent_ready=True)

    # ------------------------------------------------------------------
    # WebSocket endpoint with auth, rate limits, and size limits
    # ------------------------------------------------------------------
    @app.websocket("/ws/chat")
    async def websocket_chat(
        websocket: WebSocket,
        api_key: str | None = Query(None, alias="api_key"),
    ) -> None:
        """WebSocket endpoint for streaming chat responses.

        Authentication is done via the ``api_key`` query parameter
        (WebSocket does not support custom headers during the handshake
        in browser clients).

        Message format (JSON):
            {"message": "...", "thread_id": "..."}

        Safety limits:
            - Max message size: 8 KiB
            - Max rate: 5 messages/second
            - Idle timeout: 5 minutes
        """
        # --- Auth ---
        expected_key = os.getenv("API_KEY")
        if not expected_key or not api_key or not hmac.compare_digest(api_key, expected_key):
            await websocket.close(code=4001, reason="Unauthorized")
            return

        await websocket.accept()
        logger.info("WebSocket connection established.")

        msg_timestamps: list[float] = []

        try:
            while True:
                # Idle timeout
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=_WS_IDLE_TIMEOUT_SEC,
                    )
                except asyncio.TimeoutError:
                    await websocket.close(code=4002, reason="Idle timeout")
                    return

                # Size limit
                if len(data.encode("utf-8")) > _WS_MAX_MESSAGE_BYTES:
                    await websocket.send_json(
                        {"type": "error", "error": "Message too large (max 8 KiB)."}
                    )
                    continue

                # Per-connection rate limit (sliding window)
                now = time.monotonic()
                msg_timestamps = [t for t in msg_timestamps if now - t < 1.0]
                if len(msg_timestamps) >= _WS_RATE_LIMIT_PER_SEC:
                    await websocket.send_json(
                        {"type": "error", "error": "Rate limit exceeded."}
                    )
                    continue
                msg_timestamps.append(now)

                # Parse payload
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "error", "error": "Invalid JSON."}
                    )
                    continue

                message = payload.get("message", "")
                thread_id = payload.get("thread_id")

                if not message or not isinstance(message, str):
                    await websocket.send_json(
                        {"type": "error", "error": "Empty or invalid message."}
                    )
                    continue

                if len(message) > 4096:
                    await websocket.send_json(
                        {"type": "error", "error": "Message too long (max 4096 chars)."}
                    )
                    continue

                # Ack
                await websocket.send_json(
                    {"type": "ack", "thread_id": thread_id}
                )

                # Process
                try:
                    agent = getattr(app.state, "agent", None)
                    if agent is None:
                        await websocket.send_json(
                            {"type": "error", "error": "Agent not initialized."}
                        )
                        continue

                    from langgraph_agent.agent import chat

                    result = await chat(
                        agent=agent,
                        message=message,
                        thread_id=thread_id,
                    )

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
                    logger.exception("WebSocket chat processing error")
                    await websocket.send_json(
                        {"type": "error", "error": "Failed to process message."}
                    )

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected.")
        except Exception:
            logger.exception("WebSocket connection error")

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (for uvicorn)
# ---------------------------------------------------------------------------

app = create_app()

# ---------------------------------------------------------------------------
# Graceful shutdown for Cloud Run (SIGTERM)
# ---------------------------------------------------------------------------

signal.signal(signal.SIGTERM, lambda _sig, _frame: sys.exit(0))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENVIRONMENT", "dev") != "production",
        log_level="info",
    )
