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

from .errors import ErrorCode, error_response
from .middleware import (
    ApiKeyMiddleware,
    ErrorHandlingMiddleware,
    RateLimitMiddleware,
    RequestBodyLimitMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
)
from .models import (
    ChatRequest,
    FeedbackRequest,
    FeedbackResponse,
    GraphStructureResponse,
    HealthResponse,
    LiveResponse,
    PropertyInfoResponse,
)

# Note: settings are accessed via get_settings() at call sites rather than
# frozen at module level.  This ensures test monkeypatching works and
# avoids stale config values.  Logging is configured at app creation time.
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize agent and property data on startup, cleanup on shutdown."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Initializing Property Q&A agent...")
    try:
        from src.agent.graph import build_graph
        from src.agent.memory import get_checkpointer

        app.state.agent = build_graph(checkpointer=await get_checkpointer())
        logger.info("Agent initialized successfully.")
    except Exception:
        logger.exception("Failed to initialize agent. /chat will return 503.")
        app.state.agent = None

    # Run RAG ingestion if ChromaDB collection is empty or missing.
    # Checking the chroma.sqlite3 file is more reliable than directory existence
    # because the directory may exist but contain no data (e.g., after a failed run).
    # Wrap in asyncio.to_thread() because ChromaDB operations are synchronous
    # and would block the event loop during startup.
    #
    # SAFETY: Only ingest when VECTOR_DB=chroma (local dev). Production uses
    # pre-built Firestore/Vertex AI indexes — startup ingestion would cause
    # race conditions when multiple Cloud Run instances start simultaneously
    # (concurrent SQLite writes corrupt ChromaDB, duplicate chunks).
    if settings.VECTOR_DB == "chroma":
        chroma_db_file = Path(settings.CHROMA_PERSIST_DIR) / "chroma.sqlite3"
        if not chroma_db_file.exists():
            try:
                from src.rag.pipeline import ingest_property

                await asyncio.to_thread(ingest_property)
                logger.info("Property data ingested into ChromaDB.")
            except Exception:
                logger.exception("Failed to ingest property data.")
    else:
        logger.info(
            "Skipping startup RAG ingestion (VECTOR_DB=%s, not chroma)",
            settings.VECTOR_DB,
        )

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
    settings = get_settings()
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

    # Pure ASGI middleware — Starlette executes in REVERSE add order.
    # Add order (top to bottom) vs execution order (bottom to top):
    #   RateLimit        (added 1st, executes last / innermost)
    #   ApiKey           (added 2nd)
    #   Security         (added 3rd)
    #   Logging          (added 4th)
    #   ErrorHandling    (added 5th)
    #   BodyLimit        (added 6th, executes first / outermost)
    # BodyLimit is outermost so oversized payloads are rejected before any
    # middleware processes the request body (prevents memory consumption from
    # malicious oversized payloads). ErrorHandling wraps all inner middleware
    # so unhandled exceptions are caught and returned as structured 500s.
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(ApiKeyMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestBodyLimitMiddleware)

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
                content=error_response(ErrorCode.AGENT_UNAVAILABLE, "Agent not initialized. Try again later."),
                headers={"Retry-After": "30"},
            )

        from src.agent.graph import chat_stream

        sse_timeout = get_settings().SSE_TIMEOUT_SECONDS
        # Thread X-Request-ID from middleware into graph for observability correlation.
        # RequestLoggingMiddleware injects X-Request-ID into response headers;
        # we read the header set by the middleware (or generate a fallback).
        request_id = request.headers.get("x-request-id", None)

        async def event_generator():
            try:
                async with asyncio.timeout(sse_timeout):
                    # SSE heartbeat: send periodic pings to prevent client-side
                    # EventSource timeouts during long LLM generations (30s+).
                    # Without heartbeats, browsers close the connection and auto-
                    # reconnect, creating duplicate requests.
                    #
                    # Uses asyncio.wait_for() on each __anext__() call so that
                    # heartbeats fire even when the graph is blocked waiting for
                    # the first LLM token (router + retrieval + specialist can
                    # take 15-30s). Previous implementation only checked elapsed
                    # time INSIDE the async-for loop, which never fires while
                    # awaiting the next event.
                    _HEARTBEAT_INTERVAL = 15  # seconds
                    event_iter = chat_stream(
                        agent, body.message, body.thread_id,
                        request_id=request_id,
                    ).__aiter__()

                    while True:
                        if await request.is_disconnected():
                            logger.info("Client disconnected, cancelling stream")
                            return
                        try:
                            event = await asyncio.wait_for(
                                event_iter.__anext__(),
                                timeout=_HEARTBEAT_INTERVAL,
                            )
                            yield event
                        except TimeoutError:
                            # No event within heartbeat interval — send ping
                            yield {"event": "ping", "data": ""}
                        except StopAsyncIteration:
                            break
            except TimeoutError:
                logger.warning("SSE stream timed out after %ds", sse_timeout)
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
    # GET /live — Lightweight liveness probe (Cloud Run)
    # ------------------------------------------------------------------
    # Separated from /health to prevent instance flapping. /live confirms
    # the process is alive and the event loop is responsive. /health is
    # used as the READINESS probe (gated on agent + RAG + property data).
    # Cloud Run liveness probes that return 503 cause instance replacement,
    # amplifying outages when downstream deps (LLM API, vector store) are
    # temporarily degraded. /live avoids this by always returning 200.
    #
    # Cloud Run probe configuration (R10 fix — Gemini F18, GPT P0-F6):
    #   startupProbe:  /live  (is the process booting?)
    #   livenessProbe: /live  (is the process alive? — always 200)
    #   readinessProbe: /health (can the instance serve? — 503 on CB open)
    # WARNING: Do NOT use /health as livenessProbe. When CB is open (LLM
    # outage), /health returns 503 which would cause Cloud Run to replace
    # the instance in a loop, amplifying the outage.
    @app.get("/live", response_model=LiveResponse)
    async def liveness():
        return LiveResponse()

    # ------------------------------------------------------------------
    # GET /health — Health check (readiness/startup probe)
    # ------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request):
        from fastapi.responses import JSONResponse

        from src.observability.langfuse_client import is_observability_enabled

        ready = getattr(request.app.state, "ready", False)
        agent_ready = getattr(request.app.state, "agent", None) is not None
        property_loaded = bool(getattr(request.app.state, "property_data", None))

        # RAG health check: verify embeddings + vector store are accessible.
        # Uses cached state to stay fast (< 1s).  Catches exceptions to
        # report degraded rather than crashing the health endpoint.
        rag_ready = False
        try:
            from src.rag.pipeline import get_retriever

            retriever = get_retriever()
            # Check that the retriever has a vectorstore (not an empty fallback)
            if hasattr(retriever, "vectorstore") and retriever.vectorstore is not None:
                rag_ready = True
        except Exception:
            logger.debug("RAG health check failed", exc_info=True)

        # Circuit breaker state: use lock-protected get_state() for accurate
        # reporting.  When CB is open, the system is functionally degraded
        # (all queries return fallback messages) even though the agent is
        # initialized and RAG is accessible.
        cb_state = "unknown"
        try:
            from src.agent.circuit_breaker import _get_circuit_breaker

            cb = _get_circuit_breaker()
            cb_state = await cb.get_state()
        except Exception:
            logger.debug("Circuit breaker state check failed", exc_info=True)

        # CB open means functionally degraded — report as such
        all_healthy = ready and agent_ready and property_loaded and cb_state != "open"
        settings = get_settings()
        body = HealthResponse(
            status="healthy" if all_healthy else "degraded",
            version=settings.VERSION,
            agent_ready=agent_ready,
            property_loaded=property_loaded,
            rag_ready=rag_ready,
            observability_enabled=is_observability_enabled(),
            circuit_breaker_state=cb_state,
            environment=settings.ENVIRONMENT,
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
    # GET /graph — Graph structure for visualization
    # ------------------------------------------------------------------
    # Static fallback for /graph when agent is not initialized or
    # introspection fails.
    _STATIC_GRAPH_STRUCTURE = {
        "nodes": [
            "compliance_gate", "router", "retrieve", "whisper_planner",
            "generate", "validate", "persona_envelope",
            "respond", "fallback", "greeting", "off_topic",
        ],
        "edges": [
            {"from": "__start__", "to": "compliance_gate"},
            {"from": "compliance_gate", "to": "router", "condition": "clean (no guardrail match)"},
            {"from": "compliance_gate", "to": "greeting", "condition": "greeting"},
            {"from": "compliance_gate", "to": "off_topic", "condition": "guardrail triggered"},
            {"from": "router", "to": "retrieve", "condition": "property_qa | hours_schedule | ambiguous"},
            {"from": "router", "to": "greeting", "condition": "greeting"},
            {"from": "router", "to": "off_topic", "condition": "off_topic | gambling_advice | action_request"},
            {"from": "retrieve", "to": "whisper_planner"},
            {"from": "whisper_planner", "to": "generate"},
            {"from": "generate", "to": "validate"},
            {"from": "validate", "to": "persona_envelope", "condition": "PASS"},
            {"from": "validate", "to": "generate", "condition": "RETRY (max 1)"},
            {"from": "validate", "to": "fallback", "condition": "FAIL"},
            {"from": "persona_envelope", "to": "respond"},
            {"from": "respond", "to": "__end__"},
            {"from": "fallback", "to": "__end__"},
            {"from": "greeting", "to": "__end__"},
            {"from": "off_topic", "to": "__end__"},
        ],
    }

    @app.get("/graph", response_model=GraphStructureResponse)
    async def graph_structure(request: Request):
        """Return the StateGraph structure for visualization.

        Introspects the compiled LangGraph agent when available, falling
        back to a static structure when the agent is not initialized.
        """
        agent = getattr(request.app.state, "agent", None)
        if agent is None:
            return _STATIC_GRAPH_STRUCTURE

        try:
            graph_data = agent.get_graph()
            nodes = [
                node.id
                for node in graph_data.nodes
                if node.id not in ("__start__", "__end__")
            ]
            # Sanity check: introspection must return real nodes, otherwise
            # a mock or uninitialized graph would yield empty lists.
            if not nodes:
                return _STATIC_GRAPH_STRUCTURE
            edges = []
            for edge in graph_data.edges:
                edge_dict = {"from": edge.source, "to": edge.target}
                if hasattr(edge, "data") and edge.data:
                    edge_dict["condition"] = str(edge.data)
                edges.append(edge_dict)
            return {"nodes": nodes, "edges": edges}
        except Exception:
            logger.warning("Graph introspection failed, using static structure")
            return _STATIC_GRAPH_STRUCTURE

    # ------------------------------------------------------------------
    # POST /sms/webhook — Telnyx inbound SMS webhook
    # ------------------------------------------------------------------
    @app.post("/sms/webhook")
    async def sms_webhook(request: Request):
        """Telnyx inbound SMS webhook handler with signature verification.

        Returns 404 when SMS_ENABLED=False to prevent the endpoint from being
        reachable when the SMS channel is disabled. Without this guard, an
        attacker could POST to /sms/webhook in production even when SMS is not
        configured (no TELNYX_PUBLIC_KEY = no signature verification).
        """
        from fastapi.responses import JSONResponse

        settings = get_settings()
        if not settings.SMS_ENABLED:
            return JSONResponse(
                content=error_response(ErrorCode.NOT_FOUND, "SMS channel is not enabled."),
                status_code=404,
            )

        from src.sms.webhook import handle_inbound_sms, verify_webhook_signature

        try:
            raw_body = await request.body()

            # Verify Telnyx webhook signature when public key is configured.
            # Prevents attackers from POSTing fabricated SMS events.
            telnyx_public_key = get_settings().TELNYX_PUBLIC_KEY
            if telnyx_public_key:
                signature = request.headers.get("telnyx-signature-ed25519", "")
                timestamp = request.headers.get("telnyx-timestamp", "")
                if not await verify_webhook_signature(
                    raw_body, signature, timestamp, telnyx_public_key,
                ):
                    logger.warning("SMS webhook signature verification failed")
                    return JSONResponse(
                        content=error_response(ErrorCode.UNAUTHORIZED, "Invalid webhook signature."),
                        status_code=401,
                    )

            body = json.loads(raw_body)

            # Parse the Telnyx webhook payload
            event_type = body.get("data", {}).get("event_type", "")

            # Only process inbound messages
            if event_type != "message.received":
                return JSONResponse(content={"status": "ignored"}, status_code=200)

            payload = body.get("data", {}).get("payload", {})
            sms = await handle_inbound_sms(payload)

            # If keyword was already handled by handle_inbound_sms
            if sms.get("type") == "keyword_response":
                return JSONResponse(
                    content={
                        "status": "keyword_handled",
                        "response": sms["keyword_response"],
                    },
                    status_code=200,
                )

            # Regular message — route to agent (Phase 2.4 will handle full routing)
            return JSONResponse(
                content={"status": "received", "from": sms.get("from_", "")},
                status_code=200,
            )
        except Exception:
            logger.exception("SMS webhook error")
            return JSONResponse(
                content=error_response(ErrorCode.INTERNAL_ERROR, "Internal error"),
                status_code=500,
            )

    # ------------------------------------------------------------------
    # POST /cms/webhook — Google Sheets CMS content update
    # ------------------------------------------------------------------
    @app.post("/cms/webhook")
    async def cms_webhook(request: Request):
        """CMS webhook handler for Google Sheets content updates."""
        from fastapi.responses import JSONResponse
        from src.cms.webhook import handle_cms_webhook

        try:
            raw_body = await request.body()
            body = json.loads(raw_body)
            signature = request.headers.get("X-Webhook-Signature", "")
            timestamp = request.headers.get("X-Webhook-Timestamp", "")

            result = await handle_cms_webhook(
                payload=body,
                webhook_secret=get_settings().CMS_WEBHOOK_SECRET.get_secret_value(),
                raw_body=raw_body,
                signature=signature,
                timestamp=timestamp or None,
            )

            status_code = 200 if result["status"] != "rejected" else 403
            return JSONResponse(content=result, status_code=status_code)
        except Exception:
            logger.exception("CMS webhook error")
            return JSONResponse(
                content=error_response(ErrorCode.INTERNAL_ERROR, "Internal error"),
                status_code=500,
            )

    # ------------------------------------------------------------------
    # POST /feedback — User feedback on agent responses
    # ------------------------------------------------------------------
    @app.post("/feedback", response_model=FeedbackResponse)
    async def feedback_endpoint(body: FeedbackRequest):
        """Accept user feedback on agent responses.

        Currently logs feedback with PII redaction. Feedback data is
        available in structured logs for analysis.

        TODO(HEYSEVEN-42): Forward feedback to LangFuse as a score
        for evaluation dashboards and model improvement tracking.
        """
        from src.api.pii_redaction import redact_pii

        logger.info(
            "Feedback received: thread_id=%s rating=%d comment=%s",
            body.thread_id,
            body.rating,
            redact_pii(body.comment) if body.comment else None,
        )
        return FeedbackResponse(status="received", thread_id=body.thread_id)

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
