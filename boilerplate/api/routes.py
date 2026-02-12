"""API routes for the Casino Host Agent.

All routes are prefixed with ``/api/v1/`` for versioning. Authentication
is enforced via a dependency on ``verify_api_key``.
"""

import hmac
import logging
import os
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Casino Host API v1"])


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str:
    """Validate the ``X-API-Key`` header against the configured secret.

    If ``API_KEY`` env var is unset the service rejects ALL requests --
    fail-closed, not fail-open.

    Raises:
        HTTPException: 401 if the key is missing or incorrect.
    """
    expected = os.getenv("API_KEY")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="API key not configured on server.",
        )
    if not api_key or not hmac.compare_digest(api_key, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )
    return api_key


# Alias for use as a route dependency
ApiKey = Annotated[str, Depends(verify_api_key)]


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

# Valid comp types (closed set)
CompType = Literal["room", "dining", "entertainment", "freeplay", "cashback"]


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The user message to send to the Casino Host agent.",
        examples=["Look up player PLY-482910"],
    )
    thread_id: str | None = Field(
        default=None,
        pattern=r"^[a-zA-Z0-9_-]{1,128}$",
        description=(
            "Conversation thread ID for multi-turn conversations. "
            "If None, a new thread is created."
        ),
    )


class ChatResponse(BaseModel):
    """Response body from the chat endpoint."""

    response: str = Field(description="The agent's text response.")
    thread_id: str = Field(description="Conversation thread ID for continuation.")
    player_id: str | None = Field(
        default=None,
        description="Currently discussed player ID, if identified.",
    )
    escalation: bool = Field(
        default=False,
        description="Whether the case was escalated to a human host.",
    )
    compliance_flags: list[str] = Field(
        default_factory=list,
        description="Active compliance flags, if any.",
    )


class CompRequest(BaseModel):
    """Request body for the comp calculation endpoint."""

    player_id: str = Field(
        ...,
        pattern=r"^[a-zA-Z0-9_-]{1,64}$",
        description="The player tracking number.",
        examples=["PLY-482910"],
    )
    comp_type: CompType = Field(
        ...,
        description="Type of comp to calculate.",
        examples=["room", "dining"],
    )


class CompResponse(BaseModel):
    """Response body for comp calculation."""

    player_id: str
    comp_type: str
    eligible: bool = False
    comp_value: float | None = None
    currency: str = "USD"
    details: str | None = None


class PlayerResponse(BaseModel):
    """Response body for player lookup."""

    player_id: str
    name: str | None = None
    tier: str | None = None
    adt: float | None = None
    comp_balance: float | None = None
    last_visit: str | None = None
    visit_count_ytd: int | None = None
    preferences: dict[str, str] | None = None
    host_assigned: str | None = None
    status: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Response body for health check."""

    status: str
    version: str
    agent_ready: bool


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def chat_endpoint(
    body: ChatRequest,
    request: Request,
    _key: ApiKey,
) -> ChatResponse:
    """Send a message to the Casino Host agent.

    Supports multi-turn conversations via ``thread_id``. If no thread_id is
    provided, a new conversation thread is created.
    """
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Service is starting up.",
        )

    try:
        from langgraph_agent.agent import chat

        result = await chat(
            agent=agent,
            message=body.message,
            thread_id=body.thread_id,
        )
        return ChatResponse(**result)
    except Exception:
        logger.exception("Chat endpoint error")
        raise HTTPException(
            status_code=500,
            detail="Failed to process chat message. Please try again.",
        )


@router.get(
    "/player/{player_id}",
    response_model=PlayerResponse,
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def get_player(
    player_id: str,
    _key: ApiKey,
) -> PlayerResponse:
    """Look up a player's profile by their tracking number.

    Returns the player's loyalty tier, ADT, visit history, preferences,
    and current status.
    """
    # Validate player_id format
    import re

    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", player_id):
        raise HTTPException(
            status_code=422,
            detail="Invalid player_id format. Alphanumeric, hyphens, and underscores only.",
        )

    try:
        from langgraph_agent.tools import check_player_status

        result = check_player_status.invoke({"player_id": player_id})
        return PlayerResponse(**result)
    except Exception:
        logger.exception("Player lookup error for %s", player_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to look up player.",
        )


@router.post(
    "/comp/calculate",
    response_model=CompResponse,
    status_code=200,
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def calculate_comp(
    body: CompRequest,
    _key: ApiKey,
) -> CompResponse:
    """Calculate an eligible comp for a player.

    Uses the casino's reinvestment matrix to determine what comp value
    a player qualifies for based on their theoretical win and loyalty tier.
    """
    try:
        from langgraph_agent.tools import calculate_comp as calc_comp_tool

        result = calc_comp_tool.invoke(
            {"player_id": body.player_id, "comp_type": body.comp_type}
        )
        return CompResponse(
            player_id=body.player_id,
            comp_type=body.comp_type,
            **result,
        )
    except Exception:
        logger.exception(
            "Comp calculation error for %s/%s",
            body.player_id,
            body.comp_type,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to calculate comp.",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
)
async def health_check(request: Request) -> HealthResponse | JSONResponse:
    """Health check endpoint for Cloud Run and load balancers.

    Returns 503 if the agent is not initialized (service is unhealthy).
    No authentication required -- load balancers need unauthenticated access.
    """
    version = os.getenv("APP_VERSION", "0.1.0")
    agent = getattr(request.app.state, "agent", None)

    if agent is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "version": version,
                "agent_ready": False,
            },
        )

    return HealthResponse(
        status="healthy",
        version=version,
        agent_ready=True,
    )
