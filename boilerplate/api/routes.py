"""API routes for the Casino Host Agent.

Organized into versioned route groups. All routes are prefixed with
/api/v1/ for API versioning.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Casino Host API v1"])


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user message to send to the Casino Host agent.",
        examples=["Look up player PLY-482910"],
    )
    thread_id: str | None = Field(
        default=None,
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
        ..., description="The player tracking number.", examples=["PLY-482910"]
    )
    comp_type: str = Field(
        ...,
        description="Type of comp to calculate.",
        examples=["room", "dining", "show", "freeplay", "travel"],
    )


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
    llm_configured: bool
    firestore_connected: bool


# ---------------------------------------------------------------------------
# Agent instance (injected at startup)
# ---------------------------------------------------------------------------

_agent: Any = None


def set_agent(agent: Any) -> None:
    """Inject the compiled LangGraph agent into the routes module.

    Called during application startup in main.py.

    Args:
        agent: A compiled LangGraph StateGraph.
    """
    global _agent
    _agent = agent


def _get_agent() -> Any:
    """Get the current agent instance.

    Raises:
        RuntimeError: If the agent has not been initialized.
    """
    if _agent is None:
        raise RuntimeError(
            "Agent not initialized. Ensure set_agent() is called at startup."
        )
    return _agent


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Send a message to the Casino Host agent.

    Supports multi-turn conversations via thread_id. If no thread_id is
    provided, a new conversation thread is created.

    The agent will autonomously decide whether to use tools (player lookup,
    comp calculation, etc.) based on the message content.
    """
    from langgraph_agent.agent import chat

    agent = _get_agent()

    try:
        result = await chat(
            agent=agent,
            message=request.message,
            thread_id=request.thread_id,
        )
        return ChatResponse(**result)
    except Exception:
        logger.exception("Chat endpoint error")
        raise HTTPException(
            status_code=500,
            detail="Failed to process chat message. Please try again.",
        )


@router.get("/player/{player_id}", response_model=PlayerResponse)
async def get_player(player_id: str) -> PlayerResponse:
    """Look up a player's profile by their tracking number.

    Returns the player's loyalty tier, ADT, visit history, preferences,
    and current status. This is a direct lookup without going through
    the conversational agent.
    """
    from langgraph_agent.tools import check_player_status

    try:
        result = check_player_status.invoke({"player_id": player_id})
        return PlayerResponse(**result)
    except Exception:
        logger.exception("Player lookup error for %s", player_id)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to look up player {player_id}.",
        )


@router.post("/comp/calculate")
async def calculate_comp(request: CompRequest) -> dict:
    """Calculate an eligible comp for a player.

    Uses the casino's reinvestment matrix to determine what comp value
    a player qualifies for based on their theoretical win and loyalty tier.
    """
    from langgraph_agent.tools import calculate_comp as calc_comp_tool

    try:
        result = calc_comp_tool.invoke(
            {"player_id": request.player_id, "comp_type": request.comp_type}
        )
        return result
    except Exception:
        logger.exception(
            "Comp calculation error for %s/%s",
            request.player_id,
            request.comp_type,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to calculate comp.",
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for Cloud Run and load balancers.

    Returns the service status, version, and connectivity to external
    dependencies (LLM, Firestore).
    """
    import os

    llm_configured = bool(os.environ.get("GOOGLE_API_KEY"))
    firestore_connected = False

    try:
        from google.cloud import firestore  # type: ignore[import-untyped]

        db = firestore.Client()
        db.collection("_health").document("ping").get()
        firestore_connected = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy",
        version=os.environ.get("APP_VERSION", "0.1.0"),
        llm_configured=llm_configured,
        firestore_connected=firestore_connected,
    )
