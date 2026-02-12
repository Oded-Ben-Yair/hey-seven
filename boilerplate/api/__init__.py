"""FastAPI application for the Casino Host Agent.

Exposes REST and WebSocket endpoints for the agent, player lookups,
comp calculations, and health checks.
"""

from .main import app, create_app
from .routes import router

__all__ = ["app", "create_app", "router"]
