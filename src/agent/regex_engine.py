"""re2/re adapter for ReDoS-safe regex compilation.

Tries google-re2 first (linear-time guarantee), falls back to stdlib re
for patterns using features unsupported by re2 (e.g., lookaheads).
Only 1 of 204 guardrail patterns uses (?!...) lookahead (line 54 of
guardrails.py), so 203/204 patterns get re2 protection.

google-re2 uses its own Options object instead of stdlib re flags.
This adapter translates re.IGNORECASE and re.DOTALL into the
corresponding re2.Options fields (case_sensitive, dot_nl).
"""

import logging
import re as _stdlib_re

logger = logging.getLogger(__name__)

try:
    import re2 as _re2
    RE2_AVAILABLE = True
    logger.info("google-re2 available — using linear-time regex engine")
except ImportError:
    _re2 = None
    RE2_AVAILABLE = False
    logger.warning("google-re2 not available — falling back to stdlib re (ReDoS risk)")


def _flags_to_re2_options(flags: int) -> "_re2.Options | None":
    """Translate stdlib re flags to a re2.Options object.

    Only re.IGNORECASE and re.DOTALL are used by guardrails.py patterns.
    Returns None if re2 is not available.
    """
    if _re2 is None:
        return None
    opts = _re2.Options()
    if flags & _stdlib_re.IGNORECASE:
        opts.case_sensitive = False
    if flags & _stdlib_re.DOTALL:
        opts.dot_nl = True
    return opts


def is_re2_active() -> bool:
    """Check if re2 is being used for pattern compilation.

    Returns True if google-re2 is installed and active. Use in health checks
    and startup logs to surface ReDoS risk in production.
    """
    return RE2_AVAILABLE


def enforce_re2_in_production() -> None:
    """Fail fast if RE2 is unavailable in non-development environments.

    Call at application startup (FastAPI lifespan) to prevent deploying
    with 204 guardrail patterns vulnerable to ReDoS attacks.

    Raises:
        RuntimeError: If ENVIRONMENT != 'development' and google-re2 is not installed.
    """
    from src.config import get_settings

    settings = get_settings()
    if settings.ENVIRONMENT != "development" and not RE2_AVAILABLE:
        raise RuntimeError(
            "google-re2 is required in production (ENVIRONMENT="
            f"'{settings.ENVIRONMENT}'). Install with: pip install google-re2. "
            "Without re2, all 204 guardrail patterns are vulnerable to ReDoS."
        )


# Track stdlib fallback count for observability
_stdlib_fallback_count = 0


def compile(pattern: str, flags: int = 0) -> "_stdlib_re.Pattern":
    """Compile a regex pattern, preferring re2 for linear-time guarantees.

    Falls back to stdlib re if:
    - re2 is not installed (logs WARNING at import time)
    - The pattern uses re2-unsupported features (lookaheads, backreferences)

    In production, re2 MUST be available. The fallback exists only for
    local development without libre2-dev. Deploy with google-re2 installed
    to prevent ReDoS attacks on the 204 guardrail patterns.

    Args:
        pattern: Regex pattern string.
        flags: Regex flags (re.I, re.DOTALL, etc.).

    Returns:
        Compiled pattern object (re2.Pattern or re.Pattern).
    """
    global _stdlib_fallback_count
    if _re2 is not None:
        try:
            opts = _flags_to_re2_options(flags)
            return _re2.compile(pattern, opts)
        except Exception:
            # re2 doesn't support lookaheads, backreferences, etc.
            # Log at WARNING (not debug) so operators know which patterns
            # use stdlib re and are potentially vulnerable to ReDoS.
            _stdlib_fallback_count += 1
            logger.warning(
                "Pattern incompatible with re2 (fallback #%d), using stdlib re: %s",
                _stdlib_fallback_count,
                pattern[:60],
            )
    return _stdlib_re.compile(pattern, flags)
