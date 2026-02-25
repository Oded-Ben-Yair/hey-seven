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


def compile(pattern: str, flags: int = 0) -> "_stdlib_re.Pattern":
    """Compile a regex pattern, preferring re2 for linear-time guarantees.

    Falls back to stdlib re if:
    - re2 is not installed
    - The pattern uses re2-unsupported features (lookaheads, backreferences)

    Args:
        pattern: Regex pattern string.
        flags: Regex flags (re.I, re.DOTALL, etc.).

    Returns:
        Compiled pattern object (re2.Pattern or re.Pattern).
    """
    if _re2 is not None:
        try:
            opts = _flags_to_re2_options(flags)
            return _re2.compile(pattern, opts)
        except Exception:
            # re2 doesn't support lookaheads, backreferences, etc.
            # Fall back silently to stdlib re for these patterns.
            logger.debug(
                "Pattern incompatible with re2, falling back to stdlib re: %s",
                pattern[:60],
            )
    return _stdlib_re.compile(pattern, flags)
