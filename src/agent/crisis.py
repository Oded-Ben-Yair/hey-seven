"""Crisis detection and escalation protocol for casino guest interactions.

Graduated crisis detection system that classifies guest messages into
four levels based on language patterns indicating problem gambling,
intoxication, financial distress, or emotional crisis.

Levels:
    none     — No crisis indicators detected.
    concern  — Mild indicators (chasing losses, frustration with gambling).
               Response: empathy + gentle resource mention.
    urgent   — Strong indicators (suicidal language, self-harm references,
               financial desperation, "can't face" language).
               Response: direct resource provision + offer to connect.
    immediate — Explicit crisis language (active suicidal ideation, immediate
               danger, "going to hurt myself").
               Response: stop conversation, provide crisis resources, flag.

R72 Phase C5: Based on NCPG intervention protocols and casino responsible
gaming officer escalation paths from domain research.

Integration: Called from compliance_gate_node BEFORE existing self_harm
detection, providing graduated response instead of binary detect/don't-detect.
"""

import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

__all__ = [
    "detect_crisis_level",
    "CrisisLevel",
    "get_crisis_response_es",
    "get_crisis_followup_es",
]

CrisisLevel = Literal["none", "concern", "urgent", "immediate"]

# ---------------------------------------------------------------------------
# Pattern sets by severity (checked in order: immediate → urgent → concern)
# ---------------------------------------------------------------------------

# IMMEDIATE: Active crisis — requires stopping all other conversation
_IMMEDIATE_PATTERNS: list[re.Pattern] = [
    # Active suicidal ideation
    re.compile(
        r"(?i)\b(?:going to|gonna|want to|I'?m going to)\s+(?:kill|end|hurt)\s+(?:myself|my life)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\b(?:end(?:ing)?\s+(?:it|my life|things)|take my (?:own )?life)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\b(?:don'?t\s+want\s+to\s+(?:live|be\s+alive|exist)|rather\s+(?:be\s+dead|die))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\b(?:no\s+reason\s+to\s+(?:live|go\s+on)|nothing\s+left\s+to\s+live\s+for)\b",
        re.IGNORECASE,
    ),
    # Immediate danger
    re.compile(
        r"(?i)\b(?:I'?m\s+(?:on|at)\s+(?:the|a)\s+(?:bridge|ledge|roof|edge))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\b(?:saying\s+goodbye|this\s+is\s+goodbye|writing\s+my\s+note)\b",
        re.IGNORECASE,
    ),
    # R77 fix: Spanish immediate crisis patterns
    re.compile(
        r"(?i)\b(?:me\s+)?(?:voy|quiero)\s+a?\s*(?:matar(?:me)?|quitar(?:me)?\s+la\s+vida)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\bno\s+quiero\s+(?:vivir|seguir\s+(?:viviendo|aquí))\b", re.IGNORECASE
    ),
    re.compile(
        r"(?i)\b(?:mejor\s+(?:muerto|muerta)|(?:me\s+)?(?:quiero|voy\s+a)\s+morir)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?i)\bacabar\s+con\s+todo\b", re.IGNORECASE),
]

# URGENT: Strong distress — requires direct resource provision
_URGENT_PATTERNS: list[re.Pattern] = [
    # Financial desperation linked to gambling
    re.compile(
        r"(?i)\b(?:can'?t\s+face\s+my\s+(?:wife|husband|family|partner|kids))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\b(?:lost\s+everything|ruined\s+(?:my|our)\s+(?:life|lives|marriage|family))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\b(?:no\s+money\s+(?:for|to)\s+(?:get\s+home|hotel|flight|rent|food))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\b(?:stranded|stuck\s+here|can'?t\s+(?:get\s+home|afford\s+(?:to\s+leave|a\s+cab)))\b",
        re.IGNORECASE,
    ),
    # Self-harm references (non-immediate)
    re.compile(
        r"(?i)\b(?:hurt(?:ing)?\s+myself|self[- ]harm|cutting|harm\s+myself)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\b(?:don'?t\s+(?:care|see\s+the\s+point)\s+anymore)\b", re.IGNORECASE
    ),
    re.compile(
        r"(?i)\b(?:what'?s\s+the\s+point|give\s+up|given\s+up)\b", re.IGNORECASE
    ),
    # Substance abuse crisis
    re.compile(
        r"(?i)\b(?:too\s+drunk\s+to|can'?t\s+(?:stand|walk|see|think)\s+straight)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\b(?:blacked?\s+out|passed?\s+out|need\s+(?:medical|an?\s+ambulance))\b",
        re.IGNORECASE,
    ),
    # R77 fix: Spanish urgent crisis patterns
    re.compile(r"(?i)\b(?:no\s+puedo\s+m[aá]s|ya\s+no\s+aguanto)\b", re.IGNORECASE),
    re.compile(
        r"(?i)\b(?:perd[ií]\s+todo|arruiné\s+(?:mi|nuestra)\s+(?:vida|familia|matrimonio))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\bno\s+tengo\s+(?:dinero|plata)\s+(?:para|ni\s+para)\b", re.IGNORECASE
    ),
    re.compile(r"(?i)\b(?:hacerme|me\s+hago)\s+daño\b", re.IGNORECASE),
    re.compile(r"(?i)\bno\s+(?:le\s+)?veo\s+sentido\b", re.IGNORECASE),
]

# CONCERN: Mild distress — empathy + gentle resource mention
_CONCERN_PATTERNS: list[re.Pattern] = [
    # Chasing losses (problem gambling indicator)
    re.compile(
        r"(?i)\b(?:just\s+need\s+one\s+more\s+(?:win|hand|spin|bet))\b", re.IGNORECASE
    ),
    re.compile(
        r"(?i)\b(?:win\s+it\s+(?:all\s+)?back|make\s+it\s+back|double\s+(?:down|or\s+nothing))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\b(?:lend\s+me|borrow|loan\s+me|(?:need|get|give\s+me)\s+(?:a\s+)?(?:marker|credit|advance))\b",
        re.IGNORECASE,
    ),
    # Extended session indicators
    re.compile(
        r"(?i)\b(?:been\s+(?:here|playing|at\s+(?:the\s+)?(?:table|slots?))\s+(?:all\s+(?:day|night)|for\s+\d+\s+hours?))\b",
        re.IGNORECASE,
    ),
    # General distress about gambling
    re.compile(
        r"(?i)\b(?:can'?t\s+stop|keep\s+(?:losing|going\s+back)|problem\s+(?:with\s+)?gambl)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\b(?:addicted|addiction|gambling\s+problem|out\s+of\s+control)\b",
        re.IGNORECASE,
    ),
    # Financial stress (non-desperate)
    re.compile(
        r"(?i)\b(?:spent\s+(?:too\s+much|more\s+than|all\s+my)|over\s+budget|maxed\s+out)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\b(?:shouldn'?t\s+(?:have\s+)?(?:bet|gambled|played)\s+(?:that\s+much|so\s+much))\b",
        re.IGNORECASE,
    ),
    # R77 fix: Spanish concern patterns
    re.compile(
        r"(?i)\b(?:necesito\s+(?:recuperar|ganar)\s+(?:lo\s+que\s+)?perd[ií]|recuperar\s+mi\s+dinero)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\b(?:llevo\s+(?:todo\s+el\s+d[ií]a|toda\s+la\s+noche|horas?)\s+(?:aqu[ií]|jugando))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\b(?:gast[eé]\s+(?:demasiado|todo|m[aá]s\s+de\s+lo\s+que))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\bno\s+(?:puedo|deber[ií]a)\s+(?:parar|dejar)\s+de\s+jugar\b",
        re.IGNORECASE,
    ),
]


def detect_crisis_level(text: str) -> CrisisLevel:
    """Detect crisis severity level from guest message text.

    Checks patterns in severity order (immediate → urgent → concern)
    and returns the highest matching level. Returns "none" when no
    crisis indicators are detected.

    Args:
        text: The guest's message text.

    Returns:
        One of: "none", "concern", "urgent", "immediate".
        Returns "none" on any error (fail-silent for non-safety paths).
    """
    if not text or not isinstance(text, str):
        return "none"

    try:
        # Check highest severity first
        for pattern in _IMMEDIATE_PATTERNS:
            if pattern.search(text):
                logger.warning("CRISIS IMMEDIATE detected: %s", text[:100])
                return "immediate"

        for pattern in _URGENT_PATTERNS:
            if pattern.search(text):
                logger.warning("CRISIS URGENT detected: %s", text[:100])
                return "urgent"

        for pattern in _CONCERN_PATTERNS:
            if pattern.search(text):
                logger.info("Crisis concern detected: %s", text[:100])
                return "concern"

        return "none"

    except Exception:
        logger.debug("Crisis detection failed, returning none", exc_info=True)
        return "none"


# ---------------------------------------------------------------------------
# Spanish-localized crisis resources (Phase 1: Multilingual)
# ---------------------------------------------------------------------------
# 988 Lifeline: Spanish service available — press 2 for Spanish or say "español"
# Crisis Text Line: Spanish-speaking counselors available — text HOLA to 741741
# These are verified US national resources with confirmed Spanish support.


def get_crisis_response_es(property_name: str, property_phone: str) -> str:
    """Build Spanish-localized crisis response with verified resources.

    Provides 988 Suicide & Crisis Lifeline (Spanish option) and Crisis Text
    Line (Spanish counselors via HOLA keyword). Both are verified US national
    resources with confirmed Spanish language support.

    Args:
        property_name: Display name of the casino property.
        property_phone: Contact phone for the property.

    Returns:
        Spanish crisis response with 988 Lifeline and Crisis Text Line info.
    """
    return (
        "Puedo escuchar que estás pasando por un momento muy difícil, y quiero "
        "que sepas que hay ayuda disponible ahora mismo.\n\n"
        "**Por favor comunícate con estos recursos confidenciales:**\n\n"
        "- **988 Línea de Prevención del Suicidio y Crisis**: Llama al **988** "
        "(presiona 2 para español) o llama directamente al **1-888-628-9454** "
        "(línea en español, 24/7, gratis, confidencial)\n"
        "- **988 por Texto**: Envía **AYUDA** al **988**\n"
        "- **Línea de Crisis por Texto**: Envía **HOLA** al **741741** "
        "(consejeros en español disponibles)\n"
        "- **Emergencia**: Llama al **911** si tú o alguien está en peligro inmediato\n\n"
        "No tienes que enfrentar esto solo/a. Hay consejeros capacitados disponibles "
        "ahora mismo que entienden lo que estás pasando y pueden ayudar.\n\n"
        f"Si deseas hablar con alguien de {property_name} en persona, "
        f"cualquier miembro del equipo puede conectarte con servicios de apoyo. "
        f"También puedes llamarnos al {property_phone}."
    )


def get_crisis_followup_es(
    user_message: str,
    property_name: str,
    property_phone: str,
    crisis_turn_count: int = 2,
) -> str:
    """Turn 2+ Spanish crisis response — empathetic, shorter, acknowledges guest's words.

    Args:
        user_message: The guest's current message (used for on-site help detection).
        property_name: Display name of the casino property.
        property_phone: Contact phone for the property.
        crisis_turn_count: Current crisis turn (2 = first followup, 3+ = variation).

    Returns:
        Spanish crisis followup response with key resource reminder.
    """
    msg_lower = user_message.lower()
    if any(
        kw in msg_lower
        for kw in (
            "alguien aquí",
            "hablar con alguien",
            "en persona",
            "cara a cara",
            "someone here",
            "talk to someone",
            "in person",
        )
    ):
        return (
            f"Sí — cualquier miembro del equipo de {property_name} puede conectarte "
            f"con servicios de apoyo ahora mismo. También puedes ir a la recepción "
            f"o llamar al {property_phone} y pedir hablar con alguien de servicios "
            "al huésped.\n\n"
            "La **Línea 988** (llama o envía texto al 988, presiona 2 para español) "
            "también está siempre disponible si deseas hablar con un consejero capacitado."
        )
    # R90: Turn 3+ variation to avoid verbatim repetition across crisis turns.
    if crisis_turn_count >= 3:
        return (
            "Sigo aquí. No tienes que pasar por esto solo/a. "
            "¿Te ayudaría hablar con alguien en persona? Cualquier miembro "
            "del equipo puede conectarte ahora mismo.\n\n"
            "La **Línea 988** (llama o envía texto al 988, presiona 2 para español) "
            "sigue disponible las 24 horas."
        )
    return (
        "Te escucho, y lo que estás pasando es real. No tienes que "
        "enfrentar esto solo/a.\n\n"
        "Un consejero capacitado en la **Línea 988** puede ayudar — llama o envía "
        "texto al **988** (presiona 2 para español), las 24 horas, los 7 días. "
        "Es gratis y confidencial.\n\n"
        f"Si deseas hablar con alguien de {property_name} en persona, cualquier "
        "miembro del equipo puede ayudarte."
    )
