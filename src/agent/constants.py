"""Node name constants for the 11-node Property Q&A StateGraph.

Single source of truth for node names used across graph.py, nodes.py,
tests, and any other module referencing graph topology. Prevents
silent breakage from string typos or renames.
"""

NODE_ROUTER = "router"
NODE_RETRIEVE = "retrieve"
NODE_GENERATE = "generate"
NODE_VALIDATE = "validate"
NODE_RESPOND = "respond"
NODE_FALLBACK = "fallback"
NODE_GREETING = "greeting"
NODE_OFF_TOPIC = "off_topic"
NODE_COMPLIANCE_GATE = "compliance_gate"
NODE_PERSONA = "persona_envelope"
NODE_WHISPER = "whisper_planner"

_NON_STREAM_NODES = frozenset({
    NODE_GREETING, NODE_OFF_TOPIC, NODE_FALLBACK,
    NODE_COMPLIANCE_GATE, NODE_PERSONA, NODE_WHISPER,
})

_KNOWN_NODES = frozenset({
    NODE_ROUTER, NODE_RETRIEVE, NODE_GENERATE, NODE_VALIDATE,
    NODE_RESPOND, NODE_FALLBACK, NODE_GREETING, NODE_OFF_TOPIC,
    NODE_COMPLIANCE_GATE, NODE_PERSONA, NODE_WHISPER,
})
