# ADR-014: Message Limits (40 Total, 20 History Window)

## Status
Accepted

## Context
Long conversations cause: (1) unbounded memory growth in state, (2) increasing LLM token costs, (3) context window overflow. Two limits control this.

## Decision
- `MAX_MESSAGE_LIMIT=40`: Total messages (human + AI) before conversation end
- `MAX_HISTORY_MESSAGES=20`: Sliding window -- only last 20 messages sent to LLM

## Consequences
- Conversation history stays bounded (no unbounded growth)
- LLM sees sufficient context for multi-turn reasoning
- Older messages are checkpointed but not sent to LLM
- Guest must start a new thread after 40 messages
