# Output Guardrails Architecture

Hey Seven implements 4 layers of output protection, applied in order from most safety-critical to least:

## Layer 1: Validation Loop (LLM-based)

- `validate_node` reviews every LLM-generated response against 6 compliance criteria
- Uses structured output (`ValidationResult`) with `Literal["PASS", "FAIL", "RETRY"]`
- Max 1 retry before fallback (bounded by `GRAPH_RECURSION_LIMIT=10` as hard ceiling)
- Separate validator LLM with `temperature=0.0` for deterministic classification
- Degraded-pass strategy: first attempt + validator failure = PASS (availability); retry attempt + validator failure = FAIL (safety)
- See: `src/agent/nodes.py:validate_node`, `src/agent/nodes.py:_degraded_pass_result`

## Layer 2: PII Redaction (regex-based, fail-closed)

- `StreamingPIIRedactor` processes tokens in real-time during SSE streaming
- Catches phone numbers, SSNs, credit cards, emails across chunk boundaries
- Buffer window ensures PII spanning multiple SSE tokens is still caught
- Non-streaming paths (greeting, off_topic) use `redact_pii()` directly
- **Fails CLOSED**: on regex error, returns `[PII_REDACTION_ERROR]` placeholder — never passes through original text
- See: `src/agent/streaming_pii.py`, `src/api/pii_redaction.py`

## Layer 3: Persona Envelope (branding enforcement)

- `persona_envelope_node` enforces brand voice consistency
- Exclamation limit (max 2 per response)
- Emoji removal (casino brand voice is professional, not casual)
- Guest name personalization (only if name not already present in response)
- SMS length truncation when `PERSONA_MAX_CHARS > 0`
- **Processing order**: PII redaction -> branding -> name injection -> truncation (safety-critical first)
- See: `src/agent/persona.py`

## Layer 4: Response Formatting (respond_node)

- Final response extraction and source citation
- Sources logged for traceability (categories from retrieved context metadata)
- Clears `retry_feedback` and `retrieved_context` before checkpoint write to prevent stale data accumulation in Firestore
- See: `src/agent/nodes.py:respond_node`

## How Layers Interact

```
generate_node (specialist agent)
    |
    v
validate_node (Layer 1: LLM compliance check)
    |-- PASS --> persona_envelope_node (Layer 3: branding)
    |-- RETRY --> generate_node (with feedback, max 1 retry)
    |-- FAIL --> fallback_node (safe contact info)
    |
    v
respond_node (Layer 4: source citation + cleanup)
    |
    v
SSE streaming with StreamingPIIRedactor (Layer 2: real-time PII scrubbing)
```

**Note**: PII redaction (Layer 2) executes during SSE streaming, which is _after_ the graph nodes complete. This means PII redaction happens on the final output regardless of which path the graph took (validated response, retry response, or fallback).
