# Round 3 Gemini 3 Pro Review — Error Handling & Resilience Spotlight

**Reviewer**: Gemini 3 Pro (thinking_level=high)
**Commit**: 014335c
**Previous Scores**: R1 Average=67.3 | R2 Average=61.3
**Spotlight**: ERROR HANDLING & RESILIENCE (+1 severity on error handling findings)

---

## Scorecard

| Dimension | Score (0-10) | Notes |
|:---|:---|:---|
| 1. Graph/Agent Architecture | 7 | PII data loss on error; logic is otherwise sound. |
| 2. RAG Pipeline | 6 | Monkey-patched caching and silent purge failures degrade reliability. |
| 3. Data Model / State Design | **2** | **Catastrophic**. Recreating Firestore clients per request is a denial-of-service vulnerability. |
| 4. API Design | 5 | SSE heartbeats block on generation; middleware risks protocol violations. |
| 5. Testing Strategy | 9 | Excellent coverage, masking the architectural rot. |
| 6. Docker & DevOps | 8 | Solid config, though secrets handling needs tightening. |
| 7. Prompts & Guardrails | 9 | The strongest part of the stack. |
| 8. Scalability & Production | **1** | **Unusable**. Unbounded memory leaks and connection exhaustion ensure crashes under load. |
| 9. Documentation & Code Quality | 7 | Readable, but comments often describe flawed logic. |
| 10. Domain Intelligence | 8 | Good modeling of the casino host persona. |
| **TOTAL** | **62** | **NOT PRODUCTION READY** |

---

## Findings

| # | File & Line | Severity | Finding | Recommendation | Consensus/Unique |
|:---|:---|:---|:---|:---|:---|
| 1 | `src/data/guest_profile.py` (All methods) | **CRITICAL** | **Connection Exhaustion Vulnerability**. `_get_firestore_client()` instantiates `firestore.AsyncClient()` inside every method (`get`, `update`, `delete`) with no caching or pooling. Under load, this will exhaust file descriptors and SSL handshakes, crashing the app. | Implement a singleton `AsyncClient` managed by the app lifespan or dependency injection. Never create DB clients per-request. | Consensus |
| 2 | `src/agent/agents/_base.py` (Line 161) | **CRITICAL** | **Incomplete LLM Error Handling**. The `except` block only catches `httpx.HTTPError, asyncio.TimeoutError, ConnectionError`. It misses `google.api_core.exceptions` (ServiceUnavailable, GoogleAPICallError, etc.). When Vertex AI throws a platform error, the specialist crashes unhandled, propagating up to the graph and potentially killing the SSE stream. | Import and catch vendor-specific base exceptions (e.g., `GoogleAPICallError`) or add a broad `except Exception` fallback after the specific catches that still records circuit breaker failure. | Consensus |
| 3 | `src/cms/webhook.py` (Global Scope) | **CRITICAL** | **Unbounded Memory Leak**. `_content_hashes: dict[str, str] = {}` is a module-level dictionary that never clears, has no TTL, and no size cap. In a long-running container receiving frequent CMS updates, this guarantees an eventual OOM Kill. | Use `cachetools.TTLCache` or an external store (Redis) with expiration. Set a reasonable maxsize. | Consensus |
| 4 | `src/data/guest_profile.py` (Lines 228-261) | **CRITICAL** | **Non-Atomic CCPA Deletion**. The cascade delete (conversations → messages → behavioral signals → audit log de-identification → guest document) is NOT transactional. If the audit log update fails mid-cascade, the profile is left partially deleted — zombie data that violates privacy laws. The `except Exception: raise` at line 259 means a mid-cascade failure leaves the profile in a partially deleted state with no rollback. | Wrap the delete operations in a `client.transaction()` or Firestore batch write. Either all operations complete, or none do. At minimum, reverse the order: de-identify audit logs first, then delete data. | Consensus |
| 5 | `src/agent/agents/_base.py` (Line 137) | **HIGH** | **Premature Circuit Breaker Success**. `cb.record_success()` is called immediately after `llm.ainvoke()` returns, BEFORE the content is validated or the response is confirmed usable. If the LLM returns invalid garbage, hallucinations, or empty content, the circuit breaker counts it as "success," failing to protect the system during partial semantic outages where the LLM responds but with unusable content. | Move `record_success()` to after content extraction and basic sanity checks (e.g., non-empty content). Only record success when the output is actually usable. | Unique |
| 6 | `src/api/app.py` (SSE Loop, Lines 166-187) | **HIGH** | **Heartbeat Starvation**. The heartbeat check runs inside the `async for event in chat_stream(...)` loop. If `chat_stream()` hangs during LLM generation (waiting for the first token, which can take 5-30 seconds), the heartbeat loop is blocked because no events are yielded. The heartbeat is only checked between events, not during long waits between events. | Run the heartbeat in a separate `asyncio.Task` that pushes ping events to a shared `asyncio.Queue`, merged with the chat stream. Alternatively, use `asyncio.wait()` with a timeout to interleave heartbeat checks between event waits. | Unique |
| 7 | `src/agent/graph.py` (Line 633) | **HIGH** | **PII Buffer Data Loss on Error**. When `errored=True`, the PII buffer is explicitly NOT flushed: `if _pii_buffer and not errored`. This means the final chunk of text before the crash (potentially containing the user's answer or partial response) is silently discarded. The user sees an error with no partial content. | Attempt to sanitize and flush the buffer inside the error handler before sending the error event. If the buffer contains PII, redact it; if not, emit it. Only discard if redaction itself fails. | Unique |

---

## Summary

The system is a fragile glass cannon. The **Prompts & Guardrails** (9/10) and **Testing** (9/10) suggest a high-quality application on the surface, but the **Scalability & Production** (1/10) and **Data Model / State Design** (2/10) betray fundamental production engineering gaps.

The resilience strategy is superficial:
- You have a circuit breaker, but it records success prematurely (Finding #5).
- You have error handling in specialists, but it misses the specific exceptions your vendor throws (Finding #2).
- You have SSE streaming with heartbeats, but the heartbeat mechanism blocks on the very thing it's supposed to protect against (Finding #6).
- Most critically, the database pattern in `guest_profile.py` creates a new Firestore client on every single CRUD call — a literal textbook example of "how to crash a cloud service under load" (Finding #1).

The code is well-documented and the domain modeling is strong, but documentation quality does not compensate for architectural defects. The system is safer against prompt injection than it is against its own resource management.

**Round 3 Verdict:** Immediate refactoring of database client lifecycles, exception hierarchies, and unbounded in-memory stores is required before this code can be considered production-ready for a regulated casino environment.
