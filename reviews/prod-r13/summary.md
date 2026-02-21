# R13 Summary

## Consensus Findings (2/3+ agreement)

1. **CB record_failure() missing in _dispatch_to_specialist** -- Models: DeepSeek (F-001 HIGH), GPT-5.2 (F-011 MEDIUM). The dispatch LLM exception handlers caught errors and fell back to keyword counting but never called `cb.record_failure()`, leaving the circuit breaker unable to detect sustained dispatch LLM failures.

2. **SSE heartbeat structurally broken** -- Models: Gemini (F-001 HIGH), GPT-5.2 (F-001 MEDIUM). The heartbeat check ran inside `async for event in chat_stream(...)`, meaning it could only fire when a new event was already available. During long inter-event gaps (15-30s for first LLM token), no heartbeat was sent, defeating its purpose of preventing EventSource reconnection.

3. **Streaming PII boundary misalignment** -- Models: DeepSeek (F-003 MEDIUM), Gemini (F-004 MEDIUM). `_scan_and_release(force=False)` applied `redact_pii()` to the full buffer and emitted `redacted[:-40]` as safe text, but retained `self._buffer[-40:]` from the ORIGINAL (unredacted) buffer. Since redaction changes string lengths, the safe/lookahead split operated on different text representations, potentially allowing fragmented PII to leak across SSE chunks.

4. **CMS webhook missing test coverage** -- Models: Gemini (F-003 HIGH), GPT-5.2 (F-004 HIGH). Zero HTTP-level tests in test_api.py for the `/cms/webhook` endpoint despite handling signature verification, payload validation, hash-based change detection, and re-indexing.

5. **/feedback endpoint no tests in test_api.py** -- Models: Gemini (F-002 HIGH), GPT-5.2 (F-005 MEDIUM). While test_phase4_integration.py had good coverage, test_api.py (the canonical API test module) had zero tests for the endpoint, breaking the pattern of per-endpoint test classes in that file.

6. **/feedback docstring claims LangFuse forwarding that does not exist** -- Models: DeepSeek (F-004 MEDIUM), GPT-5.2 (F-005 MEDIUM). Docstring said "feedback is forwarded to LangFuse as a score" but no such forwarding was implemented. Violates documentation honesty vocabulary rule: overclaiming "implemented" for scaffolded behavior.

## Fixes Applied (6/7)

1. **CB record_failure() in dispatch exception handlers** -- File: `src/agent/graph.py` (lines 237-247). Added `await cb.record_failure()` in both `(ValueError, TypeError)` and broad `Exception` handlers so the circuit breaker tracks dispatch LLM health.

2. **SSE heartbeat with asyncio.wait_for()** -- File: `src/api/app.py` (lines 167-202). Replaced the inline `async for` loop with `asyncio.wait_for(event_iter.__anext__(), timeout=_HEARTBEAT_INTERVAL)` pattern. Heartbeats now fire during long inter-event gaps (e.g., 30s LLM first-token delay) by catching `TimeoutError` on the next-event await.

3. **Streaming PII redactor uses redacted buffer for lookahead** -- File: `src/agent/streaming_pii.py` (lines 110-121). Changed `self._buffer = self._buffer[-_MAX_PATTERN_LEN:]` to `self._buffer = redacted[-_MAX_PATTERN_LEN:]` so the safe/lookahead split operates on the same text representation. Re-scanning already-redacted placeholders like `[PHONE]` is a no-op, so this is safe.

4. **/feedback docstring corrected** -- File: `src/api/app.py` (lines 495-501). Updated docstring to honestly describe current behavior (log-only) and added `TODO(HEYSEVEN-42)` for LangFuse forwarding per placeholder response tracking rule.

5. **CMS webhook HTTP-level tests** -- File: `tests/test_api.py` (new `TestCmsWebhookEndpoint` class, 5 tests). Tests: valid payload returns "indexed", invalid signature returns 403, missing required fields returns "rejected", quarantined item (missing validation fields), unchanged content on duplicate submission.

6. **/feedback endpoint tests in test_api.py** -- File: `tests/test_api.py` (new `TestFeedbackEndpoint` class, 4 tests). Tests: valid feedback returns 200, invalid rating returns 422, invalid thread_id returns 422, feedback without comment succeeds.

## Deferred

- **threading.Lock in Firestore client accessors** (DeepSeek F-002 HIGH) -- Only 1/3 reviewers flagged. Real fix requires making accessors async and updating all callers, which is a larger refactor.
- **InMemoryBackend non-atomic increment** (DeepSeek F-005 MEDIUM) -- Only 1/3 reviewers flagged. Currently masked by middleware lock.
- **/sms/webhook not rate limited** (DeepSeek F-009 MEDIUM) -- Only 1/3 reviewers flagged.
- **Retriever cache not thread-safe** (Gemini F-005 MEDIUM) -- Only 1/3 reviewers flagged.
- **RateLimitMiddleware _request_counter getattr** (DeepSeek F-007 LOW) -- Only 1/3, low severity.
- **BoundedMemorySaver storage access** (GPT F-008 MEDIUM) -- Only 1/3. Mitigated by pinned LangGraph version.
- **E2E happy path test mock fragility** (GPT F-009 LOW) -- Only 1/3, low severity.

## Test Results

- Before: 1441 passed, 20 skipped
- After: 1450 passed, 20 skipped (+9 new tests)
- Coverage: 90.34% (above 90% threshold)

## Score

- DeepSeek: 85/100
- Gemini: 83/100
- GPT-5.2: 85/100
- Average: 84.3/100
