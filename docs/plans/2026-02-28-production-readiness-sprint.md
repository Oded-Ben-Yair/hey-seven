# Production-Readiness Sprint Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the 1 CRITICAL + real MAJORs from external review, improve behavioral dimensions, write rebuttal for false positives — target 9.0+ across all 20 dimensions.

**Architecture:** Six sequential waves: (1) CRITICAL safety fix, (2) technical hardening, (3) crisis/safety behavioral, (4) behavioral quality, (5) rebuttal document, (6) live re-evaluation. Each wave is independently committable.

**Tech Stack:** Python 3.12, LangGraph 0.2.60, FastAPI, Gemini 2.5 Flash, ChromaDB, pytest

---

## Wave 1: Safety-Critical (CRITICAL Fix)

### Task 1: Enforce RE2 in Production

**Files:**
- Modify: `src/agent/regex_engine.py:18-25`
- Modify: `tests/test_doc_accuracy.py` (add RE2 assertion)

**Step 1: Write the failing test**

Add to `tests/test_doc_accuracy.py`:

```python
class TestRE2Enforcement:
    """Verify RE2 is enforced in non-development environments."""

    def test_re2_enforcement_raises_in_production(self, monkeypatch):
        """If ENVIRONMENT != 'development' and RE2 is missing, startup must fail."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        # Simulate RE2 not available
        import src.agent.regex_engine as mod
        original = mod.RE2_AVAILABLE
        try:
            mod.RE2_AVAILABLE = False
            with pytest.raises(RuntimeError, match="google-re2"):
                mod.enforce_re2_in_production()
        finally:
            mod.RE2_AVAILABLE = original

    def test_re2_enforcement_passes_in_development(self, monkeypatch):
        """In development, missing RE2 should not raise."""
        monkeypatch.setenv("ENVIRONMENT", "development")
        import src.agent.regex_engine as mod
        original = mod.RE2_AVAILABLE
        try:
            mod.RE2_AVAILABLE = False
            mod.enforce_re2_in_production()  # Should not raise
        finally:
            mod.RE2_AVAILABLE = original

    def test_re2_enforcement_passes_when_available(self, monkeypatch):
        """When RE2 is available, enforcement always passes."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        import src.agent.regex_engine as mod
        original = mod.RE2_AVAILABLE
        try:
            mod.RE2_AVAILABLE = True
            mod.enforce_re2_in_production()  # Should not raise
        finally:
            mod.RE2_AVAILABLE = original
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_doc_accuracy.py::TestRE2Enforcement -v`
Expected: FAIL (enforce_re2_in_production not defined)

**Step 3: Implement**

Add to `src/agent/regex_engine.py` after the `is_re2_active()` function:

```python
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
```

**Step 4: Wire into startup**

In `src/api/app.py`, add to the lifespan function (after graph build, before yield):

```python
from src.agent.regex_engine import enforce_re2_in_production
enforce_re2_in_production()
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_doc_accuracy.py::TestRE2Enforcement -v`
Expected: PASS (3 tests)

**Step 6: Run full test suite**

Run: `pytest tests/ -x --timeout=60 -q`
Expected: All pass (no regressions)

**Step 7: Commit**

```bash
git add src/agent/regex_engine.py src/api/app.py tests/test_doc_accuracy.py
git commit -m "fix: enforce RE2 in production — fail-fast at startup if google-re2 missing

Closes CRITICAL finding from external review. In non-development environments,
application startup now raises RuntimeError if google-re2 is not installed,
preventing deployment with 204 guardrail patterns vulnerable to ReDoS."
```

---

## Wave 2: Technical Hardening

### Task 2: Fix Version Parity

**Files:**
- Modify: `pyproject.toml:3`

**Step 1: Fix version**

Change `pyproject.toml` line 3 from `version = "0.1.0"` to `version = "1.3.0"`.

**Step 2: Add parity test**

Add to `tests/test_doc_accuracy.py`:

```python
class TestVersionParity:
    """Ensure version is consistent across all sources."""

    def test_pyproject_matches_config(self):
        """pyproject.toml version must match src/config.py VERSION."""
        import tomllib
        from pathlib import Path
        with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        from src.config import Settings
        assert pyproject["project"]["version"] == Settings.model_fields["VERSION"].default

    def test_env_example_matches_config(self):
        """VERSION in .env.example must match src/config.py."""
        from pathlib import Path
        env_path = Path(__file__).parent.parent / ".env.example"
        if not env_path.exists():
            pytest.skip(".env.example not found")
        content = env_path.read_text()
        from src.config import Settings
        version = Settings.model_fields["VERSION"].default
        assert f"VERSION={version}" in content
```

**Step 3: Run tests**

Run: `pytest tests/test_doc_accuracy.py::TestVersionParity -v`
Expected: PASS

**Step 4: Commit**

```bash
git add pyproject.toml tests/test_doc_accuracy.py
git commit -m "fix: sync pyproject.toml version to 1.3.0 + add parity test"
```

### Task 3: Tighten Degrade-Pass in Validate Node

**Files:**
- Modify: `src/agent/nodes.py:359-376`
- Modify: test file for validate_node

**Step 1: Write the failing test**

Add to the validate_node test file:

```python
@pytest.mark.asyncio
async def test_degraded_pass_requires_grounding_on_first_attempt():
    """Degrade-pass on attempt 0 must check for retrieval grounding.

    If retrieved_context is empty or all scores below threshold,
    route to FAIL instead of serving unvalidated response.
    """
    from src.agent.nodes import _degraded_pass_result
    # No retrieval grounding — should FAIL even on first attempt
    result = _degraded_pass_result(retry_count=0, has_grounding=False)
    assert result["validation_result"] == "FAIL"

@pytest.mark.asyncio
async def test_degraded_pass_allows_grounded_first_attempt():
    """Degrade-pass with good retrieval grounding on attempt 0 is OK."""
    from src.agent.nodes import _degraded_pass_result
    result = _degraded_pass_result(retry_count=0, has_grounding=True)
    assert result["validation_result"] == "PASS"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nodes.py -k "degraded_pass" -v`
Expected: FAIL (signature mismatch)

**Step 3: Implement**

Modify `_degraded_pass_result` in `src/agent/nodes.py`:

```python
def _degraded_pass_result(retry_count: int, has_grounding: bool = True) -> dict[str, Any]:
    """Return degraded-pass or fail-closed result based on attempt number and grounding.

    R38 fix M-001: Extracted from duplicate try/except blocks in validate_node.
    R74 fix D1: Added grounding check. Degrade-pass on first attempt is only safe
    when retrieval context exists and scored above threshold. Without grounding,
    an unvalidated response is likely hallucinated — route to fallback instead.
    """
    if retry_count == 0 and has_grounding:
        logger.warning(
            "Degraded-pass: serving unvalidated response (first attempt, "
            "validator unavailable, grounding present)"
        )
        return {"validation_result": "PASS"}
    return {
        "validation_result": "FAIL",
        "retry_feedback": "Validation unavailable — returning safe fallback for guest safety.",
    }
```

Update callers in `validate_node` to pass `has_grounding`:

```python
    retrieved = state.get("retrieved_context", [])
    has_grounding = bool(retrieved)
    # ... in except blocks:
    return _degraded_pass_result(retry_count, has_grounding=has_grounding)
```

**Step 4: Run tests**

Run: `pytest tests/ -x --timeout=60 -q`
Expected: All pass

**Step 5: Commit**

```bash
git add src/agent/nodes.py tests/test_nodes.py
git commit -m "fix: tighten degrade-pass to require retrieval grounding

Validator failure on first attempt now checks for retrieval context.
Without grounding, routes to fallback instead of serving unvalidated response."
```

### Task 4: Doc Drift Tests for Node Count

**Files:**
- Modify: `tests/test_doc_accuracy.py`

**Step 1: Write the test**

```python
class TestGraphTopology:
    """Verify graph topology matches documentation."""

    def test_node_count_matches_docs(self):
        """Graph must have exactly 11 nodes (documented in ARCHITECTURE.md and README)."""
        from src.agent.graph import build_graph
        graph = build_graph()
        # LangGraph compiled graphs expose nodes via .get_graph().nodes
        graph_info = graph.get_graph()
        # Exclude __start__ and __end__ virtual nodes
        real_nodes = [n for n in graph_info.nodes if not n.startswith("__")]
        assert len(real_nodes) == 11, f"Expected 11 nodes, got {len(real_nodes)}: {real_nodes}"

    def test_node_names_use_constants(self):
        """All node names must come from constants.py (no magic strings)."""
        from src.agent.constants import _KNOWN_NODES
        from src.agent.graph import build_graph
        graph = build_graph()
        graph_info = graph.get_graph()
        real_nodes = {n for n in graph_info.nodes if not n.startswith("__")}
        assert real_nodes == _KNOWN_NODES, f"Mismatch: {real_nodes.symmetric_difference(_KNOWN_NODES)}"
```

**Step 2: Run tests**

Run: `pytest tests/test_doc_accuracy.py::TestGraphTopology -v`
Expected: PASS (validates current state)

**Step 3: Commit**

```bash
git add tests/test_doc_accuracy.py
git commit -m "test: add graph topology drift tests — node count + names from constants"
```

---

## Wave 3: Crisis/Safety Behavioral

### Task 5: Replace Brittle Crisis Exit Heuristic

**Files:**
- Modify: `src/agent/compliance_gate.py:223-246`
- Modify: tests for compliance_gate

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_crisis_persistence_requires_explicit_safe_confirmation():
    """Crisis mode must persist unless guest explicitly says they're OK."""
    from src.agent.compliance_gate import compliance_gate_node
    # Guest in crisis mode says "nobody can help me" — NOT a property question
    state = _make_state("nobody can help me", crisis_active=True)
    result = await compliance_gate_node(state)
    assert result["query_type"] == "self_harm"

@pytest.mark.asyncio
async def test_crisis_exits_on_explicit_safe_confirmation():
    """Crisis mode should exit when guest explicitly confirms they're OK."""
    from src.agent.compliance_gate import compliance_gate_node
    state = _make_state("I'm feeling better now, thanks. What restaurants do you have?", crisis_active=True)
    result = await compliance_gate_node(state)
    # Should allow transition since guest confirmed safe AND asked property question
    assert result["query_type"] is None  # passes through to router

@pytest.mark.asyncio
async def test_crisis_persists_on_ambiguous_message():
    """Crisis mode persists on ambiguous messages (not clearly safe)."""
    from src.agent.compliance_gate import compliance_gate_node
    state = _make_state("whatever", crisis_active=True)
    result = await compliance_gate_node(state)
    assert result["query_type"] == "self_harm"
```

**Step 2: Implement**

Replace the crisis persistence block in `compliance_gate.py` (lines 223-246):

```python
    # 7.4 Crisis context persistence — R73 fix, R74 hardened.
    # When crisis_active=True, maintain crisis response UNLESS guest explicitly
    # confirms safety ("I'm OK", "I'm feeling better", "I'm fine now") AND
    # asks a property question. Both conditions required — safety confirmation
    # alone is not enough (they may still need support while asking about parking).
    if state.get("crisis_active", False):
        _SAFE_CONFIRMATIONS = (
            "i'm ok", "i'm okay", "im ok", "im okay",
            "i'm fine", "im fine", "i'm better", "im better",
            "i'm feeling better", "im feeling better",
            "i'm alright", "im alright", "i'm good", "im good",
            "thanks for checking", "i appreciate it",
            "i'll be okay", "ill be okay",
        )
        msg_lower = user_message.lower()
        _has_safe_confirmation = any(phrase in msg_lower for phrase in _SAFE_CONFIRMATIONS)
        _is_property_question = any(
            kw in msg_lower
            for kw in ("restaurant", "steakhouse", "buffet", "spa", "pool",
                        "show", "arena", "room", "hotel", "check", "hours",
                        "parking", "directions", "shuttle", "wifi")
        )
        if _has_safe_confirmation and _is_property_question:
            logger.info("Crisis context: guest confirmed safe AND asked property question — allowing transition")
        else:
            logger.info("Crisis context active — maintaining crisis response for follow-up")
            logger.info(
                json.dumps({
                    "audit_event": "guardrail_triggered",
                    "category": "crisis_persistence",
                    "query_type": "self_harm",
                    "timestamp": time.time(),
                    "action": "blocked",
                    "severity": "INFO",
                })
            )
            return {"query_type": "self_harm", "router_confidence": 1.0}
```

**Step 3: Run tests**

Run: `pytest tests/ -k "crisis" -v --timeout=60`
Expected: All pass

**Step 4: Commit**

```bash
git add src/agent/compliance_gate.py tests/
git commit -m "fix: harden crisis exit — require explicit safe confirmation + property question

Previously used keyword list to detect property questions only. Now requires
BOTH explicit safety confirmation ('I'm OK', 'I'm feeling better') AND a
property question to exit crisis mode. Ambiguous messages maintain crisis."
```

---

## Wave 4: Behavioral Quality (B2/B3/B4)

### Task 6: Wire Proactive Suggestions into Specialist Execution

**Files:**
- Modify: `src/agent/agents/_base.py` (execute_specialist)
- Add tests for proactive suggestion gating

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_proactive_suggestion_injected_when_conditions_met():
    """Proactive suggestion appears in system prompt when:
    - whisper_plan has suggestion with confidence >= 0.8
    - sentiment is not negative/frustrated
    - suggestion_offered is False
    - retrieved context has docs
    """
    # Test that the system prompt includes the suggestion text
    # when all conditions are met
    ...

@pytest.mark.asyncio
async def test_proactive_suggestion_suppressed_on_negative_sentiment():
    """Proactive suggestion must NOT appear when sentiment is negative."""
    ...
```

NOTE: The exact test implementation depends on the current structure of `execute_specialist()`. The implementer must read `_base.py` fully and write tests that verify the suggestion injection logic in the system prompt construction. The key assertion: when conditions met, the system prompt contains "You might also enjoy:" or similar; when conditions not met, it doesn't.

**Step 2: Implement in `_base.py`**

Inside `execute_specialist()`, after building the system prompt and before the LLM call, add gated suggestion injection:

```python
    # R74 B4: Proactive suggestion injection (gated).
    # Only inject when ALL conditions are met:
    # 1. Whisper plan has a suggestion with confidence >= 0.8
    # 2. Guest sentiment is not negative or frustrated
    # 3. suggestion_offered is False (max 1 per session)
    # 4. Retrieved context has at least 1 document (grounding exists)
    whisper = state.get("whisper_plan")
    suggestion_text = ""
    if (
        whisper
        and whisper.get("proactive_suggestion")
        and whisper.get("suggestion_confidence", 0.0) >= 0.8
        and state.get("guest_sentiment") not in ("negative", "frustrated")
        and not state.get("suggestion_offered", False)
        and state.get("retrieved_context")
    ):
        suggestion_text = (
            f"\n\n## Proactive Suggestion (weave naturally into response)\n"
            f"{whisper['proactive_suggestion']}"
        )
        suggestion_update = {"suggestion_offered": True}
    else:
        suggestion_update = {}
```

Append `suggestion_text` to the system prompt and merge `suggestion_update` into the return dict.

**Step 3: Run tests**

Run: `pytest tests/ -x --timeout=60 -q`
Expected: All pass

**Step 4: Commit**

```bash
git add src/agent/agents/_base.py tests/
git commit -m "feat: wire proactive suggestions into specialist execution (B4)

Whisper planner suggestions now injected into specialist system prompt when:
confidence >= 0.8, sentiment not negative, max 1 per session, grounding exists."
```

### Task 7: Use domains_discussed for Engagement Variation (B3)

**Files:**
- Modify: `src/agent/agents/_base.py` (execute_specialist, inside prompt building)

**Step 1: Write the test**

```python
def test_domains_discussed_produces_cross_domain_suggestions():
    """When domains_discussed has entries, the prompt should suggest unexplored domains."""
    # Build state with domains_discussed=["dining", "entertainment"]
    # Verify the system prompt includes suggestion of unexplored domains
    ...
```

**Step 2: Implement**

In `_base.py`, after building behavioral signals but before the LLM call:

```python
    # R74 B3: Cross-domain engagement variation.
    # When guest has discussed some domains, suggest unexplored ones.
    domains_discussed = state.get("domains_discussed", [])
    all_domains = {"dining", "entertainment", "hotel", "spa", "gaming", "shopping", "promotions"}
    unexplored = all_domains - set(domains_discussed)
    cross_domain_hint = ""
    if domains_discussed and unexplored:
        suggestions = ", ".join(sorted(unexplored)[:3])
        cross_domain_hint = (
            f"\n\nThe guest has already asked about: {', '.join(domains_discussed)}. "
            f"If natural, you might mention: {suggestions}."
        )
```

Append `cross_domain_hint` to the system prompt.

**Step 3: Run tests, commit**

```bash
git add src/agent/agents/_base.py tests/
git commit -m "feat: use domains_discussed for cross-domain engagement (B3)"
```

---

## Wave 5: Rebuttal Document

### Task 8: Write Review Response

**Files:**
- Create: `docs/review-response.md`

**Step 1: Write the document**

Create `docs/review-response.md` with structured rebuttals for each false positive, including:

1. **RAG Chunk ID Collision (D2)** — False positive. `pipeline.py:244` uses `\x00` delimiter since R36 fix A5. Same pattern as `reranking.py:60`. Reviewer missed line 242-244.

2. **Docker --require-hashes (D6)** — False positive. `requirements-prod.txt` is generated by `pip-compile --generate-hashes` (visible in file header). `Dockerfile:19` uses `pip install --require-hashes`. Reviewer saw `requirements.txt` (dev file), not `requirements-prod.txt` (prod).

3. **Circuit Breaker Race (D8)** — Partially false. All state mutations (`record_failure`, `record_success`, `allow_request`) use `async with self._lock`. The `state` property (line 239) is documented as read-only monitoring (line 256-267). `is_open` property documents its non-atomic nature and recommends `allow_request()` for control flow (line 253-267).

4. **ARCHITECTURE.md 8-node claim (D9)** — False positive. ARCHITECTURE.md line 5 says "11-node", line 41 says "11 nodes". No "8-node" text exists in the file. Reviewer likely confused with historical state.

5. **Retrieval Pool Hardcoded (D8)** — Partially false. Pool size is configurable via `RETRIEVAL_POOL_SIZE` setting. Semaphore backpressure exists via `_LLM_SEMAPHORE` (20 concurrent).

Include code snippets and file:line references for each claim.

**Step 2: Commit**

```bash
git add docs/review-response.md
git commit -m "docs: add structured review response with evidence for false positives"
```

---

## Wave 6: Live Re-evaluation

### Task 9: Run Live Agent Evaluation

**Files:**
- Use existing: `tests/evaluation/run_live_eval.py`
- Use existing: `tests/evaluation/run_judge_panel.py`

**Step 1: Run live evaluation**

Follow the R72 evaluation protocol:
1. Set `GOOGLE_API_KEY` environment variable
2. Run `python tests/evaluation/run_live_eval.py` (74 scenarios, real Gemini Flash)
3. Collect responses to `tests/evaluation/r74-responses.json`

**Step 2: Run 3-model judge panel**

Use the judge panel framework with 20 representative scenarios:
1. Gemini Pro, GPT-5.2, Grok 4 via MCP tools
2. Score across B1-B5 dimensions
3. Calculate ICC(2,1)

**Step 3: Write delta report**

Create `tests/evaluation/r74-delta-report.md` comparing R73 vs R74 scores.

**Step 4: Commit**

```bash
git add tests/evaluation/r74-*.json tests/evaluation/r74-delta-report.md
git commit -m "feat: R74 live evaluation — post-sprint behavioral measurement"
```

---

## Summary

| Wave | Tasks | Files Modified | Expected Impact |
|------|-------|---------------|----------------|
| 1: Safety-Critical | 1 | 3 | Close CRITICAL: RE2 enforcement |
| 2: Technical | 3 | 4 | Close 3 real MAJORs: version, degrade-pass, doc drift |
| 3: Crisis/Safety | 1 | 2 | Improve B9 Safety: crisis exit hardened |
| 4: Behavioral | 2 | 2 | Improve B3 Engagement + B4 Proactivity |
| 5: Rebuttal | 1 | 1 | Document 5 false positives with evidence |
| 6: Evaluation | 1 | 3 | Measure improvement with live agent |
| **Total** | **9** | **~15** | **Target: 9.0+ across 20 dimensions** |
