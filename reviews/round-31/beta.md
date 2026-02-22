# R31 Review: Group B (Docker & DevOps, Prompts & Guardrails, Scalability & Production, Trade-off Documentation, Domain Intelligence)

**Reviewer**: reviewer-beta
**Date**: 2026-02-22
**Phase**: Phase 5 (Steps 1-4 complete)
**Baseline**: R30 score 88/100

---

## Scores

| Dimension | Score | Max | Evidence |
|-----------|-------|-----|----------|
| Docker & DevOps | 7.5 | /8 | Exec form CMD, multi-stage build, non-root user, Trivy scan, canary deploy, smoke test with rollback. Minor: no `--no-traffic` version assertion in smoke test. |
| Prompts & Guardrails | 11 | /12 | $property_description parameterization works correctly. string.Template safe_substitute prevents injection. 5-layer guardrails with 84 patterns across 4 languages. One HIGH finding: helplines not parameterized by casino_id in _base.py. |
| Scalability & Production | 10.5 | /12 | TTL caches, asyncio.Lock everywhere, circuit breaker, Firestore hot-reload config. LLM judge not production-hardened (no rate limiting, no caching). hasattr guards are unnecessary noise. |
| Trade-off Documentation | 10.5 | /12 | decisions.log has 44 entries. Feature flag dual-layer documented in graph.py. Degraded-pass strategy well-documented. Phase 5 decisions NOT yet logged in decisions.log or status.json. |
| Domain Intelligence | 14 | /16 | Entertainment guide is accurate. Golden conversations cover 6 categories. LLM judge rubric captures core quality dimensions. Proactive suggestion gate is positive-only (correct). Missing: no entertainment-specific test cases, helpline data for non-CT/NJ states. |
| **Subtotal** | **53.5** | **/60** | |

---

## CRITICAL Findings

### C-001: LLM Judge Dimension Mapping Loses Semantic Fidelity

**File**: `/home/odedbe/projects/hey-seven/src/observability/llm_judge.py:691-696`

The `evaluate_conversation_llm()` function maps LLM judge dimensions to metrics with a **semantically incorrect mapping**:

```python
scores = {
    METRIC_EMPATHY: result.groundedness.score / 10.0,              # groundedness -> empathy??
    METRIC_CULTURAL_SENSITIVITY: result.persona_fidelity.score / 10.0, # persona -> cultural??
    METRIC_CONVERSATION_FLOW: result.contextual_relevance.score / 10.0, # OK
    METRIC_PERSONA_CONSISTENCY: result.safety.score / 10.0,         # safety -> persona??
    METRIC_GUEST_EXPERIENCE: result.proactive_value.score / 10.0,   # proactive -> overall??
}
```

The LLM judge evaluates 5 dimensions (groundedness, persona_fidelity, safety, contextual_relevance, proactive_value) but maps them to completely different metric names (empathy, cultural_sensitivity, conversation_flow, persona_consistency, guest_experience). This means:

- A response scoring 10/10 on **groundedness** (factual accuracy) would report as perfect **empathy** (emotional attunement)
- A response scoring 2/10 on **safety** (gambling advice violations) would report as poor **persona_consistency** (tone adherence)
- The `detect_regression()` function would flag "empathy regression" when the actual regression is in groundedness

This mapping makes the LLM judge output unreliable for any quality decision. The offline and online modes measure fundamentally different things under the same metric names, making regression detection across modes meaningless.

**Fix**: Either (a) align LLM judge dimensions to match the offline metrics (add empathy/cultural_sensitivity/conversation_flow dimensions to the LLM prompt), or (b) add separate metric names for LLM judge dimensions and don't mix them with offline metrics.

**Severity**: CRITICAL -- any production quality decision based on these metrics would be wrong.

---

## HIGH Findings

### H-001: get_responsible_gaming_helplines() Called Without casino_id in _base.py

**File**: `/home/odedbe/projects/hey-seven/src/agent/agents/_base.py:151`

```python
responsible_gaming_helplines=get_responsible_gaming_helplines(),
```

The `_base.py` parameterizes `property_description` from the casino profile (R29 fix, line 142-146) but still calls `get_responsible_gaming_helplines()` **without passing `casino_id`**. This means:

- Mohegan Sun (CT): Gets correct CT helplines (default) -- works by accident
- Hard Rock AC (NJ): Should get `1-800-GAMBLER` per NJ DGE, but gets CT helplines instead
- Any non-CT property: Gets wrong state-specific helplines

The function signature accepts `casino_id: str | None = None` (line 32 of prompts.py) and the full multi-property helpline lookup logic exists (lines 48-65). It's just not wired.

**Fix**: Pass `settings.CASINO_ID` to `get_responsible_gaming_helplines()`:
```python
responsible_gaming_helplines=get_responsible_gaming_helplines(settings.CASINO_ID),
```

**Severity**: HIGH -- regulatory compliance issue. NJ DGE requires NJ-specific helplines.

### H-002: hasattr(settings, "CASINO_ID") Guards Are Unnecessary and Misleading

**File**: `/home/odedbe/projects/hey-seven/src/agent/agents/_base.py:145,195,284`

```python
settings.CASINO_ID if hasattr(settings, "CASINO_ID") else ""
```

`CASINO_ID` is a **required field** on the `Settings` Pydantic model (line 76 of config.py: `CASINO_ID: str = "mohegan_sun"`). Pydantic BaseSettings guarantees this field exists on every Settings instance. The `hasattr` check:
1. Can never be False (Pydantic would raise ValidationError during construction)
2. Implies CASINO_ID might not exist, confusing future maintainers
3. The `else ""` fallback returns DEFAULT_CONFIG anyway, which is functionally identical to `settings.CASINO_ID` being `"mohegan_sun"`

This is dead code that adds noise. All 3 occurrences should just use `settings.CASINO_ID` directly.

**Severity**: HIGH -- code quality/maintainability concern in a core module.

### H-003: DEFAULT_CONFIG Import in persona.py Fallback (Line 183)

**File**: `/home/odedbe/projects/hey-seven/src/agent/persona.py:19,183`

```python
from src.casino.config import DEFAULT_CONFIG  # line 19
# ...
except Exception:
    branding = DEFAULT_CONFIG.get("branding", {})  # line 183
```

The R29 fix added `get_casino_profile()` for the primary path (lines 179-181), but the `except Exception` fallback still reads `DEFAULT_CONFIG` directly. Per the project's own CLAUDE.md rule ("Every import of DEFAULT_CONFIG for runtime data is a multi-tenant bug"), this fallback path returns Mohegan Sun branding for any property when the profile lookup fails.

The correct fallback is `get_casino_profile("")` which returns DEFAULT_CONFIG anyway, but at least routes through the proper abstraction. Or better: the except block should log and use a minimal safe branding dict rather than cross-tenant defaults.

**Severity**: HIGH -- multi-tenant branding leak on error path.

### H-004: status.json and decisions.log Not Updated for Phase 5

**File**: `/home/odedbe/projects/hey-seven/.claude/status.json`

`status.json` still shows:
- `"phase": "phase4-complete"` (should be phase5-in-progress or phase5-complete)
- `"commitHash": "4e479b2"` (Phase 4 final commit, not Phase 5)
- `"lastModified": "2026-02-22T12:00:00Z"` (same date but no Phase 5 data)
- `nextSteps` still lists Phase 5 items as TODO, not as completed

`decisions.log` has no entries after `2026-02-22` for Phase 5 decisions:
- No entry for "chose G-Eval pattern for LLM judge"
- No entry for "$property_description via string.Template for multi-property prompts"
- No entry for "full-graph E2E tests with mocked LLM strategy"
- No entry for "entertainment guide content decisions"

This violates the project's own documentation requirements.

**Severity**: HIGH -- documentation honesty gap.

---

## MEDIUM Findings

### M-001: LLM Judge Has No Rate Limiting or Cost Protection

**File**: `/home/odedbe/projects/hey-seven/src/observability/llm_judge.py:675-680`

`evaluate_conversation_llm()` creates a new `ChatGoogleGenerativeAI` instance on every call with no caching, no rate limiting, and no cost controls:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    max_output_tokens=2048,
)
```

In a CI/CD pipeline running against all 6 golden conversations per build, this is manageable. But if someone runs it in a monitoring loop or against a large test suite, there's no protection against cost runaway.

**Fix**: Use the existing TTL-cached LLM singleton pattern from nodes.py, or add a simple rate limiter.

### M-002: Entertainment Guide Is Mohegan-Sun-Only

**File**: `/home/odedbe/projects/hey-seven/knowledge-base/casino-operations/entertainment-guide.md`

The entire entertainment guide references only Mohegan Sun venues (Arena, Wolf Den, Comix, Momentum Rewards). For a multi-property system, this means:
- Foxwoods queries about entertainment would get Mohegan Sun venue names
- Hard Rock AC queries would get irrelevant CT venue data

The RAG pipeline should filter by `property_id` metadata, but the entertainment guide has no property_id metadata structure -- it's a single flat markdown file. Other knowledge-base files may have the same issue.

### M-003: No Golden Conversation for Entertainment Category

**File**: `/home/odedbe/projects/hey-seven/src/observability/llm_judge.py:831-993`

The 6 golden conversations cover: dining, complaint, persona, safety, proactive, retention. There is no test case for the entertainment category, despite entertainment being a core specialist agent. An entertainment-focused test case (e.g., "What shows are on this weekend?") would exercise the entertainment guide RAG retrieval and the entertainment specialist dispatch.

### M-004: Cloud Build Smoke Test Does Not Assert Version Match

**File**: `/home/odedbe/projects/hey-seven/cloudbuild.yaml:95-97`

The smoke test captures `DEPLOYED_VERSION` and logs it alongside `COMMIT_SHA`, but does NOT assert they match:

```bash
DEPLOYED_VERSION=$(cat /tmp/health.json | python3 -c "...")
echo "Deployed version: $DEPLOYED_VERSION, Expected: $COMMIT_SHA"
# No assertion! Just logging.
```

Per the project's own rules ("Post-Deploy Version Assertion MANDATORY"), the smoke test should fail if `DEPLOYED_VERSION != COMMIT_SHA`. Currently, a stale deployment from warm instances would pass the smoke test and get 100% traffic.

### M-005: test_full_graph_e2e.py Doesn't Verify Node Traversal Order

**File**: `/home/odedbe/projects/hey-seven/tests/test_full_graph_e2e.py`

The E2E tests verify final output (response content, mock_search calls) but do NOT verify which nodes were actually traversed. For a true E2E wiring test, the tests should capture lifecycle events (node start/complete) and assert the expected traversal path. Example: dining query should traverse `compliance_gate -> router -> retrieve -> whisper -> generate -> validate -> persona -> respond`.

Without this, the tests could pass even if the graph wiring is broken (e.g., if a mock returns the expected content from an unexpected code path).

### M-006: Gemini Model Version Mismatch Between Agent and Judge

**File**: `/home/odedbe/projects/hey-seven/src/config.py:27` vs `/home/odedbe/projects/hey-seven/src/observability/llm_judge.py:676`

The agent uses `gemini-2.5-flash` (config.py line 27) while the LLM judge uses `gemini-2.0-flash` (llm_judge.py line 676). This is likely intentional (cost control for evaluation) but is not documented anywhere. It also means the judge model's understanding of "good" responses may differ from the agent model's generation style.

---

## Phase 5 Impact Assessment

### Step 1: Full-Graph E2E Tests (test_full_graph_e2e.py)
**Impact**: +1.5 points (testing_strategy dimension)
- 5 E2E tests through build_graph() -> chat() with mocked LLMs
- Covers dining, greeting, off_topic, responsible_gaming, multi-turn
- Addresses the #1 testing gap flagged by every prior reviewer
- Weakness: no node traversal verification (M-005)

### Step 2: System Prompt Parameterization ($property_description)
**Impact**: +1.0 points (prompts_guardrails + scalability dimensions)
- CONCIERGE_SYSTEM_PROMPT now uses $property_description from casino profiles
- 13 tests verify parameterization for all 3 properties + default
- Template itself verified to contain no hardcoded Mohegan references
- Weakness: helplines still not parameterized (H-001)

### Step 3: LLM-as-Judge with G-Eval Pattern (llm_judge.py)
**Impact**: +0.5 points (evaluation_framework dimension, reduced by C-001)
- Structured output via LLMJudgeOutput Pydantic model
- Stochastic retry pattern for Gemini structured output failures
- Graceful fallback to offline scoring on any failure
- 11 tests covering models, offline, fallback, and live paths
- Critical weakness: dimension mapping is semantically wrong (C-001)

### Step 4: RAG Quality Tests + Entertainment Guide
**Impact**: +0.5 points (domain_intelligence + rag dimensions)
- Entertainment guide adds a key missing knowledge-base category
- Entertainment content is accurate for Mohegan Sun
- Weakness: single-property content (M-002), no entertainment golden conversation (M-003)

### Net Assessment
Phase 5 Steps 1-4 add approximately **+3.5 points** before deductions, but C-001 (LLM judge mapping) should block any quality decision reliance on the judge, reducing effective gain to **+2.5 to +3.0 points**.

**Estimated R31 score**: 90.5-91/100 (up from 88).

The path to 95 still requires:
1. Fix C-001 (LLM judge dimension alignment)
2. Fix H-001 (helpline parameterization)
3. Add node traversal verification to E2E tests
4. Multi-property knowledge base isolation
5. Version assertion in Cloud Build smoke test
