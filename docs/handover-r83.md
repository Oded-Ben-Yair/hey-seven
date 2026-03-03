# R83 Session Handover — 2026-03-03

## Session Identity
- **Session**: R83 Overnight Sprint
- **Duration**: ~5 hours
- **Commits**: 6 on main (`07bdb3d` → `259f605`)
- **Tests**: 3502 pass, 0 fail
- **Version**: v1.5.0

## What Was Done

### Wave 1: Gemini 3.1 Migration (DONE)
- `MODEL_NAME`: `gemini-2.5-flash` → `gemini-3-flash-preview`
- `COMPLEX_MODEL_NAME`: added `gemini-3.1-pro-preview`
- `MODEL_TEMPERATURE`: `0.3` → `1.0` (Gemini 3.x requirement — ALL models default to 1.0)
- Updated 22 files (src, tests, docs, .env, pyproject.toml)
- **There is NO `gemini-3.1-flash`** — Flash is 3.0, Pro is 3.1

### Wave 2A: Few-Shot Examples Wired (DONE)
- 25 `FEW_SHOT_EXAMPLES` (5 specialists × 5 behavioral patterns) injected into specialist system prompts
- Import in `src/agent/agents/_base.py`, capped at 3 examples per specialist
- Gated by `few_shot_examples_enabled` feature flag (default True)

### Wave 2B: Acknowledgment Fix (DONE)
- `greeting_node` detects mid-conversation acks (human_count > 1)
- Returns brief contextual follow-up ("Glad to help! Anything else about dining...") instead of full greeting template
- 4 unit tests in `TestGreetingNodeAcknowledgment`

### Wave 3: Flash→Pro Model Routing (DONE)
- `_select_model(state)` — deterministic, no LLM call
- Routes to Pro for: confidence < 0.7, frustrated/grief/negative, crisis, high complexity
- `_get_complex_llm()` — separate TTL-cached singleton with +15s timeout
- `model_used` state field for observability
- 8 unit tests in `TestModelRouting`

### Critical Bug Fix: Gemini 3.x Content Normalization
- **Root cause**: `_base.py:938` did `str(response.content)` which stringified `[{'type': 'text', 'text': '...'}]`
- **Impact**: ALL downstream processing (validator, slop detector, persona, eval) received garbled text
- **Fix**: `_normalize_content()` helper — extracts text from both `str` and `list[dict]` formats
- **Scope**: 17 instances across 8 files replaced

### Wave 4: Subset Eval (DONE)
- 20 scenarios (2 per dimension), run on `gemini-3-flash-preview`
- **Result: 98% real responses** (59/60 turns), 0 raw format, 0 fallback
- Results at: `tests/evaluation/r83-subset-responses.json`
- 9/10 dimensions at 100% response rate, B5_emotional at 83%

### Artifacts Created
- `docs/gemini-api-reference.md` — Live API specs for all 27 Gemini models, pricing, rate limits
- `docs/plans/eval-v2-streaming-design.md` — Architecture for real-time eval system
- `tests/evaluation/run_subset_eval.py` — Fast 20-scenario eval script

---

## WHAT THE NEXT SESSION MUST DO FIRST

### Step 1: Run Judge Panel (MANDATORY FIRST ACTION)

Run a 3-model judge panel on `tests/evaluation/r83-subset-responses.json`:

```bash
# Option A: Use the existing judge panel script (update model refs if needed)
GOOGLE_API_KEY=$(grep "^GOOGLE_API_KEY=" .env | cut -d= -f2) \
  python3 tests/evaluation/run_judge_panel.py --round r83-subset

# Option B: Manual judge via MCP tools
# GPT-5.2: azure_chat or azure_code_review
# Grok 4: grok_reason
# DeepSeek: azure_deepseek_reason
```

**Judge prompt must include**:
1. The scenario's expected_behavior from the YAML
2. The agent's actual response from r83-subset-responses.json
3. Score B1-B10 on 1-10 scale with justification
4. Flag any: hallucinated facts, wrong tone, missed emotional cues, promotional slop

**Use ALL THREE judges** (different model families catch different things):
- **GPT-5.2**: Cross-file logic bugs, RFC compliance
- **Grok 4**: Doc-code parity, overly verbose responses
- **DeepSeek**: Normalization, encoding, logical flaws

### Step 2: Deep Analysis of Judge Results

After running judges, analyze:
1. **Per-dimension consensus scores** (B1-B10, mean of 3 judges)
2. **ICC(2,1)** inter-rater reliability — should be > 0.7
3. **Bottom 3 dimensions** — these are the fix targets
4. **Specific scenario failures** — which scenarios scored < 5.0 from 2+ judges?
5. **Compare to R82**: R82 was 4.7/10. What did R83 achieve?

### Step 3: Map Remaining Work

Based on judge results, create a prioritized fix plan:
- If B1 (sarcasm) low → tune sarcasm detection + response style
- If B3 (engagement) low → proactivity gates too strict, lower thresholds
- If B4 (agentic) low → add "suggest next step" to specialist prompts
- If B5 (emotional) low → expand SENTIMENT_TONE_GUIDES
- If B6 (tone) low → add more slop patterns
- If B7 (coherence) low → reference previous answer in follow-ups

---

## Known Issues to Fix

| Issue | Severity | Location |
|-------|----------|----------|
| Semantic injection classifier fails with Gemini 3.x | HIGH | `src/agent/guardrails.py` — enters RESTRICTED MODE, blocks everything |
| Preview model rate limits (~10 RPM free) | MEDIUM | Use `gemini-2.5-flash` (300 RPM) for bulk eval runs |
| Eval system writes atomically at end | LOW | Design doc ready at `docs/plans/eval-v2-streaming-design.md` |
| 6 commits unpushed to GitHub | LOW | `git push origin main` |

## Key Learnings from This Session

### MUST REMEMBER
1. **`grep "^GOOGLE_API_KEY="` with `^` anchor** — without it, matches comment lines in .env
2. **Gemini 3.x returns `list[dict]` content** — always use `_normalize_content()`, never `str()`
3. **Fix at the SOURCE** — 15 downstream fixes didn't help until line 938 (the source) was fixed
4. **Preview models have ~10 RPM** — use GA models (2.5-flash, 300 RPM) for bulk eval
5. **Python imports are cached at process start** — code fixes don't apply to already-running processes
6. **Background bash buffers stdout** — can't see real-time eval output; use foreground for debugging
7. **Temperature 1.0 for ALL Gemini models** — API metadata confirms all default to 1.0
8. **There is no gemini-3.1-flash** — Flash is 3.0, Pro is 3.1

### Architecture Decisions Made
- Model routing is DETERMINISTIC (no LLM call) — based on state signals
- Separate TTL-cached singletons per model (Flash + Pro + Validator)
- Few-shot examples capped at 3 per specialist (prompt size control)
- Feature flag for every new behavior (rollback without deploy)

## File Reference

| File | What Changed |
|------|-------------|
| `src/config.py` | MODEL_NAME, COMPLEX_MODEL_NAME, MODEL_ROUTING_ENABLED, temp=1.0 |
| `src/agent/nodes.py` | _normalize_content(), _select_model(), _get_complex_llm(), _get_routed_llm(), greeting ack handling |
| `src/agent/agents/_base.py` | FEW_SHOT_EXAMPLES import+injection, _normalize_content at line 938 |
| `src/agent/dispatch.py` | Model routing decision + model_used in result |
| `src/agent/state.py` | model_used field |
| `src/agent/graph.py` | _initial_state model_used, streaming normalize |
| `src/agent/persona.py` | _normalize_content |
| `src/agent/whisper_planner.py` | _normalize_content |
| `src/agent/profiling.py` | _normalize_content |
| `src/casino/feature_flags.py` | few_shot_examples_enabled |
| `src/casino/config.py` | few_shot_examples_enabled in all 6 casino configs |
| `docs/gemini-api-reference.md` | NEW: all 27 models, pricing, limits |
| `docs/plans/eval-v2-streaming-design.md` | NEW: real-time eval architecture |
| `tests/evaluation/run_subset_eval.py` | NEW: fast 20-scenario eval |
