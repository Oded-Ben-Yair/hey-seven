# Gemini API Reference for Hey Seven

> **Generated**: 2026-03-03 from live API + context7 docs + web research.
> **Purpose**: Ground truth for model selection. Stop guessing model names.

## Models We Use

| Model ID | Role | Input Tokens | Output Tokens | Temp Default | Status |
|----------|------|-------------|---------------|-------------|--------|
| `gemini-3-flash-preview` | Primary (fast) | 1,048,576 | 65,536 | 1.0 | Preview |
| `gemini-3.1-pro-preview` | Complex (reasoning) | 1,048,576 | 65,536 | 1.0 | Preview |
| `gemini-embedding-001` | Embeddings | 2,048 | 1 | N/A | GA |

## All Available Gemini Models (Live API, 2026-03-03)

### Gemini 3.x Family (Current)
| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| `gemini-3-flash-preview` | 1M | 65K | Fast, multimodal. **Our primary.** |
| `gemini-3-pro-preview` | 1M | 65K | **DEPRECATED** — use 3.1 Pro instead |
| `gemini-3.1-pro-preview` | 1M | 65K | Best reasoning. **Our complex model.** |
| `gemini-3.1-pro-preview-customtools` | 1M | 65K | Custom tools variant |
| `gemini-3-pro-image-preview` | 131K | 32K | Image generation |
| `gemini-3.1-flash-image-preview` | 65K | 65K | Fast image gen ("Nano Banana 2") |

### Gemini 2.5 Family (Previous Gen, Still Available)
| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| `gemini-2.5-flash` | 1M | 65K | GA. Was our primary before R83. |
| `gemini-2.5-pro` | 1M | 65K | GA. Good fallback. |
| `gemini-2.5-flash-lite` | 1M | 65K | Lighter/cheaper |

### Gemini 2.0 Family (Older, Limited Output)
| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| `gemini-2.0-flash` | 1M | 8K | Only 8K output. Don't use for generation. |
| `gemini-2.0-flash-lite` | 1M | 8K | Even lighter |

### Aliases (Resolve to Latest)
| Alias | Points To |
|-------|-----------|
| `gemini-flash-latest` | gemini-2.5-flash (as of 2026-03-03) |
| `gemini-pro-latest` | gemini-2.5-pro (as of 2026-03-03) |

## Critical Configuration Notes

### Temperature
- **ALL Gemini models default to temperature=1.0** (verified from API metadata)
- Gemini 3.x works fine with 1.0 — the old guidance "requires 1.0" may have been overstated
- Gemini 2.5 also defaults to 1.0 in API metadata
- For our validator LLM (structured binary classification), 1.0 is acceptable — structured output constrains randomness

### Structured Output (`with_structured_output`)
- Works on all Gemini 3.x models
- Gemini Flash has a schema complexity limit: <5 constrained fields (Literal + bounded floats)
- Flat `str|None` fields work reliably; nested `ConfidenceField` objects fail
- Our working schemas: RouterOutput (3 fields), DispatchOutput (3 fields), ValidationResult (2 fields)
- **Test structured output against live API**, not just mocks (R76 lesson)

### Thinking (Gemini 3.x)
- Supported on `gemini-3-pro-preview` and `gemini-3.1-pro-preview`
- Levels: MINIMAL, LOW, MEDIUM (new in 3.1), HIGH
- Uses `thinking_config` in generation config
- Not used in our production pipeline (adds latency), but available for eval/judge

### Context Window
- All 3.x models: 1M input tokens (matches 2.5 family)
- Output: 65K tokens (8x more than 2.0 family's 8K limit)
- Our `MAX_HISTORY_MESSAGES=20` sliding window is well within limits

## Model Routing (R83)

```
Simple query (high confidence, positive/neutral) → gemini-3-flash-preview
Complex query (any of these):
  - Router confidence < 0.7             → gemini-3.1-pro-preview
  - Sentiment: frustrated/grief/negative → gemini-3.1-pro-preview
  - Crisis state active                  → gemini-3.1-pro-preview
  - Whisper complexity: high/multi_domain → gemini-3.1-pro-preview
```

## Semantic Injection Classifier Issue (R83)

The semantic injection classifier (`classify_injection_semantic`) fails consistently with Gemini 3 Flash.
After `_CLASSIFIER_DEGRADATION_THRESHOLD` consecutive failures, it enters RESTRICTED MODE (fail-closed),
blocking ALL messages that pass deterministic guardrails.

**Workaround**: Set `SEMANTIC_INJECTION_ENABLED=false` for eval runs.
**Root cause**: Needs investigation — may be structured output schema incompatibility with Gemini 3 Flash.
**Safety**: Deterministic regex guardrails (Layer 1, 211 patterns) remain enforced.

## Fallback Strategy

If Gemini 3.x becomes unavailable:
1. `gemini-2.5-flash` — proven stable, was our primary for R52-R82
2. `gemini-2.5-pro` — proven for complex queries
3. Both have identical token limits (1M/65K)

## Rate Limits (Free Tier)

### Pricing per Million Tokens (USD, Developer API)

| Model | Input | Output | Cache Input | Batch Input | Batch Output |
|-------|-------|--------|-------------|-------------|-------------|
| `gemini-3-flash-preview` | $0.50 | $3.00 | $0.05 | $0.25 | $1.50 |
| `gemini-3.1-pro-preview` (≤200K) | $2.00 | $12.00 | $0.20 | $1.00 | $6.00 |
| `gemini-3.1-pro-preview` (>200K) | $4.00 | $18.00 | $0.40 | $2.00 | $9.00 |
| `gemini-2.5-flash` (comparison) | $0.30 | $2.50 | $0.03 | $0.15 | $1.25 |

**Cost impact**: Flash 3.0 is +67% input / +20% output vs 2.5 Flash. Pro 3.1 is ~6x more expensive.
**Free tier**: 3-flash and 2.5-flash are free. **3.1-pro has NO free tier** (AI Studio playground only).

### Rate Limits (CRITICAL for eval runs)

| Model | Free RPM | Free RPD | Tier 1 RPM | Tier 1 RPD |
|-------|----------|----------|------------|------------|
| `gemini-3-flash-preview` | ~10 | ~250 | ~20-25 | **250** |
| `gemini-3.1-pro-preview` | N/A | N/A | ~20-25 | **250** |
| `gemini-2.5-flash` | 10 | 250 | **300** | **1,500** |

**CRITICAL**: Preview models get only **250 RPD even on paid Tier 1** — same as free tier.
GA models (2.5-flash) get 300 RPM / 1,500 RPD on Tier 1. This is why our R83 eval takes 2+ hours.

**Tier 2** ($250 spend + 30 days): 1,000+ RPM, 10,000+ RPD for all models.

### Eval Run Implications

- 109 scenarios × 3 turns = 327 LLM calls (router + specialist + validator per turn ≈ 981 total)
- At ~10 RPM free tier: ~98 minutes minimum
- At ~25 RPM Tier 1: ~39 minutes minimum
- **Recommendation**: Use `gemini-2.5-flash` for eval runs (300 RPM), then 3.x for production

## LangChain Integration Notes

- We use `langchain-google-genai` package with `ChatGoogleGenerativeAI`
- API key via `GOOGLE_API_KEY` env var (auto-detected by LangChain)
- `with_structured_output(PydanticModel)` works for schemas with <5 constrained fields
- Separate TTL-cached singletons per model (Flash + Pro + Validator)
- Async via `ainvoke()` — never use sync `invoke()` in async code

## Version History

| Date | Change |
|------|--------|
| 2026-03-03 (R83) | Migrated to gemini-3-flash-preview + gemini-3.1-pro-preview |
| 2026-02-15 (R1) | Started with gemini-2.5-flash |
| 2026-02-22 (Phase 5) | Ceiling test confirmed Pro > Flash for synthesis |
