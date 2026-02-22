# R28 Final Hostile Review — Hey Seven Phase 4

**Reviewer**: Claude Opus 4.6 (hostile review mandate)
**Date**: 2026-02-22
**Rubric**: 40/60 (Code Quality 40, Agent Quality 60)
**Codebase**: ~23K LOC, 55 test files, 51 source modules across 10 packages

---

## Code Quality Dimensions (40 points)

### 1. Graph Architecture — 9/10

**Strengths:**
- Custom 11-node StateGraph with clear topology: `START -> compliance_gate -> {greeting, off_topic, router} -> retrieve -> whisper_planner -> generate -> validate -> {persona_envelope -> respond, generate (RETRY), fallback}`. This is textbook production LangGraph with validation loops, conditional edges, and multiple terminal nodes.
- State design is exemplary: `PropertyQAState` uses `Annotated[list, add_messages]` for messages, `Annotated[dict, _merge_dicts]` for accumulated fields, `Annotated[int, _keep_max]` for session-level counters. The `_initial_state()` parity check at import time (`ValueError` instead of `assert`) catches schema drift in all environments.
- Node name constants (`NODE_ROUTER`, `NODE_GENERATE`, etc.) as module-level `frozenset` prevent silent rename breakage. `_KNOWN_NODES` validates SSE streaming event filtering.
- Dual-layer feature flag architecture (build-time topology via `DEFAULT_FEATURES` + runtime behavior via async `is_feature_enabled()`) is well-reasoned and documented with a 40-line inline comment explaining why topology flags cannot be runtime.

**Remaining Gaps:**
- `persona_envelope_node` in `persona.py` line 177 reads branding from `DEFAULT_CONFIG` instead of the per-casino `get_casino_profile()`. The `_base.py` (line 188) correctly uses `get_casino_profile()`, but the persona envelope node does not. This means branding enforcement (exclamation limits, emoji) always uses default config regardless of which casino is active. This is an inconsistency between the two branding injection points.

### 2. RAG Pipeline — 8/10

**Strengths:**
- Per-item chunking with category-specific formatters (`_format_restaurant`, `_format_entertainment`, `_format_hotel`) instead of text splitters. This is the correct approach for structured data and was unanimously praised across prior reviews.
- SHA-256 content hashing for idempotent ingestion (`ids` parameter to `Chroma.from_texts()`). Re-ingestion produces identical IDs, preventing duplicates.
- RRF reranking with multi-strategy retrieval, relevance score filtering via `RAG_MIN_RELEVANCE_SCORE`, and property_id metadata isolation for multi-tenant safety.
- Retrieval timeout guard (`_RETRIEVAL_TIMEOUT = 10s`) with `asyncio.wait_for()` prevents hung ChromaDB queries from blocking the event loop.

**Remaining Gaps:**
- Version-stamp purging for stale chunks is documented in CLAUDE.md rules but not visibly implemented in `pipeline.py`. When restaurant data changes (e.g., new hours), old chunks with the previous content persist alongside new chunks. For the MVP this is acceptable (full re-ingestion on data change), but production multi-property deployment needs the `_ingestion_version` metadata pattern.
- The pipeline uses `RecursiveCharacterTextSplitter` for markdown knowledge-base files alongside the per-item JSON chunking. The markdown splitter chunk_size/overlap parameters are not tuned per content type (regulations vs. dining guides have very different optimal chunk sizes).

### 3. API Design — 8.5/10

**Strengths:**
- Pure ASGI middleware stack (no `BaseHTTPMiddleware`): `RequestLoggingMiddleware`, `SecurityHeadersMiddleware`, `RateLimitMiddleware`, `RequestBodyLimitMiddleware`, `ApiKeyMiddleware`, `ErrorHandlingMiddleware`. This is mandatory for SSE streaming and correctly implemented.
- SSE streaming via `astream_events(version="v2")` with per-node lifecycle events (`graph_node` start/complete with duration_ms), token streaming from generate node only (filtered by `langgraph_node == NODE_GENERATE`), and inline PII redaction via `StreamingPIIRedactor`.
- `CancelledError` at INFO level (not ERROR) for SSE disconnects. `record_cancellation()` on circuit breaker instead of `record_failure()` prevents inflating failure count from normal client disconnects.
- Lifespan context manager with conditional RAG ingestion (only for `VECTOR_DB=chroma`, not in production) and graceful degradation when agent initialization fails (returns 503 instead of crash).

**Remaining Gaps:**
- No health check endpoint returning agent readiness state visible in the first 100 lines of `app.py`. A `/ready` endpoint that checks `app.state.agent is not None` would improve Kubernetes/Cloud Run probing.
- Rate limiter is memory-bounded per IP, but there is no rate limiting on the `/chat` endpoint specifically (e.g., per-thread_id rate limiting to prevent a single session from overwhelming the LLM).

### 4. Testing Strategy — 8.5/10

**Strengths:**
- 55 test files across comprehensive categories: unit tests (nodes, agents, guardrails, extraction, sentiment), integration tests (full pipeline, phase 2-4), E2E tests (R26 E2E Phase 4), domain tests (R24), conversation flow tests (R26), and evaluation tests (R22 LLM judge).
- Golden conversation dataset with 6 multi-turn scenarios covering dining, complaint, persona drift, safety, proactive suggestions, and context retention. Regression detection via `detect_regression()` with configurable threshold and NaN safety.
- CI quality gate: `test_golden_conversations_pass_baseline()` runs the full golden dataset against `QUALITY_BASELINE` and fails with regression details. This catches behavioral quality regressions in CI.
- Singleton cleanup fixture pattern with `autouse=True, scope='function'` clearing all caches between tests. 13 singleton caches identified and cleared.
- State parity check at import time ensures `_initial_state()` covers all `PropertyQAState` fields (minus `messages`).

**Remaining Gaps:**
- No E2E integration test that sends a query through the FULL compiled graph with mocked LLMs verifying lifecycle events for every node. The R26 E2E tests exercise `execute_specialist()` directly but do not compile the graph and invoke `chat()` or `chat_stream()`. This means wiring bugs between nodes (wrong conditional edge, missing edge) are invisible.
- LLM judge evaluation is offline-only (deterministic keyword/heuristic). The `EVAL_LLM_ENABLED=true` path logs a fallback message but has no implementation. For production quality measurement, real LLM-as-judge scoring would provide more meaningful signal.

---

## Agent Quality Dimensions (60 points)

### 5. Prompts & Guardrails — 9.5/10

**Strengths:**
- 5-layer deterministic guardrails with 84 compiled regex patterns across 4 languages (English, Spanish, Portuguese, Mandarin): prompt injection (11 patterns with Unicode normalization and zero-width char detection), responsible gaming (19 patterns), age verification (6 patterns), BSA/AML (16 patterns with chip walking and structuring), patron privacy (10 patterns).
- Two-layer injection defense: regex first (zero-cost), then semantic LLM classifier (fail-closed). The fail-closed semantic classifier returns synthetic `InjectionClassification(is_injection=True, confidence=1.0)` on error, which is correct for a regulated environment.
- Compliance gate priority chain is meticulously ordered with a 30-line docstring explaining why injection runs at position 3 (before content guardrails) and semantic classifier runs at position 8 (after all deterministic checks).
- `string.Template.safe_substitute()` used throughout for all prompt assembly (no `.format()` on user content), preventing KeyError DoS from curly braces.
- Adversarial validation prompt with 6 explicit criteria (Grounded, On-topic, No gambling advice, Read-only, Accurate, Responsible gaming) and structured output via `ValidationResult` Pydantic model with Literal types.

**Remaining Gaps:**
- The `CONCIERGE_SYSTEM_PROMPT` has a hardcoded "About $property_name" section describing Mohegan Sun specifically ("Uncasville, Connecticut, owned by the Mohegan Tribe"). For multi-property deployment (Foxwoods, Hard Rock AC), this section should be parameterized or loaded from `CASINO_PROFILES`. Currently, Hard Rock AC guests would see Mohegan Sun's description.

### 6. EQ / Emotional Intelligence — 9/10

**Strengths:**
- HEART framework (Hear, Empathize, Apologize, Resolve, Thank) with graduated escalation: 2 consecutive frustrated messages get HEAR + EMPATHIZE, 3+ get full 5-step HEART. This is research-backed and well-implemented in `_base.py` lines 203-234.
- `_count_consecutive_frustrated()` iterates messages in reverse, running sub-1ms VADER sentiment detection on each HumanMessage. Returns count before first positive/neutral, correctly resetting on mood change.
- Sentiment-adaptive tone guides (`SENTIMENT_TONE_GUIDES`) inject different coaching into the system prompt based on detected sentiment (frustrated, negative, positive, neutral=empty).
- Sarcasm detection: 6 regex patterns catch common sarcastic constructs ("Oh great, another...", "Thanks for nothing", "Just wonderful", "Love waiting") that VADER misclassifies as positive. Anchored to sentence boundaries to avoid false positives.

**Remaining Gaps:**
- The HEART escalation offer ("Would you like me to connect you with one of our dedicated hosts who can assist you personally?") is static text in the system prompt. If the guest declines the escalation offer, the agent has no mechanism to track the decline and stop re-offering. The next frustrated turn will inject the same offer again.

### 7. Guest Experience — 8.5/10

**Strengths:**
- Proactive suggestions from WhisperPlan with triple safety gate: (1) `suggestion_confidence >= 0.8`, (2) `guest_sentiment == "positive"` (not just non-negative, requires positive evidence), (3) `suggestion_offered` flag persists via `_keep_max` reducer to enforce max-1-per-conversation.
- Guest name injection in `persona_envelope_node` with intelligent casing: `_LOWERCASE_STARTERS` frozenset ensures "Sarah, we have great restaurants" not "Sarah, We have great restaurants", while preserving proper nouns ("Sarah, Mohegan Sun has...").
- Guest context injection into system prompts: name, party_size, visit_date, preferences, occasion. Context accumulates across turns via `_merge_dicts` reducer on `extracted_fields`.
- Fallback messages include property contact info (`_fallback_message()` in `_base.py`) as a single source of truth, preventing divergent fallback text across specialist agents.

**Remaining Gaps:**
- Guest name persistence relies on `extracted_fields` (reducer-backed) as fallback when `guest_name` (per-turn, no reducer) resets. However, the persona envelope node reads `guest_name` first and falls back to `extracted_fields["name"]`. If the extraction regex misses the name on turn N but extracted it on turn N-1, the fallback works. But if a different specialist agent extracts the name (via `_dispatch_to_specialist`'s guest_context_update), it sets `guest_name` which has no reducer and resets next turn. The dual path (guest_name vs extracted_fields.name) creates a fragile chain.
- No multi-language greeting support in the persona envelope. Spanish greeting templates exist in `CASINO_PROFILES` (`greeting_template_es`) but are never used in the greeting node.

### 8. Domain Intelligence — 9/10

**Strengths:**
- Knowledge base contains rich, real casino domain data: loyalty programs (MGM, Caesars, Mohegan Momentum, Foxwoods Rewards, Wynn, Hard Rock Unity) with tier structures, earning rates, rollover benefits, and ADT formula. Dining guide with 40+ venues including dietary accommodation policies, comp dining tiers, reservation systems, and peak times.
- Comp math education handled sensitively in the knowledge base: "Players often misunderstand comp math. A player who loses $2,000 but had only $400 in theoretical may feel under-comped. Hosts must educate sensitively."
- Multi-property awareness: 3 casino profiles (Mohegan Sun CT, Foxwoods CT, Hard Rock AC NJ) with per-state regulatory data (helplines, self-exclusion authorities, gaming age), property-specific branding (persona names: Seven, Foxy, Ace), and operational details (gaming floor sqft, dining venue counts, hotel towers).
- State-specific regulatory compliance: NJ properties get 1-800-GAMBLER per NJ DGE requirements. CT properties get 1-888-789-7777. Self-exclusion URLs and authorities are per-state.

**Remaining Gaps:**
- Knowledge base is Mohegan Sun-centric. Foxwoods and Hard Rock AC have profile configurations but no dedicated knowledge-base data files. Queries about "Gordon Ramsay Hell's Kitchen" (Foxwoods) would only work if the dining guide markdown is loaded, but the RAG pipeline ingests from `PROPERTY_DATA_PATH` which points to Mohegan Sun's JSON. Multi-property RAG isolation is documented but not fully wired.

### 9. Persona Consistency — 8.5/10

**Strengths:**
- Per-property persona names via `BrandingConfig` in `CASINO_PROFILES`: Seven (Mohegan Sun, warm_professional), Foxy (Foxwoods, warm_professional), Ace (Hard Rock AC, casual). Tone and formality are configurable per property.
- `PERSONA_STYLE_TEMPLATE` maps branding config to natural language prompt guidance: tone guide, formality guide, emoji policy, exclamation limit. This produces consistent persona framing across all specialist agents.
- Persona drift prevention: after `_PERSONA_REINJECT_THRESHOLD // 2` human turns (5 turns), a condensed `PERSONA REMINDER` SystemMessage is injected. Human-turn counting (not total message counting) prevents inflation from retry AIMessages.
- Branding enforcement in `persona_envelope_node`: exclamation limit (excess replaced with periods), emoji removal (when `emoji_allowed=False`).

**Remaining Gaps:**
- Persona reminder in `_base.py` line 275 reads persona name from `DEFAULT_CONFIG` instead of `get_casino_profile()`. If the active casino is Hard Rock AC (persona "Ace"), the reminder still says "You are Seven". This is a bug — the persona drift prevention itself drifts to the wrong persona.
- The greeting node (`greeting_node`) hardcodes "I'm **Seven**" regardless of which casino is active. Multi-property deployment requires reading the persona name from the active casino's branding config.

### 10. Evaluation Framework — 8.5/10

**Strengths:**
- 5-metric offline evaluation: empathy, cultural_sensitivity, conversation_flow, persona_consistency, guest_experience. Each metric has a dedicated scoring function with regex patterns, keyword matching, and weighted composition.
- 6 golden conversation test cases covering dining (dietary needs), complaint (VIP escalation), persona drift (3-exchange consistency), safety (gambling concern pivot), proactive suggestion (anniversary + dining), and context retention (name + party size across turns).
- Regression detection with NaN safety, invalid baseline key detection, configurable threshold, and per-metric granular reporting. `QUALITY_BASELINE` calibrated from golden dataset evaluation.
- CI gate: `test_golden_conversations_pass_baseline()` integrates evaluation into the test suite. Regression descriptions include current score, baseline, drop amount, and threshold.
- `ConversationEvalReport.to_dict()` enables JSON serialization for CI output and monitoring dashboards.

**Remaining Gaps:**
- Offline scoring is heuristic-based (keyword counting). The `_score_empathy_offline` function counts matches from `_EMPATHY_PHRASES` list and `_EMPATHY_ACKNOWLEDGMENT_PATTERNS`. This is brittle: "I'd be happy to help" scores the same whether it follows "I'm so frustrated" (good empathy) or "What time is the show?" (generic politeness). Context-dependent empathy scoring needs LLM-as-judge, which is scaffolded (`EVAL_LLM_ENABLED`) but not implemented.
- Golden dataset has 6 conversations. For robust regression detection across 5 metrics, 20-30 conversations covering edge cases (multi-language, competitor questions, off-topic sequences, long conversations) would reduce false positive/negative rates.

---

## Score Summary

| # | Dimension | Score | Weight |
|---|-----------|-------|--------|
| 1 | Graph Architecture | 9 | Code (10) |
| 2 | RAG Pipeline | 8 | Code (10) |
| 3 | API Design | 8.5 | Code (10) |
| 4 | Testing Strategy | 8.5 | Code (10) |
| **Code Subtotal** | | **34/40** | |
| 5 | Prompts & Guardrails | 9.5 | Agent (10) |
| 6 | EQ / Emotional Intelligence | 9 | Agent (10) |
| 7 | Guest Experience | 8.5 | Agent (10) |
| 8 | Domain Intelligence | 9 | Agent (10) |
| 9 | Persona Consistency | 8.5 | Agent (10) |
| 10 | Evaluation Framework | 8.5 | Agent (10) |
| **Agent Subtotal** | | **53/60** | |
| **TOTAL** | | **87/100** | |

---

## Top 3 Remaining Issues

1. **Persona name hardcoded in three locations** (Medium severity, multi-property blocker): (a) `persona.py:177` reads branding from `DEFAULT_CONFIG` not `get_casino_profile()`, (b) `_base.py:275` persona drift reminder reads from `DEFAULT_CONFIG`, (c) `greeting_node` hardcodes "I'm **Seven**". All three bypass the multi-property persona system. Hard Rock AC guests see "Seven" instead of "Ace".

2. **No full-graph E2E integration test** (Medium severity, wiring safety gap): All E2E tests exercise `execute_specialist()` directly or test individual nodes. No test compiles the graph via `build_graph()` and invokes `chat()` with mocked LLMs to verify the full node wiring, conditional edges, and state flow from START to END. This means a broken conditional edge (e.g., typo in routing function) would be invisible to CI.

3. **CONCIERGE_SYSTEM_PROMPT hardcodes Mohegan Sun description** (Low-medium severity): The "About $property_name" section describes "a premier tribal casino resort in Uncasville, Connecticut, owned by the Mohegan Tribe" regardless of which property is active. For multi-property deployment, this section needs parameterization from `CASINO_PROFILES`.

---

## Ship Recommendation

**SHIP with conditions.** The codebase is production-grade for single-property (Mohegan Sun) deployment. The 87/100 score reflects mature architecture, robust safety guardrails, and strong agent quality. The three identified issues are multi-property readiness gaps, not single-property blockers. For the initial Mohegan Sun MVP deployment, this is ready. For multi-property rollout, fix issue #1 (persona hardcoding) first -- it is the only bug that would produce visibly wrong behavior to end users.
