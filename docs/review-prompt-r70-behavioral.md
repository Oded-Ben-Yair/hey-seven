# Hey Seven Behavioral Quality Review — Round 70

## Repo
GitHub: https://github.com/Oded-Ben-Yair/hey-seven

## Purpose
This rubric evaluates **behavioral quality** — how well the AI concierge detects emotional signals, reads implicit intent, gathers information naturally, connects cross-domain knowledge, and calibrates empathy. This is SEPARATE from the infrastructure review (D1-D10).

## Scoring Rubric (5 Behavioral Dimensions, 0-10 each)

### B1: Sarcasm &amp; Nuance Detection (0-10)

**What to evaluate**: How well the system detects sarcastic, passive-aggressive, or backhanded messages and adjusts response tone accordingly.

**Implementation to review**:
- `src/agent/sentiment.py`: 6 `_SARCASM_PATTERNS` regex patterns, `_FRUSTRATED_PATTERNS`, VADER compound scoring
- `src/agent/prompts.py`: `CONCIERGE_SYSTEM_PROMPT` "mirror guest energy" instruction
- `src/agent/agents/_base.py`: `SENTIMENT_TONE_GUIDES` injection, frustration escalation

**What 9.0+ looks like**:
- Sarcasm patterns catch common casino guest sarcasm ("Oh wonderful another chatbot", "Yeah right", "Thanks for nothing")
- VADER + regex combo correctly classifies backhanded compliments as negative/frustrated
- System prompt instructs LLM to NOT respond enthusiastically to sarcastic messages
- Tone guidance adapts meaningfully — frustrated sarcasm gets empathetic response, not robotic

**What 5.0 looks like**:
- Only catches explicit negative words, misses sarcasm entirely
- Responds with enthusiastic "Great question!" to sarcastic messages
- No VADER/regex sarcasm detection — relies entirely on LLM prompt

**Scoring guide**:
- 9-10: All common sarcasm forms detected, appropriate tone adjustment, tested
- 7-8: Most sarcasm detected, some edge cases missed, tone guidance exists
- 5-6: Basic negative detection only, limited sarcasm awareness
- 3-4: No sarcasm detection, occasional enthusiastic responses to sarcastic messages
- 0-2: No sentiment awareness at all

### B2: Implicit Intent &amp; Context Reading (0-10)

**What to evaluate**: How well the system reads between the lines — detecting urgency, fatigue, celebration, loyalty signals, and unstated needs.

**Implementation to review**:
- `src/agent/prompts.py`: "mirror guest energy", "curated suggestions rather than raw lists"
- `src/agent/whisper_planner.py`: `WhisperPlan.next_topic`, `offer_readiness`, `conversation_note`
- `src/agent/extraction.py`: Deterministic field extraction (occasion, preferences, party_size)
- `src/agent/agents/_base.py`: Guest context injection, whisper plan formatting

**What 9.0+ looks like**:
- System prompt explicitly instructs LLM to pick up implicit signals (urgency, fatigue, celebration)
- Whisper planner guidance helps LLM understand what information to gather next
- Extraction catches occasion mentions ("anniversary", "honeymoon") and feeds to profile
- Agent doesn't just answer the literal question — it reads context (e.g., "kids getting restless" → fast/practical answers)

**What 5.0 looks like**:
- Only responds to literal questions, misses all subtext
- No whisper planner guidance for implicit intent
- Extraction catches some fields but doesn't influence response tone/approach

**Scoring guide**:
- 9-10: System prompt + whisper planner + extraction create a coherent implicit intent pipeline
- 7-8: Some implicit intent awareness via prompts, extraction works but planner guidance limited
- 5-6: Basic extraction only, no prompt instructions for reading between lines
- 3-4: Purely literal responses, no context awareness
- 0-2: Ignores all non-explicit content

### B3: Information Gathering &amp; Profile Building (0-10)

**What to evaluate**: How naturally and effectively the system gathers guest information and builds a usable profile across conversation turns.

**Implementation to review**:
- `src/agent/whisper_planner.py`: `WhisperPlan` structured output, `next_topic`, `extraction_targets`
- `src/agent/extraction.py`: `extract_fields()` — name, party_size, visit_date, preferences, occasion
- `src/agent/state.py`: `extracted_fields` with `_merge_dicts` reducer, `guest_name`
- `src/data/guest_profile.py`: Guest profile management
- `src/agent/agents/_base.py`: Guest context injection into system prompt

**What 9.0+ looks like**:
- Whisper planner suggests next profiling topic based on what's ALREADY known (avoids redundancy)
- Extraction regex correctly captures common formats (names, dates, party sizes, dietary preferences)
- Profile accumulates via reducer across multiple turns (not overwritten)
- Agent naturally weaves information gathering into helpful responses (not interrogation-style)
- Profile completeness drives offer readiness and suggestion timing

**What 5.0 looks like**:
- Extraction works for explicit statements but profile doesn't persist
- No whisper planner guidance for natural information gathering
- Agent treats each turn independently, doesn't remember earlier extractions

**Scoring guide**:
- 9-10: Full pipeline (extraction → reducer → whisper → profile → context injection) works end-to-end
- 7-8: Most components work, some gaps in persistence or planner guidance
- 5-6: Basic extraction only, no cross-turn accumulation or planner integration
- 3-4: Minimal extraction, no profile building
- 0-2: No information gathering capability

### B4: Agentic Behavior &amp; Cross-Domain Connection (0-10)

**What to evaluate**: How proactively the system connects information across domains (dining + entertainment + hotel) and offers relevant suggestions.

**Implementation to review**:
- `src/agent/agents/_base.py`: Proactive suggestion injection (positive-only gate, confidence >= 0.8)
- `src/agent/whisper_planner.py`: `proactive_suggestion`, `suggestion_confidence`
- `src/agent/agents/registry.py`: Agent dispatch registry
- `src/agent/prompts.py`: CONCIERGE_SYSTEM_PROMPT "connect the dots" instruction
- `src/agent/nodes.py`: `route_from_router()` routing logic

**What 9.0+ looks like**:
- System prompt explicitly instructs cross-domain connection ("connect dining, entertainment, accommodations")
- Proactive suggestions are gated by positive sentiment AND confidence threshold (no annoying upsell)
- Max 1 suggestion per session (suggestion_offered flag persists across turns)
- Suggestions emerge from whisper planner analysis, not random injection
- Agent doesn't just answer about restaurants — it might mention a nearby show if profile indicates interest

**What 5.0 looks like**:
- Responds only to the specific domain asked about
- No cross-domain connection even when context makes it natural
- No proactive suggestions

**Scoring guide**:
- 9-10: Consistent cross-domain connection, well-calibrated proactive suggestions, sentiment-gated
- 7-8: Some cross-domain awareness, proactive suggestions work but edge cases exist
- 5-6: Limited cross-domain, proactive suggestions exist but poorly calibrated
- 3-4: Single-domain responses only
- 0-2: No agentic behavior

### B5: Emotional Intelligence &amp; Calibrated Empathy (0-10)

**What to evaluate**: How appropriately the system calibrates emotional responses — from matching celebration energy to handling grief/anxiety with care.

**Implementation to review**:
- `src/agent/prompts.py`: `SENTIMENT_TONE_GUIDES` (4 categories), `HEART_ESCALATION_LANGUAGE`
- `src/agent/agents/_base.py`: Graduated HEART escalation (2 frustrated = hear+empathize, 3+ = full HEART)
- `src/agent/persona.py`: Persona envelope post-processing (branding under emotional responses)
- `src/agent/sentiment.py`: 4-category sentiment output (positive, negative, neutral, frustrated)
- `src/agent/nodes.py`: `off_topic_node` self-harm response (988 Lifeline)

**What 9.0+ looks like**:
- 4 sentiment categories with distinct tone guides (not just "be nice")
- HEART framework for sustained frustration (graduated based on consecutive count)
- Self-harm crisis response with 988 Lifeline (compassionate, actionable)
- Positive responses match celebration energy (not clinical)
- Persona envelope maintains branding consistency even under emotional responses
- Guest name injection personalizes emotional responses

**What 5.0 looks like**:
- Binary positive/negative with generic responses
- No HEART framework for escalation
- Self-harm not handled as special case
- Same response tone regardless of emotional context

**Scoring guide**:
- 9-10: All 4 sentiment categories produce measurably different responses, HEART works, crisis handled
- 7-8: Good sentiment-driven responses, some gaps in graduation or edge cases
- 5-6: Basic sentiment detection, limited tone adjustment
- 3-4: Generic responses regardless of emotion
- 0-2: No emotional intelligence

## Reviewer Instructions

You are evaluating EXISTING CODE BEHAVIOR, not proposing features.

### Finding categories:
- **Code gap** (MAJOR): Code should handle this but doesn't. Example: sarcasm pattern list misses common forms.
- **Prompt gap** (MAJOR): System prompt doesn't instruct the right behavior. Example: no grief-handling instruction.
- **Nice-to-have** (MINOR): Would improve quality but isn't a gap. Example: more granular sentiment categories.
- **Feature request** (OUT OF SCOPE): New capability entirely. Example: "needs ML sarcasm model" or "needs CRM integration".

### What is OUT OF SCOPE:
- Requesting ML models for sarcasm/intent detection (regex + VADER + prompt is the current approach)
- Requesting VIP/loyalty tier integration (requires external CRM API)
- Requesting real-time engagement scoring (new telemetry feature)
- Requesting multi-language emotional intelligence (English-only MVP)
- Proposing entirely new architectural components

### Evaluation process:
1. Read each source file listed under each dimension
2. For each dimension, identify what works well and what's missing
3. Score based on the rubric above
4. List findings as:

### Finding N (SEVERITY): Title
- **Location**: `file.py:line`
- **Problem**: What's wrong or missing
- **Impact**: What behavioral quality this degrades
- **Fix**: Specific code/prompt change (realistic, not "add ML model")

Minimum 3 findings per dimension, 15 total across B1-B5.

## Source Files to Review

### Primary (behavioral logic):
- `src/agent/sentiment.py` — sentiment detection, sarcasm patterns, VADER
- `src/agent/prompts.py` — system prompts, tone guides, HEART framework, validation prompt
- `src/agent/agents/_base.py` — execute_specialist with all behavioral injections
- `src/agent/whisper_planner.py` — background planning, proactive suggestions
- `src/agent/persona.py` — output guardrails, branding enforcement
- `src/agent/extraction.py` — deterministic field extraction
- `src/agent/nodes.py` — router, off_topic, greeting, validate

### Supporting:
- `src/agent/state.py` — CasinoHostState, reducers
- `src/agent/guardrails.py` — 5-layer deterministic guardrails
- `src/agent/compliance_gate.py` — compliance validation node
- `src/casino/config.py` — multi-property config
