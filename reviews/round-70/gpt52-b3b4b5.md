# R70 Behavioral Review: B3, B4, B5 — GPT-5.2

## Reviewer: GPT-5.2 (Azure AI Foundry)
## Date: 2026-02-26

## Scores

| Dim | Name | Score |
|-----|------|-------|
| B3 | Information Gathering & Profile Building | 4.0 |
| B4 | Agentic Behavior & Cross-Domain Connection | 5.0 |
| B5 | Emotional Intelligence & Calibrated Empathy | 3.0 |

## B3: Information Gathering & Profile Building — 4/10

**Justification**: Brittle extraction + weak repair/clarification flows. Regex extraction works for explicit statements but the system has no mechanism to handle terse replies, repeated questions, or multi-part requests.

### Finding 1 (MAJOR): No repair loop for "not what I asked" / correction signals
- **Location**: Extraction + whisper planner (no "user correction" detector), system prompt (no repair directive)
- **Problem**: When the guest rejects the answer ("Not what I asked"), there's no mechanism to acknowledge mismatch, restate the request, ask a clarifying question, or suppress further extraction.
- **Impact**: Repeatedly off-target responses, user churn, compounding extraction errors.
- **Fix**: Add a lightweight "repair intent" classifier (regex + short LLM check) that overrides `next_topic` to a "clarify_request" mode.

### Finding 2 (MAJOR): Multi-part requests collapse into single lossy query
- **Location**: System prompt ("connect dots") + whisper planner (single `next_topic`)
- **Problem**: Multi-part guest asks (e.g., "hotel + show + dinner near venue + budget") are handled as one blob; planner only chooses one topic, risking dropped constraints.
- **Impact**: Missed requirements, irrelevant suggestions, more back-and-forth.
- **Fix**: Allow `next_topic` to be a list or add `subtasks[]` in planner output; require agent to confirm constraints and answer in structured sections.

### Finding 3 (MAJOR): Extraction brittle to terse confirmations ("ok", "sure", "yep")
- **Location**: Extraction regex + whisper planner "fail-silent"
- **Problem**: Short user responses yield no extraction and planner may fail-silent, leaving agent without next step.
- **Impact**: Conversation stalls or agent hallucinates preferences.
- **Fix**: Add "low-info turn" detector; force `next_topic` to safe clarifier (e.g., "visit_date") and prompt assistant to ask one minimal question.

### Finding 4 (MINOR): No explicit brevity/format detection
- **Location**: System prompt (no brevity directive), whisper schema (no verbosity field)
- **Problem**: "Quick version please" won't reliably get concise output.
- **Impact**: Perceived as not listening; higher friction on mobile.
- **Fix**: Add `response_style` field (brief/normal/detailed) inferred from user text.

## B4: Agentic Behavior & Cross-Domain Connection — 5/10

**Justification**: Intent is there — system prompt says "connect dots," proactive suggestions exist, whisper planner provides guidance. But over-gating (positive-only + 0.8 + max-1) and lack of domain-state tracking undermine proactivity.

### Finding 1 (MAJOR): Proactive suggestions over-gated
- **Location**: `_base.py` proactive suggestion injection
- **Problem**: Positive-only sentiment gate excludes neutral/uncertain guests who need help most ("first time here", "not sure what to do"). This is backwards for concierge "agentic" behavior.
- **Impact**: Missed opportunities to guide planning; agent feels reactive and unhelpful.
- **Fix**: Expand eligibility to include neutral and uncertain states. Gate on relevance + confidence, not positivity. Consider 2/session cap instead of 1.

### Finding 2 (MAJOR): No domain-state tracking for cross-domain follow-through
- **Location**: System prompt ("connect dots"), whisper planner (only `next_topic`)
- **Problem**: "Connect dots" is aspirational but there's no state machine tracking open threads across dining/entertainment/hotel. One `next_topic` encourages whiplash.
- **Impact**: Incoherent planning flow; user forced to restate constraints.
- **Fix**: Maintain a small "open_slots" state (hotel/dining/show/transport) and mark as pending/filled. Planner picks highest-impact missing slot.

### Finding 3 (MINOR): Fail-silent planner can degrade agentic behavior to dead-ends
- **Location**: `whisper_planner.py` fail-silent contract
- **Problem**: If planner fails parsing, agent loses all next-step guidance entirely.
- **Impact**: Random or generic responses; inconsistent concierge behavior.
- **Fix**: Provide deterministic fallback: if whisper output missing, default to summarize known context + ask for top missing field.

### Finding 4 (OUT OF SCOPE): No tool-use / booking actions
- **Location**: Overall system design
- **Problem**: True "agentic" concierge implies taking actions (hold reservation, check availability). Not present.
- **Fix**: Add booking/search tools — feature request.

## B5: Emotional Intelligence & Calibrated Empathy — 3/10

**Justification**: Too coarse. Only 4 sentiment categories with the neutral category producing zero guidance. Misses grief, anxiety, and safety-critical allergy handling entirely. HEART only triggers on "frustrated" — not on other emotional states that need care.

### Finding 1 (MAJOR): No grief/trauma handling beyond generic "negative" tone
- **Location**: `prompts.py` SENTIMENT_TONE_GUIDES (4 entries), HEART triggers (frustrated only)
- **Problem**: "Mom passed away" is treated as generic negative. No condolence, no pacing, no offer to adjust plans. HEART won't trigger because grief isn't "frustrated."
- **Impact**: High risk of sounding cold/transactional in most sensitive moment. Brand damage.
- **Fix**: Add a **grief** sentiment category or tone guide with response rubric: acknowledge, condolences, offer gentle options, suppress upsell.

### Finding 2 (MAJOR): Anxiety/nerves not modeled
- **Location**: `prompts.py` SENTIMENT_TONE_GUIDES
- **Problem**: "First time, kind of nervous" → anxiety is not frustration. Neutral/negative tones won't reliably elicit reassurance, step-by-step guidance, or "you're in good hands" framing.
- **Impact**: Missed emotional need; user feels unseen. Particularly important for first-time casino visitors.
- **Fix**: Add anxiety/uncertainty guide: validate feelings, reduce choices, offer simple itinerary, ask permission before suggesting.

### Finding 3 (MAJOR): Safety-critical dietary/allergy handling lacks urgency protocol
- **Location**: `extraction.py` (preferences/dietary) + system prompt (no safety escalation)
- **Problem**: "Severe peanut allergy" may extract but there's no mandatory safety language (cross-contamination warning, advise notifying staff) or constraint enforcement.
- **Impact**: Potential harm; massive liability.
- **Fix**: Add "allergy severity" detector; force agent to confirm severity, recommend contacting venue, prefer allergy-aware venues, include safety disclaimer.

### Finding 4 (MINOR): Gambling-loss frustration not specialized
- **Location**: Sentiment + HEART trigger rules
- **Problem**: "Losing all day" maps to negative, not frustrated; HEART won't kick in, and there's no responsible-gaming supportive language for this specific scenario.
- **Impact**: Tone mismatch in casino context.
- **Fix**: Add "gambling frustration" pattern triggering supportive, RG-compliant response with non-gambling alternatives.

## Severity Summary

| Severity | B3 | B4 | B5 | Total |
|----------|----|----|----|----|
| MAJOR | 3 | 2 | 3 | 8 |
| MINOR | 1 | 1 | 1 | 3 |
| OUT OF SCOPE | 0 | 1 | 0 | 1 |
