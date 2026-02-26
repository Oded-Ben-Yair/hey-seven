# R70 Behavioral Review: B1, B2 — Grok 4

## Reviewer: Grok 4 (Reasoning)
## Date: 2026-02-26

## Scores

| Dim | Name | Score |
|-----|------|-------|
| B1 | Sarcasm & Nuance Detection | 4.0 |
| B2 | Implicit Intent & Context Reading | 3.0 |

## B1: Sarcasm & Nuance Detection — 4/10

**Justification**: Catches low-hanging fruit (3/5 adversarial cases) but crumbles on subtlety. VADER misclassifies sarcastic positives as positive/neutral, bypassing the entire sarcasm handling pipeline. In a casino where sarcasm is rampant, a 40% miss rate is unacceptable.

### Finding 1 (MAJOR): Subtle Sarcasm Blind Spot
- **Location**: `sentiment.py` sarcasm regex patterns + VADER fallback
- **Problem**: Patterns miss nuanced sarcasm (e.g., "Yeah the room was clean I suppose" → no match, VADER neutral; "Great, a list. Very helpful." → no match, VADER positive), routing to neutral/positive instead of "frustrated".
- **Impact**: AI fails to empathize with sarcastic complaints, leading to mismatched tone (e.g., cheerful response to annoyed guest), delayed HEART escalation, and lost trust in high-stakes casino interactions.
- **Fix**: Add patterns for qualifiers like "I suppose", "I guess", and standalone "Very helpful/Very nice" with irony markers. Consider a post-VADER heuristic that checks for positive VADER + short sentence as a sarcasm signal.

### Finding 2 (MAJOR): VADER Misclassification Override Gap
- **Location**: `sentiment.py` detection flow (frustration → sarcasm → overrides → VADER)
- **Problem**: VADER classifies sarcastic positives as positive/neutral (e.g., "Great, a list. Very helpful."), bypassing the "all sarcasm → frustrated" rule since patterns don't catch it first.
- **Impact**: Undermines the entire sarcasm handling pipeline; guests venting sarcasm get "positive" energy mirroring, escalating frustration. No HEART intervention triggered.
- **Fix**: Add a post-VADER sarcasm heuristic (e.g., positive VADER score + frustration keywords in context = force re-route to "frustrated"). Or add "backhanded compliment" patterns.

### Finding 3 (MINOR): Limited Pattern Variety
- **Location**: 6 sarcasm regex patterns in `sentiment.py`
- **Problem**: Patterns are too anchored/specific (e.g., sentence-start requirements miss embedded sarcasm like "The service was, oh wonderful, slow"). Only 6 patterns for a domain where sarcasm is the default communication style.
- **Impact**: Reduces robustness in diverse guest dialogues, leading to inconsistent detection and uneven HEART escalations.
- **Fix**: Add 2-3 flexible patterns (e.g., non-anchored "oh [positive adj]", "Could have been worse", "I guess", "if you say so").

### Finding 4 (OUT OF SCOPE): No Real-Time Learning
- **Location**: N/A (absent feature)
- **Problem**: Static patterns don't adapt to new sarcasm styles.
- **Fix**: ML retraining feedback loop — feature request.

## B2: Implicit Intent & Context Reading — 3/10

**Justification**: This is extraction, not intent detection. Implicit needs are ignored unless they scream via VADER or regex — the AI is deaf to whispers, which is ironic given the "whisper planner." Only 2/5 adversarial cases handled well.

### Finding 1 (MAJOR): Implicit Complaint/Need Oversight
- **Location**: Extraction regex + VADER in `extraction.py`, whisper planner `conversation_note`
- **Problem**: Neutral VADER + no patterns miss key implicits (e.g., "We drove 4 hours" → no extraction/urgency; "Kids getting restless" → neutral, no kid-activity inference).
- **Impact**: AI ignores unspoken needs (e.g., no comp offers for travel fatigue, no family suggestions), leading to generic responses that frustrate guests and miss upselling opportunities.
- **Fix**: Add implicit intent patterns (e.g., regex for travel duration → flag "fatigue" in conversation_note) and prompt whisper planner LLM to infer actions ("If travel mentioned, suggest refreshments").

### Finding 2 (MAJOR): Loyalty Signal Gap
- **Location**: Absence of loyalty patterns in `extraction.py`
- **Problem**: No detection for implicit loyalty (e.g., "Momentum member, 20 years" → no pattern, treated as neutral). Missing chances to personalize via system prompt's "mirror energy" or perks.
- **Impact**: Alienates long-term guests, reducing retention. A 20-year member getting generic responses feels undervalued.
- **Fix**: Introduce loyalty regex (e.g., "[program] member, [X] years") to extract and feed into whisper planner for tailored responses. Update system prompt to prioritize loyalty in exploratory dialogues.

### Finding 3 (MINOR): Urgency Mismatch in Tone Guides
- **Location**: VADER + tone guides integration in `_base.py`
- **Problem**: Neutral VADER on urgent implicits (e.g., "Kids getting restless") means no shift to "brief for quick" mirroring, despite potential for frustration.
- **Impact**: Slow responses to time-sensitive needs worsen guest experience.
- **Fix**: Enhance tone guides with an "urgency" category (e.g., keywords like "restless/hurry" trigger brief, action-oriented responses).

### Finding 4 (OUT OF SCOPE): No Multi-Turn Implicit Building
- **Location**: N/A (whisper planner lacks accumulation)
- **Problem**: Single-turn extraction misses building implicits over conversation.
- **Fix**: Conversation_note accumulation — architecture change, out of scope.

## Severity Summary

| Severity | B1 | B2 | Total |
|----------|----|----|-------|
| MAJOR | 2 | 2 | 4 |
| MINOR | 1 | 1 | 2 |
| OUT OF SCOPE | 1 | 1 | 2 |
