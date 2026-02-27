# ADR-024: Context-Contrast Sarcasm Detection

## Status
Accepted (2026-02-28)

## Context
R70 behavioral review scored B1 (Sarcasm Awareness) at 4.0/10. The agent responded to sarcastic
messages with forced enthusiasm ("I'm happy to help!") instead of recognizing the emotional subtext.

R72 deep domain research (A6: `research/r72-sarcasm-detection.md`) surveyed production chatbot
companies (Intercom, Zendesk, Ada, PolyAI) and academic literature. Key finding: production
systems do NOT classify sarcasm — they design sarcasm-resilient responses.

Best achievable sarcasm classification F1 is 0.82-0.85 with an ensemble (Ghosh et al., 2020),
but empathetic-by-default response design achieves 100% "correct" by construction — if the guest
is genuinely frustrated, empathy is correct; if they're being sarcastic, empathy is also correct
because the underlying emotion is negative.

## Decision
Context-contrast sarcasm detection: when VADER returns positive/neutral sentiment but recent
conversation history is negative, override to "frustrated" sentiment.

Implementation: `src/agent/sentiment.py::detect_sarcasm_context()`

### Algorithm
1. Current turn sentiment is positive or neutral (VADER)
2. Count consecutive negative/frustrated sentiments in recent message history
3. Check for positive signal words in current message ("great", "wonderful", "amazing")
4. If 2+ negative history + positive words: sarcasm detected → override to frustrated
5. If 1 negative + short message (<=8 words) + positive words: sarcasm detected

### Design Properties
- **Zero LLM cost**: Uses existing VADER scores and conversation history
- **Sub-1ms latency**: No API calls, set intersection and counting
- **Sarcasm-resilient by default**: Override to "frustrated" triggers empathetic response
- **No false positives on sincere messages**: Only triggers with negative conversation history

## Alternatives Considered

1. **LLM sarcasm classifier** — Rejected. $0.001-0.01 per classification, 500ms+ latency,
   F1 ≤ 0.85, false positives would patronize sincere guests.

2. **Fine-tuned BERT classifier** — Rejected. Requires labeled sarcasm dataset, ongoing
   maintenance, still achieves only 0.78-0.82 F1 without context.

3. **No sarcasm detection** — Rejected. Agent responds with enthusiasm to "Oh wonderful,
   another chatbot" — unacceptable for premium casino host experience.

## Consequences
- Positive: Empathetic responses for both sarcastic and genuinely frustrated guests
- Positive: Zero additional cost or latency
- Negative: Cannot detect sarcasm in first message (no conversation history yet)
- Negative: Cannot detect sarcasm when conversation history is all positive
- Accepted: Both edge cases are low-risk — a cheerful response to first-message sarcasm
  is the least bad option, and sarcasm after positive history is rare
