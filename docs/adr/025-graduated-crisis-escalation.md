# ADR-025: Graduated 4-Level Crisis Escalation

## Status
Accepted (2026-02-28)

## Context
The existing crisis detection was binary: detect self-harm → provide 988 Lifeline, or don't detect
→ continue normal conversation. This creates two problems:

1. **Under-response**: "I just need one more win to get back to even" (chasing losses) doesn't
   trigger the binary detector, but it's a recognized problem gambling indicator (NCPG protocols).

2. **Over-response**: "I can't face my wife after losing" (financial distress) receives the same
   988 Lifeline response as "I want to end my life" (suicidal ideation). Different crisis types
   require different response protocols.

R72 deep domain research (A3: `research/r72-crisis-scenarios.md`) identified 3 distinct crisis
types requiring different response protocols:
- **Loss-chasing/financial distress**: Concern-level. Needs empathy + responsible gaming helpline.
- **Intoxication/stranded**: Urgent-level. Needs practical help + human connection offer.
- **Suicidal ideation**: Immediate-level. Needs 988 Lifeline, Crisis Text Line, 911.

## Decision
4-level graduated crisis detection: `none | concern | urgent | immediate`

Implementation: `src/agent/crisis.py::detect_crisis_level()`

### Severity Levels and Response Mapping

| Level | Patterns | Response |
|-------|----------|----------|
| `immediate` | Suicidal ideation, active danger ("want to die", "end my life") | Stop ALL conversation. 988 Lifeline + Crisis Text Line + 911. |
| `urgent` | Financial desperation, self-harm references, stranded guest | Direct resource provision + offer human connection. |
| `concern` | Chasing losses, extended sessions, addiction language | Empathy + gentle responsible gaming helpline mention. Route to existing RG path. |
| `none` | No crisis indicators | Continue normal conversation. |

### Integration
- Runs in `compliance_gate_node` BEFORE the binary `detect_self_harm()` detector
- `concern` → routes to existing `gambling_advice` query_type (helplines included)
- `urgent` and `immediate` → routes to `self_harm` query_type (full crisis response)
- Check highest severity first (short-circuit): immediate → urgent → concern → none

### Pattern Design
- Immediate: 8 patterns — suicidal ideation keywords, active danger phrases
- Urgent: 12 patterns — financial desperation, self-harm references, stranded indicators
- Concern: 15 patterns — chasing losses, extended play, addiction language, denial patterns

## Alternatives Considered

1. **LLM crisis classifier** — Rejected for safety-critical path. Regex is deterministic,
   zero-latency, and fail-safe. LLM classifier could miss patterns or hallucinate non-crisis.

2. **Binary with expanded patterns** — Rejected. Gives suicidal and loss-chasing guests the
   same response (988 Lifeline for someone who lost at poker is heavy-handed and potentially
   alienating).

3. **5+ levels** — Rejected. 4 levels (none/concern/urgent/immediate) maps cleanly to
   3 response protocols. More granularity adds complexity without changing the response.

## Consequences
- Positive: More humane responses — loss-chasing gets empathy + helpline, not 988 Lifeline
- Positive: Catches problem gambling indicators the binary detector misses
- Positive: Deterministic, zero-cost, testable
- Negative: Pattern-based detection misses novel crisis language not in the pattern list
- Accepted: The binary detector still runs as a backup for anything the graduated system misses
