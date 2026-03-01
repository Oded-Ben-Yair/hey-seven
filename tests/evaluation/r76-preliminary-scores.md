# R76 Preliminary Score Assessment

## Changes Applied (Rounds 1-4)

### Infrastructure Fixes (Affect P1-P10 and B1-B10)
1. ProfileExtractionOutput: 19 nested → 16 flat fields (profiling was 100% dead)
2. DispatchOutput.reasoning: 200→500 chars (dispatch was falling back to keyword every turn)
3. WhisperPlan: 10 complex → 6 flat fields (whisper planner was 100% dead)
4. ValidationResult.reason: required → optional default (unnecessary degraded-pass)

### Behavioral Fixes (Affect B1-B10)
5. Grief sentiment priority guard (VADER was overwriting grief→neutral)
6. Celebration detection + priority guard + tone guide
7. Router prompt: emotional context → property_qa (not ambiguous/off_topic)
8. Validation prompt: lenient for cross-domain suggestions
9. System prompt: anti-pattern section, no "Oh,", no slop adjectives
10. Fallback node: concise redirect instead of corporate redirect

### Expected Impact by Dimension

| Dim | R75 Score | Expected R76 | Why |
|-----|-----------|-------------|-----|
| D1 | 9.5 | 9.5-10.0 | Architecture unchanged, cleaner schemas |
| D2 | 9.5 | 9.5 | RAG unchanged |
| D3 | 9.5 | 9.5-10.0 | Simpler schemas, same reducers |
| D4 | 10.0 | 10.0 | No API changes |
| D5 | 10.0 | 10.0 | 3236 tests maintained |
| D6 | 9.5 | 9.5 | No Docker changes |
| D7 | 9.5 | 9.5 | No guardrail pattern changes |
| D8 | 9.5 | 9.5 | No scalability changes |
| D9 | 10.0 | 9.5-10.0 | Learning log added, may need ADR update |
| D10 | 9.5 | 9.5 | No domain data changes |
| B1 | 8.0 | 8.0 | No sarcasm detection changes |
| B2 | 5.0 | 7.0-8.0 | Profiling extraction now works, router better |
| B3 | 5.0 | 7.0-8.0 | Anti-pattern removal, terse response matching |
| B4 | 5.0 | 7.0-8.0 | Whisper planner now works, suggestion injection |
| B5 | 4.0 | 7.0-9.0 | Crisis persistence works, grief/celebration detection |
| B6 | 6.0 | 7.0-8.0 | Anti-pattern removal, no "Oh!" openers |
| B7 | 9.0 | 9.0 | No memory changes |
| B8 | 7.0 | 7.0 | No multilingual changes |
| B9 | 3.0 | 7.0-9.0 | Crisis persistence works, grief detection |
| B10 | 6.0 | 7.0-8.0 | Composite improvement |
| P1 | 0* | 6.0-8.0 | Profiling extraction now works (was 100% dead) |
| P2 | 0* | 5.0-7.0 | Whisper planner now works (was 100% dead) |
| P3 | 0* | 5.0-7.0 | System prompt has give-to-get guidance |
| P4 | 0* | 4.0-6.0 | System prompt has assumptive bridge guidance |
| P5 | 0* | 5.0-7.0 | Golden path logic works when profiling fires |
| P6 | 0* | 4.0-6.0 | Incentive engine exists but untested live |
| P7 | 0* | 6.0-8.0 | System prompt has privacy respect guidance |
| P8 | 0* | 4.0-6.0 | Depends on P1+P2 working together |
| P9 | 0* | 5.0-7.0 | Handoff request model exists |
| P10 | 0* | 6.0-8.0 | extracted_fields persists via _merge_dicts reducer |

*P1-P10 were never scored before; "0" = profiling was dead due to schema bug.

### Bottom Line
- Technical dimensions: Expected to remain at 9.5+
- Behavioral dimensions: Expected +2-4 points across the board due to infrastructure unfreezing
- Profiling dimensions: Expected 5-7 average (from 0) — first time they can fire at all
