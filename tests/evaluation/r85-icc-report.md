# R85 ICC Report — Multi-Model Judge Panel

**Date**: 2026-03-03 17:18 UTC
**Judges**: gpt52, grok4
**Scenarios**: 20
**R84 Baseline**: B10=6.0, P-avg=2.0, Safety=42%

## Behavioral ICC(2,1) — B1-B10

| Dimension | gpt52 | grok4 | Consensus | ICC |
|-----------|--------|--------|-----------|-----|
| B1_sarcasm | 0.9 | 6.0 | **3.4** | 0.08 (Poor) |
| B2_implicit | 5.8 | 6.1 | **6.0** | 0.82 (Excellent) |
| B3_engagement | 5.5 | 6.3 | **5.9** | 0.84 (Excellent) |
| B4_agentic | 5.2 | 5.9 | **5.6** | 0.83 (Excellent) |
| B5_emotional | 5.3 | 6.5 | **5.9** | 0.80 (Excellent) |
| B6_tone | 6.5 | 7.1 | **6.8** | 0.68 (Good) |
| B7_coherence | 6.5 | 7.5 | **7.0** | 0.49 (Fair) |
| B8_cultural | 4.5 | 6.8 | **5.7** | 0.00 (Poor) |
| B9_safety | 8.1 | 9.2 | **8.6** | 0.81 (Excellent) |
| B10_overall | 5.8 | 6.6 | **6.2** | 0.81 (Excellent) |

## Profiling ICC(2,1) — P1-P10

| Dimension | gpt52 | grok4 | Consensus | ICC |
|-----------|--------|--------|-----------|-----|
| P1_natural_extraction | 3.3 | 5.6 | **4.4** | 0.68 (Good) |
| P2_active_probing | 3.0 | 5.5 | **4.2** | 0.58 (Fair) |
| P3_give_to_get | 2.2 | 5.4 | **3.8** | 0.52 (Fair) |
| P4_assumptive_bridging | 3.6 | 5.6 | **4.6** | 0.67 (Good) |
| P5_progressive_sequencing | 4.0 | 6.4 | **5.2** | 0.55 (Fair) |
| P6_incentive_framing | 1.6 | 5.9 | **3.8** | 0.17 (Poor) |
| P7_privacy_respect | 8.3 | 9.3 | **8.8** | 0.27 (Poor) |
| P8_profile_completeness | 2.5 | 4.8 | **3.6** | 0.61 (Good) |
| P9_host_handoff | 3.7 | 6.1 | **4.9** | 0.27 (Poor) |
| P10_cross_turn_memory | 5.2 | 7.0 | **6.1** | 0.62 (Good) |

## Summary

| Metric | R84 | R85 | Delta |
|--------|-----|-----|-------|
| **B10 overall** | 6.0 | **6.2** | **+0.2** |
| **P-average** | 2.0 | **5.0** | **+3.0** |
| **Safety** | 42% | **93%** | **+51pp** |

### Target Assessment
- B10 >= 7.5: **NOT MET** (6.2)
- P-avg >= 4.0: **MET** (5.0)
- Safety >= 80%: **MET** (93%)

### Top Improvements (R85 vs R84)
- Safety: 42% → 93% (+51pp) — BSA hardening + Spanish crisis fix
- Profiling: 2.0 → 5.0 (+3.0) — extraction prompt + incentive wiring + unconditional follow-up
- Response rate: 18% → 77% — API key fix + anti-deflection

### Remaining Gaps
- B4 agentic (5.6): Cross-domain suggestions present but not specific enough
- B1 sarcasm (3.4): GPT-5.2 scores most as -1 (not testable), inflating gap
- P8 completeness (3.6): Extraction improved but still below target
- P6 incentive (3.8): Incentive engine wired but completeness gate (50%) rarely met

### Deflection Pattern
- Still appears in 3/20 scenarios (agentic-09, implicit-09, overall-01)
- Down from 7/20 in R84 — anti-deflection instruction helps but fallback responses still deflect