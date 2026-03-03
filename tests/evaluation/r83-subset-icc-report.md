# R83-SUBSET ICC Report -- Multi-Model Judge Panel

**Date**: 2026-03-03 13:06 UTC
**Judges**: gemini, gpt52, grok4
**Scenarios**: 20
**Behavioral Dimensions**: B1_sarcasm, B2_implicit, B3_engagement, B4_agentic, B5_emotional, B6_tone, B7_coherence, B8_cultural, B9_safety, B10_overall
**Profiling Dimensions**: P1_natural_extraction, P2_active_probing, P3_give_to_get, P4_assumptive_bridging, P5_progressive_sequencing, P6_incentive_framing, P7_privacy_respect, P8_profile_completeness, P9_host_handoff, P10_cross_turn_memory

## Behavioral ICC(2,1) — B1-B10

| Dimension | ICC(2,1) | Interpretation |
|-----------|----------|----------------|
| B1_sarcasm | -0.256 | Poor |
| B2_implicit | 0.006 | Poor |
| B3_engagement | 0.013 | Poor |
| B4_agentic | 0.022 | Poor |
| B5_emotional | -0.069 | Poor |
| B6_tone | 0.013 | Poor |
| B7_coherence | 0.007 | Poor |
| B8_cultural | 0.245 | Poor |
| B9_safety | -0.000 | Poor |
| B10_overall | -0.018 | Poor |

## Profiling ICC(2,1) — P1-P10

| Dimension | ICC(2,1) | Interpretation |
|-----------|----------|----------------|
| P1_natural_extraction | 0.074 | Poor |
| P2_active_probing | 0.182 | Poor |
| P3_give_to_get | 0.078 | Poor |
| P4_assumptive_bridging | 0.085 | Poor |
| P5_progressive_sequencing | 0.146 | Poor |
| P6_incentive_framing | 0.130 | Poor |
| P7_privacy_respect | nan | N/A |
| P8_profile_completeness | 0.066 | Poor |
| P9_host_handoff | -0.050 | Poor |
| P10_cross_turn_memory | 0.076 | Poor |

## Overall ICC

| **Overall** | **-0.014** | **Poor** |

## Per-Dimension Averages — Behavioral (B1-B10)

| Dimension | gemini | gpt52 | grok4 | Consensus |
|-----------|----------|----------|----------|-----------|
| B1_sarcasm | 0.5 | 8.0 | 1.1 | **3.2** |
| B2_implicit | 1.1 | 5.2 | 3.7 | **3.3** |
| B3_engagement | 0.8 | 5.4 | 4.7 | **3.6** |
| B4_agentic | 0.6 | 4.7 | 4.2 | **3.2** |
| B5_emotional | 0.5 | 6.3 | 2.1 | **3.0** |
| B6_tone | 1.1 | 6.6 | 5.2 | **4.3** |
| B7_coherence | 0.9 | 6.4 | 5.4 | **4.2** |
| B8_cultural | 0.0 | 4.7 | 1.1 | **1.9** |
| B9_safety | 0.0 | 6.0 | 1.5 | **2.5** |
| B10_overall | 0.8 | 5.4 | 4.8 | **3.7** |
| overall | 0.8 | 5.3 | 4.7 | **3.6** |

## Per-Dimension Averages — Profiling (P1-P10)

| Dimension | gemini | gpt52 | grok4 | Consensus |
|-----------|----------|----------|----------|-----------|
| P1_natural_extraction | 0.0 | 2.4 | 3.3 | **1.9** |
| P2_active_probing | 0.3 | 3.5 | 3.9 | **2.6** |
| P3_give_to_get | 0.2 | 5.8 | 4.8 | **3.6** |
| P4_assumptive_bridging | 0.0 | 3.4 | 2.5 | **2.0** |
| P5_progressive_sequencing | 0.0 | 2.8 | 3.2 | **2.0** |
| P6_incentive_framing | 0.0 | 1.4 | 2.1 | **1.2** |
| P7_privacy_respect | 0.0 | 5.0 | 0.0 | **1.7** |
| P8_profile_completeness | 0.0 | 1.4 | 2.0 | **1.1** |
| P9_host_handoff | 0.0 | 2.4 | 0.8 | **1.1** |
| P10_cross_turn_memory | 0.5 | 4.6 | 4.6 | **3.2** |

## Safety Compliance

- Safety-relevant scenarios: 4
- Safety pass rate: 5/12 (42%)

## Summary

- **Behavioral average (B1-B10)**: 3.6/10
- **Profiling average (P1-P10)**: 2.0/10
- **ICC range**: -0.256 -- 0.245
- **ICC target (>0.7)**: NOT MET -- revise rubric or check model calibration