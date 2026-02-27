# ADR-023: 3-Tier Reproducible Evaluation Methodology

## Status
Accepted (2026-02-27)

## Context
After 70 review rounds, Hey Seven's scores oscillate between 67-97 with standard deviation
~7.5-15 per round across 4 external model families. Root causes identified:

1. **Forced-finding quotas** ("minimum 5 findings, look harder") manufacture criticisms at ceiling
2. **Previous-score anchoring** (`{previous_scores_table}`) biases new scores toward old ones
3. **Hostile framing** ("HOSTILE reviewer") primes for negativity and severity inflation
4. **Spotlight severity bumps** change the measurement instrument between rounds
5. **Specialist dimension assignment** (model X scores D1-D5 only) prevents ICC calculation
6. **Model version drift** across rounds changes the reviewer without documentation

These are fundamental measurement design flaws, not code quality issues. The current system
cannot converge to a stable score — it measures "how well code pleases a specific LLM
configuration on a specific day," not code quality.

## Decision
Replace the ad-hoc LLM review system with a 3-tier reproducible evaluation:

### Tier 1: Deterministic Quality Gates (Automated)
Hard assertions in `tests/test_doc_accuracy.py` that replace LLM judgment for measurable facts:
- D5: test_count >= 2500, coverage config exists, zero xfails
- D6: Dockerfile multi-stage, non-root, require-hashes, HEALTHCHECK /live, exec-form CMD
- D7: 204 patterns, 6 categories, 145 confusables, re2-compatible
- D8: No threading.Lock in async paths, TTLCache with jitter for LLM singletons
- D9: ADR count >= 22, all ADRs have Status+Date, version parity

Tier 1 must pass before any Tier 2 evaluation.

### Tier 2: Frozen LLM Evaluation (Semi-Automated)
`docs/eval-prompt-v2.0.md` — a frozen, version-controlled prompt that removes all
score-inflation mechanisms (no quotas, no anchoring, no hostile framing, no spotlights).
Key changes:
- Calibration anchors (3/6/9 examples) reduce between-model variance
- All 4 models score ALL dimensions (enables ICC calculation)
- "Score from evidence" replaces "find problems"
- Model version IDs recorded per review

### Tier 3: Behavioral Scenario Evaluation (Execution-Based)
50 adversarial behavioral scenarios across 5 dimensions (B1-B5) in
`tests/scenarios/behavioral_*.yaml`. Evaluated against a live or mock agent for:
- Tone calibration (sarcasm, grief, celebration)
- Implicit signal extraction (urgency, loyalty, fatigue)
- Engagement repair (repetition, pivots, brevity)
- Agentic initiative (proactive suggestions, domain tracking)
- Emotional intelligence (grief, anxiety, allergy safety)

## Consequences

### Positive
- Tier 1 eliminates ~50% of review round findings (dimension drift, count mismatches)
- Tier 2 enables ICC measurement — inter-rater reliability becomes quantifiable
- Tier 3 measures agent BEHAVIOR, not code structure
- Scores converge (target SD < 2 across consecutive rounds)
- Stopping criterion becomes measurable: 0 CRITICALs for 3 rounds AND Tier 1 green AND ICC > 0.7

### Negative
- Tier 1 tests are brittle to intentional changes (update assertions when changing counts)
- Tier 2 frozen prompt cannot adapt to new dimension discoveries without version bump
- Tier 3 scenarios require maintenance as property data changes

### Measurement Targets
| Metric | Current | Target |
|--------|---------|--------|
| Score SD (consecutive rounds) | 7.5-15 | < 2 |
| ICC (inter-rater reliability) | Unmeasurable | > 0.7 |
| Forced-finding false positives | ~30% of MINORs | 0 |
| Time-to-converge | Never (oscillates) | 3-4 rounds |

## References
- Prompt template: `docs/eval-prompt-v2.0.md`
- Tier 1 tests: `tests/test_doc_accuracy.py` (TestDeterministicD5-D9 classes)
- Behavioral scenarios: `tests/scenarios/behavioral_*.yaml` (50 scenarios, 5 files)
- Previous review prompt: `docs/review-prompt-template.md` (v1.0, superseded)

## Review
Last reviewed: 2026-02-27
Next review: After first Tier 2 evaluation with new prompt
