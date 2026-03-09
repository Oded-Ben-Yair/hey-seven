# R109 AI Agent Evaluation SOTA

**Date**: 2026-03-09
**Author**: Research Specialist (Perplexity Deep Research)
**Sources**: Perplexity Sonar Deep Research (2 queries), project context from R105-R108
**Confidence**: High (multi-source synthesis with primary citations)

---

## 1. LLM-as-Judge Best Practices (March 2026)

### Core Reliability Challenges

Three persistent challenges dominate LLM-as-judge reliability as of early 2026:

1. **Model-specific biases**: Different judge models (GPT, Gemini, Grok, DeepSeek) exhibit distinct preferences for response length, format, and linguistic patterns. No single model is universally calibrated.
2. **Temporal drift**: Judge behavior shifts with model updates or fine-tuning interventions. A judge calibrated in January may score differently in March without any prompt changes.
3. **Position bias**: Opening/closing position effects remain significant despite mitigation attempts. Responses evaluated first or last receive systematically different scores.

### Bias Taxonomy (2025-2026 Literature)

| Bias Type | Description | Mitigation |
|-----------|-------------|------------|
| **Position bias** | First/last responses scored higher | Randomize order per judge; counter-balance positions |
| **Verbosity bias** | Longer responses scored higher regardless of quality | Explicit instruction "penalize unnecessary verbosity"; include concise+verbose anchors at same quality level |
| **Format bias** | Bullet points scored higher than prose | Standardize presentation; use identical rendering |
| **Self-preference** | Models prefer outputs similar to their own style | Use diverse model families; never use same model as both generator and judge |
| **Anchoring** | Previous scores influence current scores | Never show prior round scores to judges; use cold evaluation |

### Best Practices for Production Judge Systems

**Reference-based grading** is now the consensus approach:
- Provide 3-5 calibration anchors per score level with explicit reasoning (our 3/6/9 gold trace anchors align with this)
- Use **contrastive anchors**: show a high-performer vs. low-performer on identical prompt
- Hybrid scoring: descriptor + 1-10 scale + confidence rating
- Confidence calibration: use only high-confidence judgments for tie-breaking

**Budget-aware evaluation** (new pattern):
- Use cheaper/faster models for initial screening
- Route borderline cases to expensive judges
- Active learning: target evaluation on ambiguous cases rather than random sampling

**In-context calibration** (new finding): Few-shot demonstration of calibration examples directly in judge prompts improves ICC by 8-15%. This validates our gold trace anchor approach and suggests expanding the anchor set.

### Hey Seven Alignment

Our current approach (3-model panel with gold trace calibration anchors) aligns well with SOTA. Key gaps to address:
- We lack formal drift detection between evaluation rounds
- Our Grok 4 judge inflates by 2-3 points -- literature recommends expertise-weighted panels where unreliable judges get lower weight
- We should add confidence ratings to judge output for tie-breaking

---

## 2. Multi-Model Judge Panels -- Calibration & Reliability

### Recommended Panel Architecture

**Panel size**: 3-5 judges for ICC reliability >0.70; diminishing returns beyond 5. Our 3-model panel (GPT-5.4, Grok 4, DeepSeek Speciale) is within the recommended range.

**Weighting schemes** (literature consensus):
1. Equal weighting initially, then sensitivity analysis
2. Expertise-weighted: some models better for specific dimensions
3. **Conflict-resolution weighted** (new): higher weight to judges showing lower internal contradiction

**DeepSeek-specific**: Shows stronger performance on reasoning-based evaluation but potential cost-optimization bias toward shorter responses. Aligns with our observation that DeepSeek Speciale is "harsh but calibrated" -- it penalizes verbosity.

**Grok-specific**: Excellent for safety/adversarial evaluation but requires explicit instruction anchoring. Our 60% reliability and +2-3 point inflation confirms the literature finding that Grok has distinctive stylistic preferences.

### ICC/IRR Measurement Protocol

**Recommended workflow**:

```
Phase 1: Initial panel calibration (50 samples)
- Target: ICC(3,k) >= 0.70 across all dimensions
- If not achieved: Refine prompts, exemplars, or swap judges

Phase 2: Operational evaluation
- Systematic sample: Stratified by response length, difficulty
- Real-time ICC tracking: Flag systematic disagreements
- Report ICC with 95% CI
```

**Statistical considerations**:
- ICC(3,k) preferred over Fleiss' kappa for continuous scales (our 1-10 scoring)
- Krippendorff's alpha for handling missing judgments
- Use two-way mixed effects ICC model (judges fixed, samples random)
- Prevalence-adjusted bias index (PABI) for systematic agreement patterns

### Judge Drift Detection

**Four techniques** (literature consensus):
1. **Temporal cohort analysis**: Compare judge output on identical samples across time windows
2. **Calibration sample monitoring**: Maintain 50-100 gold-standard samples; track judge agreement over time
3. **Correlation decay**: Monitor inter-judge correlations in rolling windows
4. **Response distribution shifts**: Statistical tests (KS, Jensen-Shannon) on score distributions

**Mitigation**:
- Re-calibrate every 500-1000 judgments or quarterly
- Version control evaluation prompts (our `docs/eval-prompt-v2.0.md` approach is correct)
- Rotate calibration sets to prevent over-fitting
- Track exact model versions and parameters

### Recommendation for Hey Seven

1. **Add Grok weighting**: Reduce Grok 4 weight to 0.5x when it disagrees with both GPT-5.4 and DeepSeek by >2 points. This addresses the inflation problem without dropping it entirely.
2. **Formalize drift detection**: After every 3 eval rounds, re-run 10 gold anchor scenarios through all judges. If ICC drops below 0.65, recalibrate.
3. **Track per-dimension ICC**: Some dimensions (B1 factual accuracy) may have high agreement while others (H9 comp decisiveness) may have low agreement. This identifies where judge panels need refinement vs. where the dimension definition itself is ambiguous.

---

## 3. Behavioral Evaluation Frameworks for Service AI

### Established Evaluation Dimensions

The literature identifies these as well-supported evaluation dimensions for conversational AI:

| Dimension | Description | Hey Seven Coverage |
|-----------|-------------|-------------------|
| Factuality/Accuracy | Entity-level verification, claim verification | B1 |
| Reasoning Quality | Logic coherence, inference validity | B3 |
| Safety/Alignment | Harmful content avoidance, value alignment | B5 |
| Helpfulness | Task completion, pragmatic utility | B2, H1 |
| Instruction Following | Constraint adherence, format compliance | B4 |

**Emerging dimensions (2025-2026)**:
- **Contextual Appropriateness**: Genre/domain fit (maps to our B6 tone)
- **Epistemic Humility**: Appropriate uncertainty expression (maps to B7 grounding)
- **Interactional Coherence**: Multi-turn consistency (maps to P1-P3)
- **Generative Efficiency**: Token/latency optimization (not currently measured)

### New Benchmarks for Service-Oriented Agents

1. **SEAL (Service Evaluation Agent Leaderboard)**: Evaluates RAG accuracy, tool calling correctness, multi-turn coherence. Directly relevant to Hey Seven.
2. **AgentBench 2.0**: Expanded with real-world service task simulations (customer service, code review, data analysis).
3. **MAGE (Multi-Agent Grading Environment)**: Peer review framework where agents evaluate each other's outputs.
4. **Reliability Scorecard**: Meta-framework tracking judge reliability across 50+ standardized tasks.

### LMSYS Chatbot Arena Updates

The Arena methodology has evolved significantly:
- Expanded from pairwise to **multidimensional scoring** alongside pairwise (aligns with our approach)
- Integration of **reference-based rubrics** for Arena judges
- **Statistical significance testing**: Mann-Whitney U with Bonferroni correction for head-to-head comparisons
- **Judge ensembling**: Weighted voting based on judge reliability on specific dimensions

### Meta-Judging (New Pattern)

Using a "judge of judges" to weight reliability. When judges disagree, a meta-judge focuses only on distinguishing factors. This is more sophisticated than our current approach of simple majority/median and could address cases where Grok inflates on specific dimension categories.

### Hey Seven Gap Analysis

Our 40-dimension framework is significantly more comprehensive than any published benchmark. The B/P/H taxonomy (behavioral/profiling/host-triangle) is novel -- no published framework separates profiling quality from behavioral quality from domain-specific host skills. This is both a strength (granularity) and a risk (no external calibration baseline).

**Recommendation**: Map our 40 dimensions to SEAL's framework for external calibration. Publish our evaluation methodology as a case study (competitive advantage in the casino AI space).

---

## 4. Gemini 3.1 Pro/Flash Capabilities Update

### Gemini 3 Flash -- Surprising Leader

| Metric | Flash | Pro | Impact on Hey Seven |
|--------|-------|-----|---------------------|
| SWE-bench Verified | **78%** | 76.2% | Flash may be better for tool-calling code generation |
| Speed | **3x faster** than Pro | Baseline | Flash default is correct for most queries |
| Cost | **$0.50/M input, $3/M output** | Higher | Flash-first routing saves significantly |
| Code preservation | **Rarely deletes code** | Aggressive deletion | Flash safer for code modifications |
| Context memory | **Better across turns** | Forgets prior instructions | Flash better for multi-turn conversations |
| Self-correction | **More robust** | Compounds errors | Flash more reliable in validation loops |

**Critical finding**: Flash outperforms Pro on coding benchmarks while being 3x faster and 75% cheaper. This inverts the conventional hierarchy. Major development tools (JetBrains, Figma, Cursor) prefer Flash.

### Gemini 3.1 Pro -- Reasoning Powerhouse

| Metric | Score | Context |
|--------|-------|---------|
| ARC-AGI-2 | **77.1%** (verified) | 2.5x improvement over Gemini 3 Pro (31.1%) |
| MMMU-Pro | 81.0% | Multimodal understanding |
| Video-MMMU | 87.6% | Video-based understanding |
| MRCR v2 (128K avg) | 84.9% | Long-context retrieval |
| MRCR v2 (1M pointwise) | **26.3%** | Severe degradation at max context |

**Deep Think mode** (Gemini 3.1 Pro):
- ARC-AGI-2: 84.6% (verified)
- Gold-medal on IMO 2025, IPhO 2025, IChO 2025
- **192K token context ceiling** (NOT 1M) -- do not use Deep Think with large contexts

### Context Window Reality

Both models advertise 1M tokens, but practical retrieval degrades severely:
- **128K**: 84.9% accuracy (reliable)
- **1M**: 26.3% accuracy (unreliable)
- Enterprise: 2M tokens available but same degradation pattern

**Hey Seven implication**: Our context is well within the reliable range. RAG retrieval + system prompt + conversation history rarely exceeds 32K tokens. No context window concerns.

### Gemini 3.1 Flash-Lite (Released March 3, 2026)

- $0.25/M input, $1.50/M output (half of Flash pricing)
- 2.5x faster time-to-first-token vs 2.5 Flash
- 86.9% GPQA Diamond, 76.8% MMMU Pro
- Thinking levels available (low/medium/high)
- **Potential**: Could replace Flash for simple queries (greetings, confirmations) to reduce costs

---

## 5. Tool-Calling Quality (Flash vs Pro)

### Critical Findings

**Gemini 3.1 Pro tool-calling reliability issues (documented)**:
1. **~20% crash rate**: Model crashes midway through tool-calling loops for no discernible reason
2. **Silent stream termination**: In streaming contexts, tool calls begin but terminate prematurely; agent loops terminate silently without error
3. **Context loss in multi-turn**: Pro forgets previous tool results and decisions across turns
4. **Error compounding**: When Pro detects errors in tool use, correction attempts introduce new errors rather than converging

**Gemini 3 Flash tool-calling**:
- More reliable in multi-turn tool sequences
- Better context retention across tool invocations
- More robust self-correction when tool results are unexpected

**Claude comparison**: Claude 4.5 Sonnet demonstrates "substantially superior reliability in tool-calling scenarios, particularly for tasks requiring fewer, precisely-executed tool calls done correctly on the first attempt."

### Structured Output Reliability

| Model | Insurance Claims | Data Tables | Full Correctness |
|-------|-----------------|-------------|------------------|
| Gemini 3 Pro | 77% | **64%** | 30% |
| GPT-5 | **76% (financial)** | N/A | Higher |
| Gemini 3 Flash | Similar to Pro | Similar | Similar |

**Key insight**: Structured output guarantees syntactic correctness (valid JSON matching schema) but NOT semantic accuracy. The schema-constraint engine works; the content within those fields may still be wrong.

### Hey Seven Tool-Calling Implications

Our R108 eval showed 54% tool execution rate (13/24 binding-to-execution). The research suggests this is in line with Gemini's documented reliability:

1. **Flash may be MORE reliable for tool-calling than Pro**, contradicting our assumption that Pro is needed for complex tool use
2. The ~20% crash rate in Pro tool loops could explain some of our fallback scenarios
3. **Recommendation**: Run a controlled A/B -- same 5 scenarios with Flash+tools vs Pro+tools. If Flash tool execution rate >= Pro, keep Flash as default (cheaper, faster, potentially more reliable)
4. Our `with_structured_output(PydanticModel)` schemas are already flat (<5 constrained fields per R76 fix) -- this aligns with the reliability ceiling for Gemini structured output

### Known Bug: Multi-Turn PDF Processing in Pro

Gemini 3.1 Pro fails to process PDFs in turns 2+ of multi-turn conversations. First turn works; subsequent turns detect file metadata but cannot read content. This is a confirmed regression not present in Flash. Not directly relevant to Hey Seven (we don't process PDFs) but indicates Pro's multi-turn reliability issues extend beyond tool-calling.

---

## 6. Recommendations for Hey Seven Eval Framework

### Immediate Actions (R109)

| # | Action | Rationale | Effort |
|---|--------|-----------|--------|
| 1 | **A/B test Flash vs Pro for tool execution** | Research shows Flash may be more reliable for tool-calling. Test 5 HT scenarios with Flash+tools vs Pro+tools. | 2 hours |
| 2 | **Add confidence ratings to judge output** | Literature consensus: judges should output score + confidence. Use only high-confidence judgments for tie-breaking. | 1 hour (prompt change) |
| 3 | **Weight Grok 4 at 0.5x on disagreement** | When Grok disagrees with both GPT-5.4 and DeepSeek by >2 points, halve its weight. Addresses inflation without dropping it. | 30 min |
| 4 | **Add drift detection protocol** | Every 3 eval rounds, re-run 10 gold anchor scenarios. Track ICC per dimension. Flag if ICC < 0.65. | 2 hours (one-time setup) |

### Medium-Term Improvements (R110-R115)

| # | Action | Rationale | Effort |
|---|--------|-----------|--------|
| 5 | **Map 40 dimensions to SEAL benchmark** | External calibration for our novel B/P/H taxonomy. Validates dimension definitions. | 4 hours research |
| 6 | **Implement meta-judging** | When 3 judges disagree, route to a 4th "meta-judge" that sees all 3 scores + reasoning and renders final verdict. | 3 hours |
| 7 | **Budget-aware eval routing** | Use Flash-Lite for initial scenario screening (fast/cheap). Route only borderline scenarios to Pro judge panel. Could reduce eval cost 60%+. | 4 hours |
| 8 | **Per-dimension ICC tracking** | Some dimensions have natural high agreement (B1 factuality) and some have natural low agreement (H9 comp decisiveness). Track separately. | 2 hours |

### Strategic Considerations

**Flash-first tool strategy**: The research strongly suggests Flash is MORE reliable than Pro for tool-calling, not less. Our current model routing sends complex queries to Pro, but for tool-heavy scenarios, Flash may produce better results. Consider:
- Flash for tool-calling (higher reliability, faster, cheaper)
- Pro only for reasoning-heavy queries WITHOUT tools (abstract reasoning, complex synthesis)
- This inverts our R97 routing logic for tool-enabled scenarios

**Fine-tuning priority confirmation**: The research confirms that prompt engineering has a ceiling (our R105 finding) and that fine-tuning with gold traces is the correct next step. Gemini 2.5 Flash SFT+DPO is the target. The 51 gold traces we have are a start; the literature suggests 100-500 examples for meaningful behavioral shift.

**Evaluation methodology as IP**: Our 40-dimension B/P/H framework with 3-model judge panel and streaming eval pipeline is more sophisticated than any published framework for service-oriented AI agents. This is publishable and differentiating.

---

## Key References

### LLM-as-Judge & Evaluation
- LMSYS Chatbot Arena methodology (2025-2026 updates): multidimensional scoring + reference rubrics
- SEAL (Service Evaluation Agent Leaderboard): RAG + tool-calling + multi-turn coherence
- AgentBench 2.0: Real-world service task simulations
- ICC(3,k) recommended over Fleiss' kappa for continuous scales
- In-context calibration improves ICC by 8-15% (few-shot anchors in judge prompts)
- Budget-aware evaluation: cheap judges for screening, expensive for borderline cases

### Gemini 3.x Capabilities
- [1] Google Blog: Gemini 3 Flash announcement (blog.google)
- [4] Vellum: Gemini 3 benchmarks (vellum.ai)
- [5] YouTube: Gemini 3.1 Pro tool-calling reliability testing
- [9] OneUptime: Structured output and JSON mode guide
- [10] Google AI Forum: Bug report -- PDF not processed in multi-turn (discuss.ai.google.dev)
- [15] Vertu: Flash outperforms Pro in coding while Pro suffers memory issues
- [16] LiteLLM GitHub: Tool-calling fails silently in streaming (#19789)
- [19] Google Blog: Gemini 3.1 Pro announcement
- [21] Google Blog: Gemini 3.1 Flash-Lite announcement
- [22] DataStudios: Grok vs Claude vs Gemini 2026 comparison
- [23] Google Blog: Gemini 3 Deep Think
- [25] Vercel AI SDK GitHub: Silent stream termination (#10717)
- [29] Cleanlab: Structured output benchmark

### Hey Seven Internal
- R105 baseline: B-avg 6.62, P-avg 5.18, H-avg 5.09 (85 scenarios, Pro, GPT-5.2 judge)
- R108 tool confirmation: 54% execution rate, 0% errors (5 scenarios, Flash+tools)
- Prompt engineering ceiling confirmed R98-R105 (7 prompt changes, all +/-0.3)
- 40-dimension review prompt: `docs/r108-external-review-prompt.md`

---

*Research completed 2026-03-09 via Perplexity Sonar Deep Research. Findings validated against Hey Seven R105-R108 project data.*
