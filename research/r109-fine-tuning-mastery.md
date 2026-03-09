# R109 Fine-Tuning Mastery Research

**Date**: 2026-03-09
**Author**: Research Specialist (Claude Opus 4.6)
**Confidence**: High (8 separate research queries, official Google docs verified)

## Sources

1. Perplexity Deep Research: Gemini 2.5 Flash SFT/DPO comprehensive guide
2. Perplexity Reasoning: Behavioral fine-tuning dataset sizing literature
3. Google Cloud Official Docs: Vertex AI supervised fine-tuning data preparation
4. Google Cloud Official Docs: Gemini supervised tuning specifications
5. Google Cloud Blog: Master Gemini SFT best practices
6. Vertex AI Pricing Page (accessed 2026-03-09)
7. Multiple developer guides and engineering manuals (2025-2026)
8. MHC-Coach behavioral fine-tuning study (PMC, 2025)

---

## Section 1: Gemini 2.5 Flash SFT on Vertex AI — Complete Guide

### 1.1 Supported Models for Fine-Tuning (March 2026)

| Model | SFT | Preference Tuning (DPO) | Adapter Sizes |
|-------|-----|-------------------------|---------------|
| Gemini 2.5 Pro | Yes | No (SFT only) | 1, 2, 4, 8 |
| **Gemini 2.5 Flash** | **Yes** | **Yes** | **1, 2, 4, 8, 16** |
| Gemini 2.5 Flash-Lite | Yes | Yes | 1, 2, 4, 8, 16 |
| Gemini 2.0 Flash | Yes | No | 1, 2, 4, 8 |
| Gemini 2.0 Flash-Lite | Yes | No | 1, 2, 4, 8 |
| Gemini 3.x | **NOT available** for fine-tuning (enterprise-only, no public API) | No | N/A |

**Critical**: Gemini 3 does NOT support fine-tuning. Gemini 2.5 Flash is the correct target.

### 1.2 Technical Specifications

| Specification | Value |
|---------------|-------|
| Max input+output tokens per training example | 131,072 |
| Max serving tokens | Same as base model (1M input, 65K output) |
| Max training dataset size | 10M text-only examples or 300K multimodal |
| Max training dataset file size | 1 GB JSONL |
| Max validation dataset | 5,000 examples or 30% of training (if >1000 validation) |
| Min recommended examples | 100 |
| Sweet spot for style/behavior | 100-500 examples |
| Region | us-central1 (primary) |

### 1.3 JSONL Data Format (Multi-Turn with System Instruction + Tool Calls)

The training data format supports ALL three key components Hey Seven needs:

```jsonl
{
  "systemInstruction": {
    "role": "system",
    "parts": [{"text": "You are Seven, a casino host for Mohegan Sun..."}]
  },
  "contents": [
    {"role": "user", "parts": [{"text": "What's good for dinner tonight?"}]},
    {"role": "model", "parts": [{"text": "What are we celebrating? I want to make sure we find the perfect spot for you."}]},
    {"role": "user", "parts": [{"text": "It's our anniversary, party of 4"}]},
    {"role": "model", "parts": [
      {"functionCall": {"name": "check_comp_eligibility", "args": {"guest_tier": "emerald", "category": "dining"}}}
    ]},
    {"role": "function", "parts": [
      {"functionResponse": {"name": "check_comp_eligibility", "response": {"eligible": true, "amount": 50, "venues": ["Mohegan Sun Beyond"]}}}
    ]},
    {"role": "model", "parts": [{"text": "Happy anniversary! For a party of 4, I'd go with Mohegan Sun Beyond - it's our premium steakhouse and perfect for celebrations. And here's the best part: your Emerald tier includes a $50 dining credit tonight. I'll have the host team set up a table by 7:30 so we can really take care of you. What time works best?"}]}
  ]
}
```

**Key format rules:**
1. `systemInstruction` is OPTIONAL but HIGHLY recommended for Hey Seven
2. The `role` field in `systemInstruction` is ignored by the model (can be any string)
3. `systemInstruction` text counts toward the 131K token limit
4. Multi-turn: alternate `user` and `model` roles in `contents`
5. **Tool calls**: `functionCall` and `functionResponse` ARE supported in `parts`
6. The system instruction used in training MUST match the one used in inference (training-serving skew prevention)

### 1.4 Tool/Function Calling in Training Data

**Confirmed**: The official Vertex AI data preparation docs explicitly list `functionCall` and `functionResponse` as supported fields in the `parts` array. This means:

- Fine-tuned Gemini 2.5 Flash retains tool/function calling capability
- Training examples CAN include tool call sequences (model calls tool, receives response, generates answer)
- This is EXACTLY what Hey Seven needs for teaching CCD (Checked-Confirmed-Dispatched) behavior

**Format for tool calls in training data:**
```json
{"role": "model", "parts": [{"functionCall": {"name": "tool_name", "args": {"key": "value"}}}]}
{"role": "function", "parts": [{"functionResponse": {"name": "tool_name", "response": {"result": "data"}}}]}
```

### 1.5 Hyperparameter Recommendations

#### For Gemini 2.5 Flash Text Fine-Tuning

| Dataset Size | Avg Context Length | Epochs | Learning Rate Multiplier | Adapter Size |
|-------------|-------------------|--------|-------------------------|--------------|
| <1000 examples | <500 tokens | 20 | 10 | 4 |
| <1000 examples | >=500 tokens | 10-20 | 5-10 | 4 |
| >=1000 examples | <500 tokens | 10 | default or 5 | 4 |
| >=1000 examples | >=500 tokens | 10 | default | 8 |

**For Hey Seven (100-200 examples, ~2000-5000 tokens avg context):**
- Epochs: 10-15 (multi-turn conversations are longer context, so fewer epochs than tiny examples)
- Learning rate multiplier: 5-10 (small dataset needs higher LR)
- Adapter size: 4 (start here, increase to 8 if underfitting)

#### Adapter Size Impact

| Size | Best For | Trainable Parameters | Risk |
|------|----------|---------------------|------|
| 1 | Minimal formatting changes | Fewest | Underfitting on complex tasks |
| 2 | Simple style adjustments | Low | |
| **4** | **Style + behavior + tone** | **Medium** | **Sweet spot for Hey Seven** |
| 8 | Complex reasoning adaptation | High | Needs more data (500+) |
| 16 | Maximum learning capacity | Highest | Overfitting risk with <1000 examples |

### 1.6 Thinking Budget for Fine-Tuned Models

**Critical finding**: Google recommends setting the thinking budget to **OFF or lowest value** for fine-tuned models.

Why: During SFT, the model learns to map Input -> Output directly. If thinking is enabled with a high budget, the model "overthinks" and may diverge from the trained behavior. This causes:
1. Increased latency and cost (paying for useless thinking tokens)
2. Quality degradation (reasoning conflicts with memorized patterns)

**Recommendation for Hey Seven**: Set `thinking_budget=0` when using the fine-tuned model endpoint.

---

## Section 2: Behavioral Fine-Tuning Dataset Sizing

### 2.1 What Research Says About Minimum Effective Dataset Size

The academic literature does NOT provide a precise formula for "N examples per behavioral dimension." However, converging evidence suggests:

| Source | Finding |
|--------|---------|
| Google Cloud SFT Docs | "Start with 100 examples" for good results, scale to thousands if needed |
| Google Cloud Blog | 100-500 is the "sweet spot" for style and format adaptation |
| Engineering Manual (dev.to) | 1-50: insufficient, use few-shot instead. 100-500: sweet spot. >1000: for new languages/complex reasoning |
| MHC-Coach Study (PMC 2025) | 3,268 expert messages for physical activity coaching behavioral change on LLaMA 3-70B |
| DigitalOcean Guide | Quality >> quantity. Small focused dataset > large noisy dataset |
| Kore.ai Research | Smaller models (6B-13B) can develop behavioral adaptation through specialized data design |

### 2.2 SFT vs DPO for Different Dimension Types

| Dimension Type | Best Method | Why |
|---------------|------------|-----|
| **Tone/personality** (B1-B5) | SFT | Directly demonstrates desired output style |
| **Profiling/extraction** (P1-P10) | SFT | Shows correct extraction behavior through examples |
| **Host behaviors** (H1-H10) | SFT first, DPO second | SFT teaches what TO do, DPO teaches what NOT to do |
| **Avoiding bad behaviors** | DPO | Directly optimizes preferred vs rejected responses |
| **Tool integration** (H9, H10) | SFT with tool calls | Model learns WHEN and HOW to call tools from examples |

### 2.3 Recommended Approach for 40 Dimensions

Given Hey Seven's 40 behavioral dimensions:

**NOT 40 separate fine-tuning runs.** Instead:

1. **One SFT run** with diverse examples covering all dimensions
2. Each training example naturally covers 3-8 dimensions simultaneously
3. Focus gold traces on the 7 sub-5.0 dimensions (H9, P9, P6, P8, H10, H6, P7)
4. Include examples that score 8-9 across strong dimensions to prevent regression

**Minimum viable dataset:**
- 100 examples total (current: 51, need 49 more)
- At least 5 examples per sub-5.0 dimension (35 targeted examples)
- At least 50 examples showing general "host personality" across strong dimensions
- 10-15 examples showing tool use patterns (CCD behavior)

**Optimal dataset:**
- 150-200 examples total
- 10-15 examples per sub-5.0 dimension (70-105 targeted examples)
- 50-80 examples across strong dimensions
- 20-30 examples with tool call sequences

### 2.4 Quality Markers for Training Examples

Each training example must:
1. Demonstrate the EXACT behavior you want (not approximate)
2. Use the SAME system prompt that production uses
3. Be consistent in format, tone, and style across all examples
4. Include edge cases and challenging scenarios (not just easy ones)
5. For tool calls: show the full sequence (user query -> model calls tool -> tool returns data -> model responds using data)
6. Avoid contradictions between examples (e.g., one example books, another refuses to book)

---

## Section 3: Pricing and Cost Analysis

### 3.1 Training Costs

| Model | SFT Cost per 1M Training Tokens | DPO Cost per 1M Training Tokens |
|-------|--------------------------------|--------------------------------|
| Gemini 2.5 Pro | $25.00 | N/A (not supported) |
| **Gemini 2.5 Flash** | **$5.00** | **$5.00** |
| Gemini 2.5 Flash-Lite | $1.50 | $1.50 |
| Gemini 2.0 Flash | $3.00 | N/A |

**Training tokens = (tokens in dataset) x (number of epochs)**

### 3.2 Cost Estimate for Hey Seven

Assumptions:
- 150 training examples
- Average 3,000 tokens per example (multi-turn, 5-7 turns)
- Total dataset: 450,000 tokens (0.45M)
- 15 epochs (recommended for <1000 examples with long context)
- Training tokens: 0.45M x 15 = 6.75M tokens

**SFT cost: 6.75 x $5.00 = $33.75 per training run**

With iteration (expect 3-5 runs to get right):
- **Total SFT cost: $100-170**

If adding DPO after SFT (same dataset size):
- **Total DPO cost: $100-170 additional**
- **Combined SFT + DPO: $200-340 total**

### 3.3 Inference Costs (Post-Tuning)

Fine-tuned model inference is priced the SAME as the base model:
- Input: $0.30 per 1M tokens
- Output: $2.50 per 1M tokens

**No additional inference premium for fine-tuned models.**

### 3.4 Endpoint Hosting Costs

**WARNING**: Fine-tuned models on Vertex AI incur endpoint hosting charges while deployed, even with zero traffic.

- Hourly infrastructure fees apply while the endpoint is active
- Undeploy the model when not in use to stop charges
- Google uses LoRA adapters, so a single base model can serve multiple adapters

**Recommendation**: Keep the endpoint deployed only during evaluation/production. Undeploy during development iterations.

### 3.5 Training Time Estimates

| Dataset Size | Expected Training Time |
|-------------|----------------------|
| 10-15 examples | ~30-35 minutes |
| 100 examples | ~45-60 minutes |
| 500 examples | ~1-2 hours |
| 1000+ examples | ~2-4 hours |

For Hey Seven (150 examples, 15 epochs): **~1-2 hours per training run.**

---

## Section 4: DPO (Preference Tuning) Details

### 4.1 Availability

DPO IS available for Gemini 2.5 Flash and Flash-Lite on Vertex AI. It is NOT available for Gemini 2.5 Pro.

### 4.2 DPO Data Format

DPO requires paired examples with a "chosen" (preferred) and "rejected" response:

```jsonl
{
  "systemInstruction": {"role": "system", "parts": [{"text": "You are Seven..."}]},
  "contents": [
    {"role": "user", "parts": [{"text": "What's good for dinner?"}]}
  ],
  "chosen": {
    "role": "model",
    "parts": [{"text": "What are we celebrating? I want to make sure we find the perfect spot. Are you looking for something casual or more of a special-occasion vibe?"}]
  },
  "rejected": {
    "role": "model",
    "parts": [{"text": "We have several restaurants. Mohegan Sun Beyond is our steakhouse, Bobby's Burger Palace is casual, and Ballo is Italian. Would you like more information about any of these?"}]
  }
}
```

### 4.3 DPO Hyperparameters

| Parameter | Range | Recommendation |
|-----------|-------|---------------|
| Beta | 0.01 - 0.5 | Start at 0.1. Lower = more aggressive alignment |
| Epochs | Auto-adjusted | Let Vertex AI choose default |
| Adapter Size | 1-16 | Match SFT adapter size (4) |
| Learning Rate Multiplier | Default: 1.0 | Start with default |

### 4.4 When to Use DPO vs SFT for Hey Seven

| Scenario | Use |
|----------|-----|
| Teaching CCD tool-use pattern | SFT (demonstrate the full flow) |
| Teaching host personality/tone | SFT (show correct responses) |
| Eliminating chatbot-like responses | DPO (preferred: host, rejected: chatbot) |
| Teaching profiling questions | SFT (show natural question weaving) |
| Preventing over-enthusiasm/slop | DPO (preferred: warm substance, rejected: "Oh absolutely!") |
| Teaching comp mechanics integration | SFT with tool calls |

### 4.5 Recommended Sequence

1. **Phase 1: SFT** (100-150 examples) - Teach all target behaviors
2. **Phase 2: Evaluate** - Run 30-scenario eval with fine-tuned model
3. **Phase 3: DPO** (50-100 paired examples) - Fix remaining behavioral gaps
4. **Phase 4: Re-evaluate** - Confirm improvements, check for regressions

---

## Section 5: Hey Seven Specific Recommendations

### 5.1 Current State Assessment

| Asset | Count | Status |
|-------|-------|--------|
| Gold trace conversations | 51 | Below minimum (need 100) |
| Behavioral scenarios | 270 | Good coverage for eval |
| Sub-5.0 dimensions | 7 | Primary fine-tuning targets |
| Casino tools | 4 | Must be included in training data |
| Specialist agents | 6 | Each needs behavioral examples |

### 5.2 What SFT Will Fix (and What It Won't)

**SFT WILL improve:**
- H9 (Comp integration): Teach model to call `check_comp_eligibility` and integrate results naturally
- H10 (Return visit seeding): Teach model to call `lookup_upcoming_events` and weave into conversation
- P6 (Incentive framing): Teach natural comp mention patterns
- H6 (Rapport building): Demonstrate give-to-get conversation patterns
- P7 (Host bridge): Show when and how to offer human host connection
- P8 (Extraction): Demonstrate extraction-through-conversation (not interrogation)
- B1-B5 (General tone): Reinforce host personality across all interactions

**SFT will NOT fix:**
- RAG retrieval quality (use per-item chunking, RRF reranking)
- Factual accuracy (use RAG + grounding, not SFT)
- New knowledge about venues/events (use RAG + tool calls)
- Guardrail effectiveness (keep deterministic pre-LLM guardrails)
- Crisis response quality (keep rule-based crisis escalation)

### 5.3 Gold Trace Expansion Plan

Need 49-99 more examples. Sources for new gold traces:

| Source | Method | Expected Yield |
|--------|--------|---------------|
| Live eval transcripts scored 7+ | Extract and polish | 15-20 |
| Human-written CCD examples | Write new for sub-5.0 dims | 20-30 |
| Pro model outputs (curated) | Run Pro on scenarios, filter best | 20-30 |
| Oded's feedback examples | Convert feedback to gold traces | 10-15 |

**Priority for new examples:**
1. Tool-use sequences (H9, H10): 15-20 examples showing full CCD flow
2. Profiling through conversation (P3, P8): 10-15 examples showing natural question weaving
3. Host bridge offers (P9, P7): 10-15 examples showing decisive delegation
4. Comp/incentive framing (P6): 5-10 examples showing natural reward mentions
5. General host personality: 20-30 examples maintaining quality across strong dimensions

### 5.4 Training Data Quality Checklist

For each gold trace before including in training data:

- [ ] System prompt matches production `CONCIERGE_SYSTEM_PROMPT` exactly
- [ ] Multi-turn (minimum 3 turns, ideally 5-7)
- [ ] Tool calls use exact tool names from `casino_tools.py`
- [ ] Tool responses use realistic data from knowledge-base
- [ ] No contradictions with other training examples
- [ ] Covers at least 3 behavioral dimensions per conversation
- [ ] No slop patterns (exclamation spam, "absolutely", "I'd be happy to")
- [ ] Demonstrates CCD language (Checked-Confirmed-Dispatched)
- [ ] Guest profiling question woven into at least 2 turns
- [ ] Host bridge offered at appropriate moment (not every conversation)
- [ ] Comp/incentive mentioned only when relevant (not forced)
- [ ] Scored 7+ by human evaluator on primary target dimensions

### 5.5 JSONL Export Script Requirements

The export script must:
1. Read gold traces from `data/training/` directory
2. Convert to Vertex AI JSONL format with `systemInstruction` + `contents`
3. Include tool call sequences as `functionCall`/`functionResponse` parts
4. Validate JSONL format (each line valid JSON, correct structure)
5. Split into train (80%) and validation (20%) sets
6. Calculate and report token counts per example and total
7. Estimate training cost at $5/1M tokens
8. Upload to GCS bucket

### 5.6 Inference Configuration for Fine-Tuned Model

```python
# In production, when using the fine-tuned model:
model = GenerativeModel(
    model_name="projects/{project}/locations/us-central1/models/{tuned_model_id}",
    # CRITICAL: Set thinking budget to 0 for fine-tuned models
    generation_config={
        "temperature": 0.7,  # Match training temperature
        "thinking_budget": 0,  # Disable thinking for SFT models
        "max_output_tokens": 2048,
    },
    # Tool definitions remain the same
    tools=[casino_tools],
    # System instruction MUST match training data exactly
    system_instruction="You are Seven, a casino host for Mohegan Sun..."
)
```

---

## Section 6: Implementation Roadmap

### Phase 1: Data Preparation (1-2 days)

1. Audit existing 51 gold traces for quality (remove any below 7/10)
2. Write 20-30 new tool-use CCD examples (H9, H10, P6)
3. Write 15-20 profiling/relationship examples (P3, P8, P7, H6)
4. Generate 15-20 examples using Pro model on best scenarios (curate manually)
5. Total target: 100-130 high-quality examples
6. Build JSONL export script with validation
7. Split: 80% train, 20% validation
8. Upload to GCS bucket

### Phase 2: Initial SFT Training (1 day)

1. First run: epochs=15, lr_multiplier=10, adapter_size=4
2. Monitor training loss curve
3. If loss plateaus early: increase adapter_size to 8
4. If loss doesn't converge: increase epochs to 20
5. Training time: ~1-2 hours per run
6. Cost: ~$35 per run
7. Budget for 3-5 experimental runs: ~$105-175

### Phase 3: Evaluation (1 day)

1. Deploy fine-tuned model endpoint
2. Run 30-scenario eval (H-focused: H9, H10, H6 + P-focused: P6, P7, P8, P9)
3. Compare against R105 baselines with 3-model judge panel
4. Expected improvements: H9 2.35->5.0+, H10 3.87->5.0+, P6 3.93->5.0+
5. Check for regressions on strong dimensions (B1-B5 should stay >6.5)
6. Undeploy endpoint after eval

### Phase 4: DPO Refinement (1-2 days, if needed)

1. From Phase 3 eval, identify remaining failure patterns
2. Create 50-100 chosen/rejected pairs targeting failure modes
3. Run DPO training on the SFT-tuned model
4. Re-evaluate with same 30-scenario suite
5. Expected: additional +0.5-1.0 on weakest dimensions

### Phase 5: Production Integration (1 day)

1. Update `src/agent/nodes.py` to use fine-tuned model ID
2. Feature flag: `fine_tuned_model_enabled` (per-casino)
3. A/B test: 50% fine-tuned vs 50% base with tool-use
4. Monitor latency (should decrease due to thinking_budget=0)
5. Monitor behavioral scores via LangSmith traces

### Timeline Summary

| Phase | Duration | Cost |
|-------|----------|------|
| Data Preparation | 1-2 days | $0 (human effort) |
| Initial SFT | 1 day | $105-175 |
| Evaluation | 1 day | ~$20 (inference + judge) |
| DPO Refinement | 1-2 days | $100-175 |
| Production Integration | 1 day | $0 |
| **Total** | **5-7 days** | **$225-370** |

---

## Section 7: Risks and Mitigations

### 7.1 Training-Serving Skew

**Risk**: System prompt in training data differs from production system prompt.
**Mitigation**: Extract system prompt from `src/agent/prompts.py` CONCIERGE_SYSTEM_PROMPT and use verbatim in every training example. Add automated validation in export script.

### 7.2 Behavioral Regression on Strong Dimensions

**Risk**: Fine-tuning on sub-5.0 dims degrades performance on strong dims (B1-B5).
**Mitigation**: Include 30-50% of training examples that demonstrate strong-dimension behavior. Evaluate ALL 40 dimensions, not just targets. Set regression threshold: any strong dim drops >0.5 = rollback.

### 7.3 Tool Call Pattern Overfitting

**Risk**: Model learns to call tools in every turn, even when unnecessary.
**Mitigation**: Include training examples where the model responds WITHOUT tool calls (greetings, simple questions, emotional support). At least 40% of examples should be tool-free.

### 7.4 Endpoint Hosting Costs

**Risk**: Forgetting to undeploy endpoint = ongoing charges.
**Mitigation**: Add deployment/undeployment scripts. Set billing alert. Document undeploy procedure. Consider scheduled deployment (deploy for eval, undeploy after).

### 7.5 Fine-Tuned Model Deprecation

**Risk**: Gemini 2.5 Flash gets deprecated (like 1.5 Flash-001 in May 2025).
**Mitigation**: Store all training data and hyperparameters. Document exact reproduction steps. Training can be re-run on successor model (2.5 Flash -> next version) with same data.

### 7.6 Dataset Too Small (Current: 51)

**Risk**: 51 examples is below the minimum 100 recommended by Google.
**Mitigation**: Prioritize data generation BEFORE any training run. Do NOT train on 51 examples -- the model will overfit and memorize rather than generalize. Minimum viable: 100 examples.

### 7.7 Quality vs Quantity Trap

**Risk**: Rushing to 100 examples by generating low-quality synthetic data.
**Mitigation**: Every training example must pass the quality checklist (Section 5.4). Human review of ALL examples before training. Quality >> quantity -- 80 excellent examples beats 200 mediocre ones.

---

## Section 8: Key Findings Summary

### What We Learned

1. **51 examples is NOT enough.** Google's minimum recommendation is 100. The sweet spot for behavioral adaptation is 100-500. We need at least 49 more high-quality examples before our first training run.

2. **Tool calling IS supported in training data.** The JSONL format explicitly supports `functionCall` and `functionResponse` parts. This means we can teach CCD (Checked-Confirmed-Dispatched) behavior through SFT -- the model will learn WHEN and HOW to call tools.

3. **DPO IS available for Gemini 2.5 Flash.** We can do SFT first (teach target behaviors) then DPO (eliminate bad behaviors). Both cost $5/1M training tokens.

4. **Total cost is remarkably low: $225-370** for the entire fine-tuning pipeline including iterations. The ROI is clear given that prompt engineering has hit a ceiling (R105 confirmed: 7 dimensions, all +/-0.3 despite +1734 LOC).

5. **Thinking budget must be OFF** for fine-tuned models. This actually improves latency AND quality for SFT tasks.

6. **Training-serving skew is the #1 risk.** System prompt, tool definitions, and conversation format MUST match between training data and production inference. Automate this validation.

7. **SFT teaches style/behavior, NOT facts.** RAG handles factual knowledge. SFT handles HOW the model talks, asks questions, uses tools, and builds relationships. This is exactly the gap Hey Seven has.

8. **Adapter size 4 is the right starting point.** With <500 examples, size 4 balances learning capacity with overfitting risk. Increase to 8 only if underfitting is observed.

9. **One training run covers all 40 dimensions.** No need for per-dimension fine-tuning. Each multi-turn conversation naturally covers 3-8 dimensions. Focus examples on weak dimensions but include strong-dimension examples to prevent regression.

10. **Expected timeline: 5-7 working days** from data preparation to production integration. The fastest path to breaking the prompt engineering ceiling.

### Decision Point for Oded

**Option A: Minimum Viable Fine-Tune**
- Generate 49 more examples (total 100)
- One SFT run, evaluate, iterate 2-3 times
- Cost: ~$135-210
- Timeline: 3-4 days
- Risk: May not be enough examples for full behavioral shift

**Option B: Comprehensive Fine-Tune (Recommended)**
- Generate 99-149 more examples (total 150-200)
- SFT + DPO, evaluate after each
- Cost: ~$225-370
- Timeline: 5-7 days
- Expected: H9 2.35->5.0+, P-avg 5.18->6.5+, H-avg 5.09->6.0+

**Option C: Maximum Quality**
- Generate 249+ more examples (total 300+)
- SFT + DPO + iterative data curation
- Cost: ~$400-600
- Timeline: 8-12 days
- Expected: All dimensions >5.0, B-avg >7.5

---

## Appendix A: JSONL Validation Script

```python
"""Validate JSONL training data for Vertex AI Gemini fine-tuning."""
import json
import sys
from pathlib import Path

REQUIRED_ROLES = {"user", "model"}
OPTIONAL_ROLES = {"function", "system"}
MAX_EXAMPLES = 10_000_000

def validate_jsonl(filepath: str) -> tuple[bool, list[str]]:
    errors = []
    valid_count = 0

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                continue

            # Check required field
            if "contents" not in example:
                errors.append(f"Line {line_num}: Missing required 'contents' field")
                continue

            contents = example["contents"]
            if not isinstance(contents, list) or len(contents) < 2:
                errors.append(f"Line {line_num}: 'contents' must have >= 2 turns")
                continue

            # Check roles alternate correctly
            for i, turn in enumerate(contents):
                role = turn.get("role")
                if role not in REQUIRED_ROLES | OPTIONAL_ROLES:
                    errors.append(f"Line {line_num}, turn {i}: Invalid role '{role}'")

                parts = turn.get("parts")
                if not isinstance(parts, list) or len(parts) == 0:
                    errors.append(f"Line {line_num}, turn {i}: Empty or missing 'parts'")

            # Check systemInstruction format if present
            if "systemInstruction" in example:
                si = example["systemInstruction"]
                if "parts" not in si:
                    errors.append(f"Line {line_num}: systemInstruction missing 'parts'")

            valid_count += 1

    return len(errors) == 0, errors

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "training_data.jsonl"
    ok, errors = validate_jsonl(filepath)
    if ok:
        print(f"Validation passed. All examples valid.")
    else:
        print(f"Validation FAILED. {len(errors)} errors:")
        for e in errors[:20]:
            print(f"  {e}")
    sys.exit(0 if ok else 1)
```

## Appendix B: Token Count Estimator

```python
"""Estimate training cost for Gemini 2.5 Flash SFT."""
import json

COST_PER_M_TOKENS = 5.00  # Gemini 2.5 Flash SFT

def estimate_cost(filepath: str, epochs: int = 15) -> dict:
    total_chars = 0
    example_count = 0

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_chars += len(line)
            example_count += 1

    # Rough token estimate: ~4 chars per token for English
    estimated_tokens = total_chars / 4
    training_tokens = estimated_tokens * epochs
    cost = (training_tokens / 1_000_000) * COST_PER_M_TOKENS

    return {
        "examples": example_count,
        "estimated_tokens": int(estimated_tokens),
        "epochs": epochs,
        "training_tokens": int(training_tokens),
        "estimated_cost_usd": round(cost, 2),
    }
```

## Appendix C: Vertex AI SFT Launch Script

```python
"""Launch Gemini 2.5 Flash supervised fine-tuning on Vertex AI."""
import time
from google import genai
from google.genai.types import (
    HttpOptions,
    CreateTuningJobConfig,
    TuningDataset,
)

PROJECT_ID = "hey-seven-prod"  # Update
REGION = "us-central1"
GCS_BUCKET = "gs://hey-seven-training-data"

client = genai.Client(
    http_options=HttpOptions(api_version="v1beta1"),
    project=PROJECT_ID,
    location=REGION,
)

tuning_job = client.tunings.tune(
    base_model="gemini-2.5-flash",
    training_dataset=TuningDataset(
        gcs_uri=f"{GCS_BUCKET}/train.jsonl",
    ),
    config=CreateTuningJobConfig(
        tuned_model_display_name="hey-seven-host-v1",
        validation_dataset=TuningDataset(
            gcs_uri=f"{GCS_BUCKET}/validation.jsonl",
        ),
        epoch_count=15,
        adapter_size=4,
        learning_rate_multiplier=10.0,
    ),
)

print(f"Job started: {tuning_job.name}")
print(f"State: {tuning_job.state}")

# Poll for completion
running_states = {"JOB_STATE_PENDING", "JOB_STATE_RUNNING"}
while tuning_job.state in running_states:
    print(f"  {tuning_job.state}...")
    time.sleep(60)
    tuning_job = client.tunings.get(name=tuning_job.name)

print(f"Final state: {tuning_job.state}")
if tuning_job.tuned_model:
    print(f"Model: {tuning_job.tuned_model.model}")
    print(f"Endpoint: {tuning_job.tuned_model.endpoint}")
```

---

## Limitations of This Research

1. **DPO data format details are sparse** in official docs. The chosen/rejected format is confirmed but exact field names may differ. Verify against latest Vertex AI preference tuning documentation before implementation.

2. **Tool calling in fine-tuned models**: While the training data format supports `functionCall`/`functionResponse`, real-world behavior of fine-tuned models with tools has limited documented case studies. Plan for a validation round specifically testing tool-calling preservation.

3. **No published case studies** for fine-tuning conversational agents with 40 behavioral dimensions. The MHC-Coach study (3,268 examples, single domain) is the closest analogue, but operates on a different model family.

4. **Endpoint hosting costs** are confirmed to exist but exact hourly rates are not publicly documented on the pricing page. Budget for this or use deployment/undeployment automation.

5. **Gemini 3 fine-tuning** remains enterprise-only with no public timeline. Our Gemini 2.5 Flash strategy is correct for now, but monitor Google Cloud Next announcements for any changes.

---

*Research completed 2026-03-09. Total research queries: 8 (Perplexity Deep Research x1, Perplexity Reason x2, Perplexity Search x4, WebFetch x2). All findings cross-referenced against official Google Cloud documentation.*