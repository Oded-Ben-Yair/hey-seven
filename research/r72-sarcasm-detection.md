# R72 Research: Sarcasm Detection for Production AI Casino Host

**Date**: 2026-02-27
**Scope**: State-of-the-art sarcasm detection approaches for Hey Seven's AI casino host agent
**Sources**: Perplexity Deep Research (4 queries), academic benchmarks (iSarcasm, SemEval-2018 Task 3, MMSD2.0), production case studies (Intercom Fin, Zendesk AI, Ada, PolyAI, Kore.ai)

---

## Executive Summary

The critical finding from this research is that **production chatbot companies do NOT explicitly detect sarcasm**. Instead, they use "sarcasm-resilient" response strategies that work regardless of whether the user is being sarcastic. This is the industry consensus, supported by both academic evidence and production case studies.

For Hey Seven specifically, the recommended approach is a **two-layer hybrid**: (1) lightweight sarcasm signal detection that adjusts response tone, combined with (2) empathetic-by-default response design that works correctly even when sarcasm detection fails. This is fundamentally different from a binary sarcasm classifier.

**Key numbers**:
- Human inter-annotator agreement on sarcasm: ~80-85% (natural ceiling)
- Best regex/keyword F1: 0.71-0.75
- Best fine-tuned DistilBERT F1: 0.83-0.87 (in-domain)
- Best LLM zero-shot F1: 0.60-0.68
- Best LLM few-shot F1: 0.62-0.70
- Best ensemble F1: 0.74-0.85 (domain-dependent)
- Production systems: None use explicit sarcasm classifiers; all use sentiment + empathy fallback

---

## Approach Comparison Table

| Approach | Latency | Accuracy (F1) | Cost/Call | Handles Deadpan | Handles Understatement | Production-Ready |
|----------|---------|---------------|-----------|-----------------|----------------------|-----------------|
| Regex/keyword patterns | <1ms | 0.71-0.75 | $0 | No | No | Yes (as filter) |
| Embedding incongruity | 10-20ms | 0.75-0.80 | ~$0.001 | Partially | No | Yes (as signal) |
| Fine-tuned DistilBERT | 30-50ms | 0.83-0.87 | ~$0.002 | Yes | Partially | Yes (requires training) |
| LLM micro-call (Flash) | 50-150ms | 0.60-0.70 | ~$0.01 | Yes | Yes | Yes (no training) |
| WM-SAR framework | 300-500ms | 0.85-0.88 | ~$0.08 | Yes | Yes | No (too slow) |
| Ensemble (regex+embed+LLM) | 30-80ms avg | 0.82-0.85 | ~$0.005 | Yes | Partially | Yes |
| **Empathetic-by-default** | **0ms** | **N/A** | **$0** | **Yes** | **Yes** | **Yes** |

---

## 1. Regex/Keyword Approach: What It Catches and What It Misses

### What Hey Seven currently has (R71)
Hey Seven's current implementation uses 5 sarcasm keyword patterns in `guardrails.py`:
- "I suppose", "could be worse", ironic "Very helpful", "whatever", "if you say so"

### Accuracy ceiling
Published research on the SARC dataset shows regex/keyword approaches hit an F1 ceiling of 0.71-0.75 on balanced data. A basic linear SVM with TF-IDF features achieves 72.3% accuracy. Adding conversational context features improves this to only 77.3%.

### What regex misses (critical for casino context)
1. **Deadpan sarcasm**: "What a surprise. The pool is closed again." (No markers, flat delivery)
2. **Understatement**: "That was a minor inconvenience" (after a major issue)
3. **Backhanded compliments**: "You're much more helpful than the last chatbot" (ambiguous intent)
4. **Contextual sarcasm**: "Sure, I'd love that" (sincere after good news, sarcastic after bad)
5. **Cultural sarcasm**: British understatement patterns common among international casino guests
6. **Implicit incongruity**: "Wonderful timing" (said when something bad just happened)

### When regex IS sufficient
- Explicit markers: "/s", sarcastic emoji (eye-roll, smirk)
- Domain-specific patterns: "Oh great, another chatbot", "Very helpful, thanks" in complaint contexts
- Known trigger phrases after repeated failures: "Sure, whatever you say"

**Verdict**: Keep regex as Stage 1 filter for obvious cases. Do not rely on it for subtle sarcasm.

---

## 2. Lightweight LLM Micro-Classification (Gemini Flash)

### Prompt engineering for sarcasm
Research identifies several effective prompting strategies:

**Few-shot (BEST for fast models)**: Include 3-5 examples of sarcastic and non-sarcastic statements with context. For GPT-4o-mini, few-shot outperformed chain-of-thought (CoT actually degraded performance).

**Chain of Contradiction (CoC)**: Decompose into:
1. Identify surface sentiment from keywords
2. Determine contextually expected sentiment
3. Detect contradiction between them
4. Classify if contradiction indicates sarcasm

CoC outperformed standard CoT on GPT-4o by a significant margin on certain benchmarks.

**SarcasmCue framework** (best academic results):
- Graph of Cues (GoC): Multiple cue categories (linguistic, contextual, emotional) in parallel
- Non-sequential methods outperformed sequential, indicating sarcasm detection is NOT step-by-step reasoning

### Practical accuracy
- Phi-4 zero-shot: 60.36% accuracy
- Llama 3.1 zero-shot: 49.53%
- Best LLM few-shot: 62.29%
- Fine-tuned DistilBERT on same task: 83.87% (beats LLMs by 20+ points)

### For Hey Seven
A Gemini Flash micro-call with few-shot prompting could work within ~100ms budget, but accuracy (60-70%) is below the bar for reliable classification. Better used as one signal in an ensemble, not as sole detector.

**Recommended prompt pattern** (if used):
```
You are analyzing customer sentiment in a casino concierge conversation.
Given the conversation context and the latest message, classify the tone.

Context: {last_2_messages}
Message: {current_message}

Is the customer being sarcastic? Respond with JSON:
{"sarcasm_probability": 0.0-1.0, "detected_type": "none|marked|deadpan|understatement|backhanded"}
```

---

## 3. Embedding-Based Semantic Incongruity Detection

### How it works
Compute embeddings for the user's statement and the conversational context. If cosine distance exceeds a threshold, flag potential sarcasm (the statement semantically contradicts what would be expected).

### Published results
- Augmenting features with Word2Vec embeddings improved F-score by ~4% consistently
- Thresholds: >0.7 cosine distance = likely sarcasm, <0.3 = likely sincere, 0.3-0.7 = ambiguous
- Latency: 10-20ms (embedding computation + similarity)

### Limitations
- Captures semantic relationship but NOT pragmatic intention
- Requires learned thresholds that risk overfitting to training data
- Fails on sarcasm that is semantically consistent with context (e.g., understatement)

### For Hey Seven
We already compute embeddings for RAG retrieval. Adding an incongruity signal by comparing the user message embedding against the expected response context is nearly free (10ms additional). This provides a useful "something doesn't match" signal without requiring a separate model.

---

## 4. WM-SAR Framework (World Model Inspired Sarcasm Reasoning)

### Architecture
4 parallel LLM agents:
1. **Literal Meaning Evaluator**: Surface polarity + rationale
2. **Context Model**: Reconstructs background situation (social relations, scene, events)
3. **Normative Expectation Evaluator**: What sentiment SHOULD be expected given context
4. **Intention Reasoner**: Theory-of-Mind inference for speaker intent

Outputs feed into logistic regression: P(Sarcasm) = sigmoid(w1*D + w2*I + bias)

### Results
Near state-of-the-art on benchmarks. Key advantage: interpretability (each prediction has explicit rationale). Key disadvantage: 300-500ms latency (4 parallel LLM calls).

### For Hey Seven
**Not viable for real-time chat** (exceeds 200ms budget by 2x). However, the decomposition pattern is intellectually valuable: it confirms that sarcasm detection is best approached as multi-signal fusion, not single-classifier. We can apply the CONCEPT (literal vs expected mismatch) without the ARCHITECTURE (4 parallel LLMs).

---

## 5. What Production Companies Actually Do (The Critical Finding)

### The universal pattern: None detect sarcasm explicitly

| Company | Approach | Details |
|---------|----------|---------|
| **Intercom Fin** | Sentiment + escalation routing | 96% resolution rate. Uses "Custom Answers" for known difficult patterns. Escalates on negative sentiment, doesn't classify sarcasm. |
| **Zendesk AI** | Broad sentiment framework | Categorizes tone as "positive", "negative", "neutral". Routes to human on persistent negative. No sarcasm-specific model. |
| **Ada** | Real-time response adaptation | Uses sentiment signals to adjust tone. "Sarcasm resilient" design. |
| **PolyAI** | Conversational flow management | "Raven" reasoning model maintains contextual memory. Detects emotional cues but doesn't classify sarcasm type. |
| **Kore.ai** | Standard response templates | Flexible templates that work across sentiment types. No sarcasm detection. |

### The "empathetic by default" pattern
All production systems converge on the same design:
1. Detect negative/ambiguous sentiment (NOT sarcasm specifically)
2. Respond empathetically to ANY ambiguous tone
3. Offer escalation when sentiment remains negative
4. Use tone ladders (formal -> warm -> empathetic -> urgent) based on conversation trajectory

### Why this works better than detection
- **No false positives**: Empathetic responses to genuine statements are never harmful
- **No false negatives**: Even missed sarcasm gets empathy, which is the correct response anyway
- **Lower cost**: No separate classification model needed
- **Higher CSAT**: Zendesk data shows 20% CSAT improvement from sentiment-tuned responses
- **Handles edge cases**: Backhanded compliments, cultural differences, and deadpan all get appropriate treatment

### The hospitality-specific insight
In hospitality contexts (hotels, casinos, restaurants), sarcasm almost always masks genuine frustration. The appropriate response is the same whether the sarcasm is detected or not:
1. Acknowledge the feeling (not the literal words)
2. Offer to help
3. Don't mirror the sarcasm (stay professional, never playful in response to frustration)

Casino-specific examples:
- "Oh great, another chatbot" -> "I understand — let me show you I can actually help. What do you need?"
- "Very helpful, thanks" (after bad answer) -> "I hear you. Let me take another approach to help with that."
- "What a surprise, the pool is closed" -> "I know that's frustrating. Here are the other amenities available right now..."

---

## 6. The Ensemble Approach: Architecture for Hey Seven

### Recommended three-stage pipeline

```
Stage 1: Regex filter (<1ms)
├── Explicit markers (/s, sarcastic emoji) -> HIGH_CONFIDENCE_SARCASM
├── Known sincere patterns ("I genuinely", "thank you so much") -> LOW_CONFIDENCE_SARCASM
└── Everything else -> Stage 2

Stage 2: Sentiment incongruity signal (5-10ms)
├── VADER sentiment + conversation trajectory
├── If positive words + negative context -> POSSIBLE_SARCASM
├── If sentiment shift (positive after frustration) -> POSSIBLE_SARCASM
└── Use as SIGNAL, not classifier

Stage 3: Response strategy selection (0ms, rule-based)
├── HIGH_CONFIDENCE_SARCASM -> Empathetic acknowledgment, no mirror
├── POSSIBLE_SARCASM -> Empathetic by default, offer help
├── NEUTRAL/UNCLEAR -> Standard response
└── FRUSTRATED (from VADER trajectory) -> HEART framework
```

### Why NOT add LLM micro-call
- 60-70% accuracy is worse than empathetic-by-default (which is "correct" 100% of the time)
- Adds 100ms latency for marginal benefit
- False positives (flagging sincere messages as sarcastic) cause worse responses than just being empathetic
- Cost scales linearly with messages

---

## 7. Benchmark Context: What "Good" Looks Like

### Published benchmarks

| Dataset | Task | Best System | F1 Score |
|---------|------|-------------|----------|
| SemEval-2018 Task 3 | Binary irony (tweets) | Top system | 0.71 |
| SemEval-2018 Task 3 | Fine-grained irony | Top system | 0.51 |
| iSarcasm | Author-intended sarcasm | Best model | 0.36 (!) |
| SARC (Reddit) | Sarcasm detection | Ensemble | 0.67-0.74 |
| Code-mixed (Hinglish) | Sarcasm detection | Fine-tuned DistilBERT | 0.84 |
| MMSD2.0 | Multimodal sarcasm | Best model | 0.92 |

### The iSarcasm reality check
iSarcasm is specifically designed around AUTHOR-INTENDED sarcasm (not annotator-perceived). Models achieving 0.87 F1 on other datasets dropped to 0.36 F1 on iSarcasm. This proves that most "sarcasm detection" benchmarks are testing pattern matching, not true understanding. Real-world sarcasm detection is much harder than benchmarks suggest.

### Human ceiling
Inter-annotator agreement on sarcasm labels is ~80-85%. This means even a perfect model cannot exceed 85% F1 without systematic bias. Any system claiming >85% accuracy on general sarcasm is likely overfitting to dataset artifacts.

---

## Recommended Approach for Hey Seven

### Architecture: "Sarcasm-Resilient Response Design" + Signal Detection

**Do NOT build a sarcasm classifier.** Instead:

1. **Expand sarcasm pattern detection** (existing regex in guardrails.py) to cover 15-20 common casino-context patterns. These are not for classification -- they are tone signals that adjust response warmth.

2. **Add sentiment trajectory analysis** (already partially implemented in `_detect_conversation_dynamics`). Track whether sentiment shifted from negative/frustrated to suddenly positive (a strong sarcasm indicator).

3. **Design all responses to be sarcasm-resilient**: Any response that works for a sincere message MUST also work for a sarcastic version. Test every response template against the sarcastic reading.

4. **Use empathetic acknowledgment for all ambiguous negative sentiment**. The response to detected sarcasm and to genuine frustration is identical: acknowledge the feeling, offer to help, don't mirror.

5. **Never mirror sarcasm**. Production research unanimously says bots should stay professional. Playful responses to sarcasm are high-risk (misread sincerity -> looks mocking).

### What this looks like in practice

```python
# In _base.py or response formatting:
if sarcasm_signal or sentiment in ("frustrated", "negative"):
    # Same response for both -- empathetic acknowledgment
    tone_guidance = (
        "The guest may be frustrated or sarcastic. Either way: "
        "acknowledge their feeling, offer concrete help, "
        "keep it brief and professional. No exclamation marks, "
        "no 'I'd be happy to help!' — just direct assistance."
    )
```

### Why this is the right call for Hey Seven

1. **Casino guests who are sarcastic are almost always frustrated.** The correct response to both is empathy.
2. **False positive sarcasm detection is worse than no detection.** Treating a sincere "Thanks for your help!" as sarcastic and responding with "I'm sorry you feel that way" is a terrible experience.
3. **The 200ms latency budget is better spent on RAG quality** than sarcasm classification.
4. **Our behavioral score (B1 Sarcasm: 6.8/10)** can be improved more by better response design than by better detection.

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
- [ ] Expand sarcasm patterns in guardrails.py from 5 to 15-20 casino-specific patterns
- [ ] Add sentiment trajectory signal: detect positive-after-negative shift
- [ ] Add "sarcasm-resilient" tone guidance to system prompt for ambiguous cases
- [ ] Test all specialist agent response templates against sarcastic readings

### Phase 2: Structural (3-5 days)
- [ ] Add `tone_signal` field to AgentState (sarcasm_detected, frustration_detected, ambiguous)
- [ ] Wire tone signal into specialist prompt injection (suppress exclamation marks, enthusiasm)
- [ ] Create sarcasm-resilient response templates for top 10 scenarios
- [ ] Add integration tests: same query delivered sarcastically vs sincerely should get equally helpful responses

### Phase 3: Evaluation (2-3 days)
- [ ] Add 10 more B1 sarcasm scenarios to behavioral evaluation (backhanded compliments, deadpan, understatement)
- [ ] Run B1 evaluation with live LLM to measure improvement
- [ ] A/B test: empathetic-by-default vs no-sarcasm-handling baseline
- [ ] Document approach in ADR-024

### What NOT to build
- No DistilBERT fine-tuning (overkill for our use case)
- No LLM micro-call for classification (too slow, too inaccurate)
- No WM-SAR implementation (way too slow)
- No embedding incongruity model (marginal benefit over sentiment trajectory)

---

## Key Citations

1. SemEval-2018 Task 3 benchmark: https://aclanthology.org/S18-1005.pdf (F1=0.71 binary irony)
2. iSarcasm dataset: https://aclanthology.org/2020.acl-main.118.pdf (0.36 F1 on author-intended)
3. SarcasmCue prompting: https://arxiv.org/html/2407.12725v1 (CoC outperforms CoT)
4. WM-SAR framework: https://arxiv.org/abs/2512.24329 (multi-agent structured reasoning)
5. LLM sarcasm detection: https://arxiv.org/abs/2601.08302 (few-shot > CoT for small models)
6. Code-mixed DistilBERT: https://arxiv.org/html/2602.21933v1 (83.87% vs LLM 62.29%)
7. Embedding augmentation: https://aclanthology.org/D16-1104.pdf (+4% F-score)
8. Ensemble approaches: https://aclanthology.org/2020.figlang-1.36/ (ensemble learning for sarcasm)
9. Production latency budgets: https://hamming.ai/resources/debugging-voice-agents (500ms voice, 200ms text)
10. PolyAI Raven model: https://skywork.ai/skypage/en/PolyAI-Deep-Dive (conversational flow management)
11. Intercom Fin: https://www.intercom.com/help/en/articles/7120684-fin-ai-agent-explained (escalation patterns)
12. Zendesk sentiment: https://www.zendesk.com/apps/support/1058555/ai-sentiment-analysis/ (broad framework)
13. Production systems comparison: https://cosupport.ai/articles/leading-ai-agents-2026-real-world-comparison
14. Multimodal sarcasm: https://www.emergentmind.com/topics/multimodal-sarcasm-detection-msd (0.92 F1 with images)
15. DistilBERT performance: https://pmc.ncbi.nlm.nih.gov/articles/PMC11224257/ (1.74x speedup, competitive accuracy)
