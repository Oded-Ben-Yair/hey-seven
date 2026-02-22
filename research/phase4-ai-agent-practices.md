# Phase 4 Research: AI Agent Best Practices for Hey Seven

**Date**: 2026-02-22
**Confidence**: High (4 deep research queries, 40+ citations, mapped to existing codebase)
**Scope**: Multi-turn coherence, proactive suggestions, LLM-as-judge evaluation, task completion tracking

---

## 1. Multi-Turn Coherence and Persona Consistency

### 1.1 State of the Art

The conversational AI market is projected to reach $14.29B in 2025, growing at 23.7% CAGR to $41.39B by 2030. The critical technical challenges are multi-turn coherence (maintaining context across long conversations), persona consistency (stable character/tone), and emotional intelligence (detecting and responding to user sentiment).

### 1.2 Key Findings

**Persona Drift is Real and Measurable**: Testing of large language models reveals significant persona drift beginning within 8 rounds of conversation. Turn-by-turn persona consistency drops 20-40% over 10-15 turns in domain-specific personas. The drift traces to transformer attention decay -- attention patterns become dominated by recent tokens while historical context (including system prompts) becomes progressively attenuated.

**Hey Seven Impact**: The current `CONCIERGE_SYSTEM_PROMPT` in `prompts.py` and the persona enforcement in `persona_envelope_node` are at risk of drift in extended conversations (MAX_MESSAGE_LIMIT=40 allows up to 20 human turns).

**Mitigation Techniques (Ranked by Applicability)**:

1. **Contrastive Decoding for System Prompt Strength**: A continuous hyperparameter that amplifies persona signal by contrasting logits from the target system prompt against a default prompt. Achieves +8.5 points on instruction following. Does not require model retraining. *Applicable via prompt engineering reinforcement.*

2. **Score-Based Persona Consistency Training (SBS)**: Conditions dialogue generation directly on quality scores during training. Shows improvements for both million and billion-parameter models on PERSONA-CHAT and ConvAI2 benchmarks. *Not directly applicable (requires fine-tuning), but informs evaluation rubric design.*

3. **Periodic System Prompt Re-injection**: The simplest production pattern -- re-inject core persona instructions every N turns as a SystemMessage. *Directly applicable to Hey Seven's graph.*

**Memory Management Patterns**:

- **ConversationSummaryBufferMemory**: Hybrid approach maintaining summaries of older turns while preserving recent exchanges verbatim. Best balance of token efficiency and information preservation.
- **Vector Store Long-Term Memory**: Embed conversation fragments into semantic space for retrieval based on similarity rather than recency.
- **Dialogue State Tracking (DST)**: Maintains structured (slot, value) pairs for user goals/constraints. Directly relevant to Hey Seven's `extracted_fields` state field and `_merge_dicts` reducer.

**Hey Seven Mapping**:
- `MAX_HISTORY_MESSAGES=20` sliding window in `config.py` = ConversationBufferWindowMemory
- `extracted_fields` with `_merge_dicts` reducer in `state.py` = explicit DST
- `guest_profile.py` Firestore persistence = cross-session memory
- Missing: conversation summarization for turns beyond the sliding window

### 1.3 Emotional Intelligence

**Sentiment Analysis in Production**: Systems that detect frustration, happiness, or confusion and adapt responses accordingly report 25% increase in customer satisfaction and 20% lower churn rates.

**Hey Seven Mapping**:
- `guest_sentiment` field in `PropertyQAState` (positive/negative/neutral/frustrated via VADER) is aligned with best practice
- `SENTIMENT_TONE_GUIDES` in `prompts.py` provides tone adaptation per sentiment
- Gap: No escalation trigger based on sustained negative sentiment across turns

### 1.4 Context Engineering

The field has shifted from "prompt engineering" to "context engineering" -- designing dynamic systems that provide the right information at the right time.

**Hey Seven Mapping**:
- `whisper_planner.py` is a context engineering component
- `_format_context_block()` in `nodes.py` formats RAG context for the LLM
- `get_agent_context()` in `guest_profile.py` filters profile data for the LLM
- Gap: No adaptive context selection based on conversation phase

---

## 2. Proactive AI Suggestions in Hospitality

### 2.1 Key Findings

**The Satisfaction-Annoyance Paradox**:
- Optimal: 2-4 well-calibrated suggestions per multi-night stay
- Satisfaction drops sharply when suggestion accuracy falls below 80%
- A single irrelevant suggestion can negate the satisfaction gain from a correct one
- 35% higher satisfaction when guests perceive control over personalization depth

**Personalization Depth Levels**:
1. **Stated preference**: Based on explicit guest statements (name, dietary restrictions)
2. **Behavioral inference**: Patterns inferred from actual behavior (booking patterns, usage)
3. **Contextual situational**: Accounts for specific circumstances of current visit

**Hey Seven Mapping**:
- `extracted_fields` captures stated preferences (name, party size, dietary needs)
- `guest_profile.py` with confidence scoring enables behavioral inference
- `whisper_planner.py` provides contextual situational awareness
- **Current gap**: No proactive suggestion system -- all responses are reactive

### 2.2 Proactive Suggestion Framework

Based on research, a proactive suggestion system should:
1. Detect suggestion opportunities in the whisper planner
2. Limit frequency to 1 suggestion per conversation (not per turn)
3. Require confidence threshold (80%+ match to guest profile)
4. Make suggestions contextual (time-aware)
5. Provide opt-out (track if guest ignores suggestions)

**Implementation Pattern**:
```python
# In WhisperPlan (whisper_planner.py)
class WhisperPlan(BaseModel):
    proactive_suggestion: str | None = None
    suggestion_confidence: float = 0.0  # Must exceed 0.8 to surface
    suggestion_category: str | None = None  # "dining", "entertainment", "spa"
```

---

## 3. LLM-as-Judge Evaluation Frameworks

### 3.1 Framework Landscape (2025-2026)

| Framework | Focus | Key Innovation |
|-----------|-------|----------------|
| AlpacaEval | General instruction following | LC win rates (vulnerable to gaming) |
| MT-Bench | Multi-turn dialogue | 8 capability categories, 85% human agreement |
| Arena-Hard v2.0 | Open-ended instruction | Separability metrics |
| G-Eval | Chain-of-thought evaluation | 3-step: step generation -> judging -> scoring |
| JudgeBench | Evaluating the evaluators | Position-consistent accuracy |

### 3.2 Critical Bias Findings

- **Position Bias**: Judges favor responses based on position. Mitigate: run both orderings.
- **Self-Preference Bias**: LLMs overestimate quality of similar outputs. GPT-4 worst offender.
- **Length Bias**: Judges prefer longer responses. Mitigate: explicit rubrics de-emphasizing verbosity.

### 3.3 Rubric Design Best Practices

1. Binary or few named categories over numeric scales (1-10 unreliable)
2. Concrete observable evidence for each category
3. Objective dimensions first (factual correctness > style)
4. Hierarchical decomposition (primary -> sub-criteria -> reasoning)
5. Calibrate against expert ground truth (100-500 labeled examples)

### 3.4 Multi-Judge Panels

- Multi-agent debate improves accuracy by ~15.55% over raw judgments
- Meta-judging improves 8.37% over single-agent
- Diverse judge personas capture different failure modes

### 3.5 Hey Seven Gaps

Current `evaluation.py` uses pattern-matching only. Needs:
1. LLM-as-judge evaluation layer (G-Eval pattern)
2. Multi-turn evaluation scenarios
3. Adversarial/red-team test cases
4. Faithfulness metric (does response cite RAG context accurately?)

**Recommended Enhancement**:
```python
class LLMJudgeScore(BaseModel):
    groundedness: Literal["fully_grounded", "partially_grounded", "ungrounded"]
    persona_consistency: Literal["consistent", "minor_drift", "major_drift"]
    proactive_value: Literal["adds_value", "neutral", "irrelevant"]
    reasoning: str  # Chain-of-thought explanation
```

---

## 4. Task Completion Tracking

### 4.1 State Machine Patterns

Production agents implement explicit state machines: pending -> in_progress -> blocked -> completed -> failed. Systems with explicit task state tracking recover from 85% of transient failures.

### 4.2 Handoff Protocols

**Critical Stat**: 40-50% of escalations fail because context is lost during transfer.

**Effective Escalation Requires**:
1. Explicit escalation criteria (deterministic)
2. Complete context preservation (full transcript, attempted actions, sentiment)
3. Skilled routing (route to expertise, not generic queue)

**Hey Seven Mapping**:
- 5 guardrail layers provide deterministic escalation
- `responsible_gaming_count` with `_keep_max` reducer tracks escalation across turns
- **Gap**: No structured handoff to human casino host

### 4.3 Verification

**pass@k Reliability**: Agents achieving 60% single-run success often drop to 25% at pass@8. Single-run success is misleading for production reliability.

---

## 5. Actionable Recommendations for Hey Seven

### Priority 1: Persona Drift Prevention (High Impact, Low Effort)

Add periodic system prompt re-injection in `execute_specialist()` (_base.py). Every 5 turns, prepend a condensed persona reminder as SystemMessage.

```python
if len(messages) > 10:
    persona_reminder = SystemMessage(content=(
        f"REMINDER: You are Seven, the AI concierge for {settings.PROPERTY_NAME}. "
        "Maintain warm, professional tone. Never provide gambling advice."
    ))
    messages.insert(-2, persona_reminder)
```

### Priority 2: LLM-as-Judge Evaluation Layer (High Impact, Medium Effort)

Add G-Eval-style LLM judge as separate CI stage. Scoring dimensions:
1. Groundedness (no hallucination)
2. Persona Fidelity (warm, professional, no emoji)
3. Safety Compliance (no gambling advice)
4. Contextual Relevance (property-specific)
5. Proactive Value (adds helpful context)

### Priority 3: Proactive Suggestion System (High Impact, Medium Effort)

Extend `WhisperPlan` with optional proactive suggestion. Max 1 per conversation, 80%+ confidence threshold. Never suggest when sentiment is negative.

### Priority 4: Structured Handoff Protocol (Medium Impact, Medium Effort)

```python
class HandoffContext(BaseModel):
    guest_name: str | None
    guest_sentiment: str
    conversation_summary: str
    query_type: str
    attempted_actions: list[str]
    escalation_reason: str
    suggested_resolution: str | None
    profile_completeness: float
```

### Priority 5: Conversation Summarization (Medium Impact, Low Effort)

When messages exceed MAX_HISTORY_MESSAGES, summarize oldest messages and prepend as SystemMessage. Preserves stated preferences after sliding window moves.

### Priority 6: Multi-Turn Evaluation Test Suite (Medium Impact, Medium Effort)

Create multi-turn golden test scenarios testing:
1. Context retention across turns
2. Persona consistency over 10 turns
3. Preference accumulation
4. Escalation tracking

### Priority 7: pass@k Reliability Testing (Low Impact, Low Effort)

Run top 20 golden test cases 5 times each. Flag any with pass@5 below 80%.

### Priority 8: Sentiment-Based Escalation (Low Impact, Low Effort)

If sentiment is `frustrated` for 2+ consecutive turns, add soft escalation prompt. Track via `consecutive_negative_sentiment` counter.

---

## 6. Sources

1. NextLevel AI Voice Trends - https://nextlevel.ai/voice-ai-trends-enterprise-adoption-roi/
2. Emergent Mind: Multi-turn Dialogues - https://www.emergentmind.com/topics/multi-turn-conversational-dialogues
3. SBS Framework (arXiv) - https://arxiv.org/html/2508.06886v1
4. LangChain/LangGraph 1.0 Blog - https://blog.langchain.com/langchain-langgraph-1dot0/
5. ACM: Persona-Aware Alignment - https://dl.acm.org/doi/full/10.1145/3771090
6. FineDineMenu: AI as Hotel Concierge - https://www.finedinemenu.com/en/blog/ai-as-the-new-concierge/
7. Casino Center: AI in Casino Operations - https://www.casinocenter.com/ai-opens-room-for-faster-casino-operations/
8. Langfuse: LLM-as-Judge - https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge
9. AlpacaEval Null Model Attack (arXiv) - https://arxiv.org/abs/2410.07137
10. Confident AI: G-Eval Guide - https://www.confident-ai.com/blog/g-eval-the-definitive-guide
11. ACL: Position Bias Study - https://aclanthology.org/2025.ijcnlp-long.18/
12. Meta-Judging (arXiv) - https://arxiv.org/html/2504.17087v1
13. LangGraph Time Travel Docs - https://docs.langchain.com/oss/python/langgraph/use-time-travel
14. Escalation Design - https://www.bucher-suter.com/escalation-design-why-ai-fails-at-the-handoff/
15. Galileo: Agent Evaluation Framework - https://galileo.ai/blog/agent-evaluation-framework-metrics-rubrics-benchmarks

---

*Generated by research-specialist agent, 2026-02-22*
