# R72 Research: Multi-Turn Conversation Repair for Production AI Casino Host

**Date**: 2026-02-27
**Scope**: State-of-the-art conversation repair and intent tracking for Hey Seven's AI casino host agent
**Sources**: Perplexity Deep Research (4 queries), production case studies (Intercom Fin, Zendesk AI, Ada, PolyAI, Kore.ai, Asksuite), academic research on dialogue repair, intent tracking frameworks (Rasa, Dialogflow CX, Amazon Lex, Kore.ai XO)

---

## Executive Summary

Conversation repair is the single highest-leverage area for improving Hey Seven's behavioral scores (B3 Engagement: 8.1, B4 Agentic: 6.4). The research reveals that **production systems treat repair as a first-class architectural concern**, not an afterthought. The best systems combine: (1) multi-signal failure detection (repeated questions, sentiment trajectory, multi-part collapse), (2) structured repair sequences (HEARD/HEART frameworks), and (3) explicit intent resolution tracking across turns.

**Key finding**: The most impactful improvement for Hey Seven is **intent resolution tracking** -- maintaining a structured record of what the guest asked vs what was answered, and proactively addressing gaps. This is currently missing from our architecture and would address both B3 (multi-part collapse) and B4 (proactive helpfulness).

**Key numbers**:
- Repeated question detection (embedding similarity): 0.75-0.85 cosine threshold, 5-10% false positive rate
- Frustration trajectory detection: 85-90% accuracy with ensemble signals (vs 60-70% with single-message sentiment)
- Multi-intent detection accuracy: 88-92% precision in leading models
- Intercom Fin escalation routing: 98% accuracy with custom multi-task model
- Warm handoff CSAT improvement: 2-3x over cold transfer

---

## Part 1: Failure Detection Techniques

### 1.1 Repeated Question Detection

| Method | Threshold | Latency | False Positive Rate | Best For |
|--------|-----------|---------|-------------------|----------|
| Jaccard word overlap | 0.35-0.55 | <1ms | 15-25% | Quick pre-filter |
| Embedding similarity (BERT/ADA) | 0.75-0.85 cosine | 50-150ms | 5-10% | General detection |
| Intent-based matching | Same intent + high confidence (>0.75) | 100-200ms | 2-5% | Production-grade |
| Sequence matching (slot tracking) | Same slot queried 2x without new info | <5ms | 1-3% | Slot-filling dialogs |

**What Hey Seven currently has (R71)**:
`_detect_conversation_dynamics()` in `_base.py` detects `repeated_question` via "semantic overlap detection between last 2 human messages." This is a basic embedding similarity approach.

**What's missing**:
1. **Intent-level matching**: Comparing intents, not just surface text. "What restaurants are open?" and "Where can I eat?" are the same intent, different words.
2. **Partial resolution tracking**: Guest asks about restaurants AND pool hours, bot answers restaurants only, guest re-asks about pool. This is NOT a repeated question -- it's an unresolved sub-question.
3. **Slot-level tracking**: Has the bot asked for the same clarification twice? (e.g., "How many in your party?" asked twice = bot failure)

**Recommended thresholds for Hey Seven**:
- Embedding cosine similarity > 0.80 between consecutive human messages = likely repetition
- Same intent detected twice within 3 turns without resolution = definite repetition
- Bot asks same clarifying question twice = immediate repair mode

### 1.2 Escalating Frustration Detection

**Current approach** (R71): VADER sentiment with 4 categories (positive, negative, neutral, frustrated) + `_count_consecutive_frustrated()` from message history.

**State of the art approach**: Multi-signal sentiment TRAJECTORY analysis.

**Signals to combine into frustration score (ensemble, 85-90% accuracy)**:

| Signal | Weight | Detection Method | Threshold |
|--------|--------|-----------------|-----------|
| Sentiment trajectory slope | 0.30 | Linear slope over last 5-7 turns | Negative slope > -0.15/turn for 3+ turns |
| Message length decrease | 0.15 | Avg length drops >50% vs conversation average | "Fix it" after earlier 20-word messages |
| Punctuation escalation | 0.10 | Multiple !, ?, CAPS detection | "???" or "!!!" or sustained CAPS |
| Explicit frustration keywords | 0.20 | Pattern matching | "frustrated", "angry", "confused", "annoyed" |
| Acknowledgment gaps | 0.15 | Guest says "I'm confused" + bot gives generic response | Explicit confusion statement + no repair |
| Turn count without resolution | 0.10 | Turns since last resolved intent | > 5 turns without resolution |

**Composite frustration score** > 0.65-0.75 triggers repair mode.

**Critical distinction**: Frustration AT the bot vs frustration ABOUT the situation.
- "I'm angry at the restaurant's policy" = external frustration (acknowledge, help navigate)
- "I'm frustrated that you can't understand me" = bot frustration (immediate repair/escalate)

**How to differentiate**: Look for second-person pronouns ("you", "your") combined with negative sentiment = bot-directed frustration. Third-person or situation-directed language = external frustration.

### 1.3 Multi-Part Answer Collapse Detection

**The problem**: Guest asks 3 things, bot answers 1. This is the most common chatbot failure mode and the most damaging for hospitality.

**Example**:
> Guest: "What restaurants are open tonight, what's the pool schedule, and where's the spa?"
> Bot: "Our steakhouse is open until 11 PM and the Italian restaurant until 10 PM."
> (Pool schedule and spa location unanswered)

**Detection approach (intent decomposition + state tracking)**:

```
Step 1: Decompose user message into sub-intents
  - intent_1: "restaurant_hours_inquiry"
  - intent_2: "pool_schedule_inquiry"
  - intent_3: "spa_location_inquiry"

Step 2: After bot responds, check which intents were addressed
  - intent_1: RESOLVED (restaurant info provided)
  - intent_2: UNRESOLVED (no pool info in response)
  - intent_3: UNRESOLVED (no spa info in response)

Step 3: On next turn, proactively address unresolved intents
  Bot: "I also wanted to let you know -- the pool is open until 9 PM,
        and the spa is located on the 3rd floor of the wellness center."
```

**Implementation pattern** (from Kore.ai XO and Dialogflow CX):
- Multi-intent NLU model identifies all intents in a message (88-92% precision in leading models)
- Dialogue state tracker maintains filled vs unfilled status per intent
- Response validator checks semantic coverage: does the response contain information markers matching each intent? (temporal info for schedule queries, location info for direction queries, etc.)
- If coverage < 100%, system injects remaining intents into next response

**For Hey Seven**: This is the HIGHEST IMPACT improvement. Our router dispatches to a single specialist per turn. If a guest asks about dining AND entertainment, only one specialist responds. We need:
1. Multi-intent decomposition at router level
2. Intent resolution tracking in AgentState
3. Proactive follow-up injection in persona_envelope_node

---

## Part 2: Repair Strategies (HEARD/HEART Frameworks)

### 2.1 HEARD Framework Implementation

**H - Hear**: Explicitly demonstrate comprehension of BOTH the stated problem AND the implicit emotional context.

```
Implementation:
1. Extract from user message: explicit_request, preferred_outcome,
   complaint_context, emotional_charge
2. Generate acknowledgment incorporating all 4 elements

Example:
  Guest: "I booked a standard room but the website showed oceanfront.
          This is supposed to be our anniversary trip."

  Slots extracted:
    Problem_Type: "room_discrepancy"
    Current_State: "standard_room"
    Desired_State: "oceanfront"
    Failure_Attribution: "website_mismatch"
    Emotional_Charge: "disappointment" + "occasion_importance"

  Response: "I understand you were expecting an oceanfront room for your
  anniversary, and the website may have shown something different from
  what was booked. Let me look into this right now."
```

**E - Empathize**: Express understanding without performative emotion.

```
BAD: "I'm so sorry you're experiencing this! I totally understand how
      frustrating that must be!"  (performative, generic)

GOOD: "I understand why you'd want to be closer to the beach for your
       anniversary -- that makes the experience much better."
       (validates the specific request, acknowledges the occasion)

BETTER (with context): "I see this is your anniversary trip -- let me
        find you the most romantic oceanfront room we have available."
        (turns empathy into action)
```

**A - Apologize**: Own the failure without being defensive.

```
BAD:  "I apologize for the inconvenience, I'm sorry about that."
       (defensive loop, no ownership)

BAD:  "This isn't really my fault, the system was..."
       (blame-shifting)

GOOD: "I apologize that the room booking didn't match what you expected.
       Here's what I'm doing to fix it..."
       (apologize + take ownership + immediate action)
```

**R - Resolve**: Actually address the problem, not just acknowledge it.

```
BAD:  "I'll pass this along to our team."
       (vague, no commitment)

GOOD: "I've moved you to Room 806 (oceanfront), at no additional charge
       for your two-night stay. Your new key will be waiting at the desk."
       (concrete action, specific details, confirmed completion)
```

**D - Diagnose**: Log the failure type for system improvement (backend, not user-facing).

```
Log entry: {
  "failure_type": "multi_part_collapse",
  "intents_identified": 3,
  "intents_resolved": 1,
  "guest_frustration_score": 0.72,
  "repair_strategy": "proactive_followup",
  "repair_outcome": "resolved_without_escalation"
}
```

### 2.2 HEART Framework (Organizational, Complementary to HEARD)

| Element | Implementation for Hey Seven |
|---------|----------------------------|
| **Humanize** what matters most | Escalate cases requiring judgment (compensation, policy exceptions) to humans. Don't automate service recovery decisions. |
| **Eliminate** routine busywork | Bot handles standard repairs: rebooking, information correction, alternative suggestions. Free humans for complex recovery. |
| **Amplify** capabilities | When escalating, provide human agent with: full transcript, sentiment trajectory, attempted solutions, suggested next steps. |
| **Redefine** roles | Position escalation as expert routing: "Let me connect you with our VIP host specialist" (expertise), not "I can't help" (failure). |
| **Transform** interactions | Make repair feel like service enhancement: "Our chef wants to personally recommend tonight's special for you." |

### 2.3 Graduated Repair Responses (Based on Frustration Level)

| Frustration Level | Consecutive Signals | Response Pattern |
|-------------------|-------------------|-----------------|
| Level 0: Normal | No frustration signals | Standard response |
| Level 1: Mild | 1 signal (terse reply OR slight negative) | Slightly warmer tone, check understanding |
| Level 2: Moderate | 2 signals OR repeated question | HEARD framework: "I want to make sure I help you properly. You're asking about..." |
| Level 3: Elevated | 3+ consecutive signals | Full HEARD + offer alternative: "Would you prefer I connect you with our concierge team?" |
| Level 4: Critical | Explicit anger OR 5+ turns without resolution | Immediate warm handoff with context summary |

**For Hey Seven**: We currently have a binary model (frustrated -> HEART framework at 2+ consecutive signals, per R21). The 5-level graduated model above provides more nuanced repair.

---

## Part 3: Intent Resolution Tracking Architecture

### 3.1 Intent Object Data Structure

```python
@dataclass
class IntentRecord:
    intent_id: str                    # Unique ID
    intent_type: str                  # "restaurant_query", "pool_hours", etc.
    status: Literal[
        "identified",                 # Detected in user message
        "in_progress",               # Bot is working on it
        "resolved",                  # Bot provided answer
        "partially_resolved",        # Partial info provided
        "superseded",               # User changed mind
        "abandoned",                # User moved on without resolution
        "escalated"                 # Handed to human
    ]
    created_turn: int                # Turn number when identified
    resolved_turn: int | None        # Turn number when resolved
    confidence: float                # NLU confidence in intent detection
    extracted_slots: dict            # Filled parameters
    resolution_evidence: str | None  # What info was provided
```

### 3.2 State Management Pattern for LangGraph

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # ... existing fields ...

    # NEW: Intent resolution tracking
    active_intents: Annotated[list[dict], _merge_intent_lists]
    resolved_intents: Annotated[list[dict], _append_only]

def _merge_intent_lists(existing: list | None, new: list | None) -> list:
    """Merge intent lists, updating status of existing intents."""
    if not new:
        return existing or []
    existing_map = {i["intent_id"]: i for i in (existing or [])}
    for intent in (new or []):
        existing_map[intent["intent_id"]] = intent  # Update or add
    return list(existing_map.values())
```

### 3.3 Intent Decomposition at Router Level

```python
async def router_node(state: AgentState) -> dict:
    """Decompose user message into sub-intents and track resolution."""
    user_msg = state["messages"][-1].content

    # Step 1: Detect all intents in the message
    decomposed = await intent_decomposer.ainvoke(user_msg)
    # Returns: [{"type": "restaurant_query", "slots": {...}},
    #           {"type": "pool_hours", "slots": {...}}]

    # Step 2: Create intent records
    new_intents = []
    for intent in decomposed.intents:
        new_intents.append({
            "intent_id": f"{intent.type}_{turn_count}",
            "intent_type": intent.type,
            "status": "identified",
            "created_turn": turn_count,
            "confidence": intent.confidence,
            "extracted_slots": intent.slots,
        })

    # Step 3: Route to primary specialist, mark others as pending
    primary = decomposed.intents[0]  # Highest confidence
    return {
        "query_type": primary.type,
        "active_intents": new_intents,
    }
```

### 3.4 Intent Graveyard Pattern

At conversation end or after N turns of inactivity, surface unresolved intents:

```python
def _check_unresolved_intents(state: AgentState) -> list[dict]:
    """Find intents that were identified but never resolved."""
    turn_count = len([m for m in state["messages"] if isinstance(m, HumanMessage)])
    stale_threshold = 3  # Turns without resolution

    unresolved = []
    for intent in state.get("active_intents", []):
        if intent["status"] in ("identified", "partially_resolved"):
            turns_since = turn_count - intent["created_turn"]
            if turns_since >= stale_threshold:
                unresolved.append(intent)
    return unresolved
```

When unresolved intents are found, inject a proactive follow-up:
> "By the way, you asked earlier about pool hours -- the pool is open until 9 PM tonight."

### 3.5 Platform Comparison: How Production Systems Track Intents

| Platform | Architecture | Intent Tracking | Multi-Intent | Resolution Detection |
|----------|-------------|----------------|-------------|---------------------|
| **Rasa** | Custom ML pipeline | Entity-slot tracking with FormAction | Limited (single active form) | Slot completeness check |
| **Dialogflow CX** | Flows + Pages + State Handlers | Page-level state transitions | Parallel pages in same flow | Page transition = resolved |
| **Amazon Lex** | Slot-filling state machine | Built-in slot confirmation | Sequential (one intent at a time) | All required slots filled |
| **Kore.ai XO** | Enterprise dialog engine | Multi-intent detection + group intents | Yes (parallel sub-intents) | Task completion tracking |
| **LangGraph (current)** | Custom StateGraph | No built-in tracking | No (single router dispatch) | No detection |

**Gap for Hey Seven**: We have NO intent resolution tracking. The router dispatches to one specialist per turn. Multi-intent messages lose sub-intents silently.

---

## Part 4: Product-Specific Repair Patterns

### Intercom Fin
- **Escalation routing**: Custom multi-task model predicts: (1) whether to escalate, (2) reason (8 categories), (3) matched guidelines. 98% accuracy.
- **Question-answer-feedback loop**: After answering, generates context-specific follow-up: "Did those steps resolve your knowledge base access?" (not generic "Did this help?")
- **Repair sequence**: (1) Clarify if answer was wrong or incomplete, (2) Provide supplementary info, (3) Offer escalation if still needed.
- **Custom Answers**: Companies define exact responses for known difficult patterns.

### Zendesk AI
- **Intelligent Triage**: Sentiment trajectory analysis prioritizes deteriorating conversations over high-volume stable ones.
- **Real-time coaching**: When human takes over, surfaces: what bot attempted, customer sentiment, suggested phrases, relevant articles.
- **Warm handoff**: Agent's first message acknowledges prior bot interaction: "I see you've been working with our automated system about X. Let me take a different approach."

### Ada
- **Confidence threshold tuning**: Professional tier = 0.7, Starter = 0.8, Enterprise = 0.5. Reviewed after 1-2 weeks of operation.
- **Action-oriented repair**: When detection fails, immediately invokes backend action (booking lookup, status check) rather than asking for clarification.
- **Revenue protection**: Tracks which repair failures lead to lost transactions (abandoned bookings).

### PolyAI
- **Customer-led conversation**: Allows guests to interrupt and redirect at any point. System maintains context across interruptions.
- **Raven reasoning model**: Maintains contextual memory throughout conversation. Detects subtle emotional cues.
- **Natural recovery**: Repair language matches conversation style (formal for business, casual for leisure).

### Asksuite (Hospitality-Specific)
- **1.3 billion messages annually** across 1000+ hotels.
- **Multi-turn repair loops**: Can recover from failures without escalation.
- **Deep PMS integration**: Repair involves direct system action (room change, rebooking) rather than just explanation.
- **Guest context persistence**: Maintains context across pre-arrival, check-in, mid-stay, departure.

---

## Part 5: Escalation Decision Framework

### Multi-Factor Escalation Decision Matrix

| Factor | Weight | Trigger |
|--------|--------|---------|
| Confidence below threshold | 0.25 | Bot confidence < 0.7 on response |
| Repeated question (2nd time) | 0.20 | Same intent detected twice without resolution |
| Frustration score > threshold | 0.20 | Composite frustration > 0.65 |
| Explicit escalation request | 1.00 | "Let me talk to a person" (immediate) |
| Turn count exceeded | 0.15 | > 8 turns without resolution |
| High-stakes topic | 1.00 | Compensation, complaint, regulatory (immediate) |
| Sentiment trajectory | 0.20 | Negative slope > -0.15/turn for 3+ turns |

**Composite score > 0.70 = escalate. Immediate triggers (explicit request, high-stakes) always escalate.**

### Warm vs Cold Transfer Impact

| Handoff Type | CSAT Score | Description |
|-------------|-----------|-------------|
| Cold transfer | 2.1/5.0 | Guest joins queue, agent has no context, must repeat everything |
| Warm transfer | 4.2/5.0 | Agent receives: transcript, sentiment, attempted solutions, suggested next steps |
| Whisper transfer (voice) | 4.5/5.0 | Private context to agent that guest doesn't hear |

**For Hey Seven**: Always provide warm handoff context. Include: conversation summary, guest profile, sentiment trajectory, unresolved intents, and recommended next action.

### When to Retry vs Escalate

| Situation | Action | Rationale |
|-----------|--------|-----------|
| First failure, guest calm | Retry with different approach | Most failures are recoverable |
| First failure, guest frustrated | Retry with HEARD framework | Empathy + second attempt |
| Second failure, same intent | Offer escalation as option | Guest should choose |
| Any failure on high-stakes topic | Immediate escalation | Compensation, complaints, safety |
| 3+ failures regardless | Force escalation | Continued bot interaction = harm |
| Guest explicitly requests human | Immediate escalation | Never argue with escalation requests |

---

## Recommended Approach for Hey Seven

### Architecture: Intent-Aware Conversation Manager

The highest-impact changes are:

1. **Add intent decomposition to router**: Detect multi-intent messages and create tracking records for each sub-intent.

2. **Add intent resolution tracking to AgentState**: New `active_intents` field with custom reducer that tracks identified/resolved/unresolved status per intent.

3. **Add proactive follow-up injection**: After primary specialist responds, check for unresolved sub-intents and inject follow-up information.

4. **Upgrade frustration detection to trajectory-based**: Replace binary frustrated/not-frustrated with a composite score using 6 signals (sentiment slope, message length, punctuation, keywords, acknowledgment gaps, turn count).

5. **Implement graduated repair**: 5-level response adjustment based on frustration intensity, not just binary HEART trigger.

6. **Add warm handoff context**: When escalating, format a structured handoff including: conversation summary, guest profile, unresolved intents, attempted solutions, recommended action.

### What this changes in the graph

```
Current:   router -> specialist -> validate -> respond
Proposed:  router -> specialist -> intent_checker -> validate -> respond
                                      |
                                      v (if unresolved intents)
                                  followup_specialist -> validate -> respond
```

The `intent_checker` node is lightweight (no LLM call) -- it compares resolved intents against identified intents and injects any gaps into the response or dispatches a second specialist.

---

## Implementation Roadmap

### Phase 1: Foundation (3-4 days)
- [ ] Add `IntentRecord` data structure to `state.py`
- [ ] Add `active_intents` and `resolved_intents` to `AgentState` with custom reducer
- [ ] Add intent decomposition prompt to router (structured output: list of sub-intents)
- [ ] Add resolution tracking: after specialist responds, mark matched intents as resolved
- [ ] Add 10 tests for intent decomposition and resolution tracking

### Phase 2: Repair Detection (2-3 days)
- [ ] Upgrade frustration detection from binary to 6-signal composite score
- [ ] Add sentiment trajectory tracking (sliding window, slope calculation)
- [ ] Add message length variance detection
- [ ] Implement graduated repair levels (0-4) based on composite score
- [ ] Wire graduated repair into specialist prompt injection (tone guidance)
- [ ] Add 15 tests for frustration detection and graduated repair

### Phase 3: Multi-Part Recovery (3-4 days)
- [ ] Add `intent_checker` node to graph (post-specialist, pre-validate)
- [ ] Implement unresolved intent detection (compare identified vs resolved)
- [ ] Add proactive follow-up injection for unresolved sub-intents
- [ ] Add intent graveyard check at conversation boundaries (5+ turns stale)
- [ ] Add 10 tests for multi-part recovery scenarios

### Phase 4: Escalation Framework (2-3 days)
- [ ] Implement multi-factor escalation decision matrix
- [ ] Add warm handoff context formatter
- [ ] Wire escalation triggers into graph (explicit request, high-stakes, composite score)
- [ ] Add ADR-025 documenting escalation decision criteria
- [ ] Add 10 tests for escalation scenarios

### Phase 5: Evaluation (2-3 days)
- [ ] Add 10 new B3 engagement scenarios testing multi-part recovery
- [ ] Add 5 new B4 agentic scenarios testing proactive follow-up
- [ ] Run behavioral evaluation to measure improvement
- [ ] Document results in review summary

### What NOT to build
- No acoustic frustration detection (text-only channel)
- No per-customer sarcasm/frustration baselines (insufficient data at MVP)
- No automated CSAT tracking (requires production deployment)
- No custom multi-task escalation model like Intercom Fin (their training data is 100M+ conversations; we have 0)

---

## Key Citations

1. Intercom Fin escalation routing: https://intercom.com/help/en/articles/7120684 (98% accuracy, 8 escalation categories)
2. Intercom multi-task model: Research paper on conversational escalation prediction (0.5s latency reduction)
3. Zendesk Intelligent Triage: https://www.zendesk.com/service/ai/ (sentiment trajectory prioritization)
4. Ada confidence thresholds: Ada documentation (0.5-0.8 range by tier)
5. PolyAI Raven model: https://skywork.ai/skypage/en/PolyAI-Deep-Dive (contextual memory)
6. Asksuite: 1.3B messages/year across 1000+ hotels (multi-turn repair)
7. HEARD framework in customer service: Call center de-escalation literature (Hear, Empathize, Apologize, Resolve, Diagnose)
8. HEART framework: Organizational AI implementation methodology (Humanize, Eliminate, Amplify, Redefine, Transform)
9. Dialogue state tracking: https://aclanthology.org research on DST for multi-intent conversations
10. Multi-intent detection: Leading NLU models achieve 88-92% precision on compound queries
11. Sentiment trajectory: Growth mixture models on counseling conversations (3 trajectory patterns)
12. Frustration ensemble detection: 85-90% accuracy vs 60-70% for single-message sentiment
13. Warm handoff CSAT: 2-3x improvement over cold transfer in hospitality contexts
14. Intent graveyard pattern: Kore.ai XO platform documentation (unresolved intent tracking)
15. Dialogflow CX: Page-based state management for multi-intent flows
16. Rasa intent tracking: FormAction + slot-filling with entity tracking
17. LLM-native intent tracking: Mediator-assistant architectures for intent alignment
18. Klarna AI: Conversation repair patterns in fintech (real-time breakdown detection)
19. Hoteza, Revinate Ivy: Hotel-specific chatbot repair patterns
20. Speech emotion recognition: >85% accuracy for frustration/anger/resignation classification
