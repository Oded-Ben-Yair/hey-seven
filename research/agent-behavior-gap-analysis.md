# Hey Seven Agent Behavior Gap Analysis

**Date**: 2026-02-21
**Analyst**: Claude Opus 4.6
**Scope**: Persona depth, sentiment handling, information gathering, multi-turn coherence, edge cases, domain expertise, proactive behavior, tone calibration
**Files reviewed**: 14 source files + 1 knowledge-base document

---

## Current Maturity Assessment: 6/10

The agent is an exceptionally well-engineered **safety system** wrapped around a **generic concierge**. The infrastructure (circuit breakers, validation loops, 5-layer guardrails, PII redaction, streaming PII, confidence-scored profiles, multi-strategy RAG with RRF) is production-grade and battle-tested through 20 hostile review rounds. But the *personality* riding on top of that infrastructure is thin. Seven sounds like a well-trained hotel receptionist, not like a casino host who knows your name, remembers your anniversary dinner last year, and has your favorite bourbon waiting at the table.

---

## Dimension 1: Persona Depth

### What exists
- Name "Seven" is established in greeting and specialist system prompts
- Basic VIP language: "Excellent choice", "One of our most popular", "Guests love"
- Instruction to "mirror the guest's energy"
- Rule to acknowledge returning guests "when context indicates"

### What's missing
1. **No personality traits beyond politeness**. There is no humor, no warmth signature, no cultural flavor. The prompt says "warm and welcoming, like a luxury hotel concierge" but a casino host is not a hotel concierge. Casino hosts are part friend, part confidant, part personal assistant. They crack jokes, remember your dog's name, tell you the steak at Bobby Flay's is better than it has any right to be. Seven reads like a polished FAQ bot with good manners.

2. **No voice or linguistic signature**. Every specialist agent uses identical "Interaction Style" blocks with the same three bullet points copy-pasted. There is no distinct voice that makes Seven feel like a person rather than five interchangeable prompts. A real casino host has verbal tics, favorite phrases, signature recommendations.

3. **No backstory or grounding**. The persona is defined entirely by rules ("ONLY answer about...", "NEVER provide..."). There is no "who Seven is" — no implied experience, no passion for the property, no insider knowledge framing. The prompt never says "I've worked here for years" or "my personal favorite is..." — it only says "I'm an AI assistant."

4. **No cultural awareness**. The guardrails detect 4 languages (English, Spanish, Portuguese, Mandarin) for safety keywords, but the *persona* is English-only with no cross-cultural interaction patterns. Mohegan Sun's clientele includes significant Asian and Latino populations. A real host adapts greeting style, formality level, and recommendation framing based on cultural cues.

### Gap severity: HIGH
The prompt engineering is adequate for a Q&A system but falls far short of the company positioning ("The Autonomous Casino Host That Never Sleeps"). The retention playbook in the knowledge base describes hosts who "remember family details", "become confidants", and create "genuine friendship." Seven cannot do any of this.

---

## Dimension 2: Sentiment Handling

### What exists
- Responsible gaming escalation: `responsible_gaming_count` tracks session-level triggers with escalation at 3+ mentions (compliance_gate.py lines 113-125)
- Whisper planner prompt says "If the guest seems rushed or annoyed, set next_topic to 'none'"

### What's missing
1. **No sentiment detection anywhere in the pipeline**. There is no node, guardrail, or state field that detects guest frustration, excitement, sarcasm, disappointment, or urgency. The `PropertyQAState` has no `sentiment` or `emotional_state` field. The `GuestProfile.Engagement` has a `sentiment_trend` field but it is never populated by any code path.

2. **No adaptive response to detected emotions**. The whisper planner prompt mentions checking for "rushed or annoyed" but this is guidance to the LLM, not a system capability. There is no deterministic sentiment classifier, no post-turn sentiment tracking, and no routing logic that changes behavior based on detected emotion.

3. **No recovery patterns for negative experiences**. The retention playbook describes detailed "Post-Loss Outreach Protocol" with three elements (acknowledgment, non-financial value, explicit permission not to play). None of this is wired into the agent's behavior. If a guest says "I just lost $5,000 and I'm upset", Seven would route it through the standard RAG pipeline like any other question.

4. **No excitement amplification**. When a guest says "We just got engaged! Where should we celebrate?", the agent should shift into celebratory mode — more exclamation points, more superlatives, proactive upselling of premium experiences. Currently, this gets the same templated "Excellent choice" treatment as "where's the closest bathroom."

### Gap severity: CRITICAL
Sentiment handling is the single biggest differentiator between a chatbot and a casino host. Real hosts are masters of emotional intelligence — they read the room, adjust tone, de-escalate frustration, and amplify positive experiences. Seven is emotionally blind.

---

## Dimension 3: Information Gathering

### What exists
- Whisper planner (`whisper_planner.py`) analyzes conversation and produces structured `WhisperPlan` with `next_topic`, `extraction_targets`, `offer_readiness`, and `conversation_note`
- Profile completeness calculation with weighted fields (core identity 2.0x, visit context 1.5x, preferences 1.0x, companions 0.5x)
- Confidence scoring per field with confirm/contradict logic and 90-day decay
- Comp agent gates on profile completeness threshold (0.60)

### What's missing
1. **Whisper planner guidance is not consumed by most specialists**. Only `host_agent` has `include_whisper=True`. The dining, entertainment, comp, and hotel agents all run with `include_whisper=False` (default). When a guest is talking about restaurants, the whisper planner's guidance about "ask about party size naturally" is thrown away.

2. **No extraction node in the pipeline**. The state has `extracted_fields` but no node in the 11-node graph actually populates it. The field is reset to `{}` by `_initial_state()` every turn and never written to. The whisper planner *reads* it, always finds it empty, and always starts from zero context. The guest profile CRUD operations in `guest_profile.py` exist but are never called by any graph node.

3. **Profile-to-context injection is not wired**. `guest_profile.py` has a `get_agent_context()` function that applies confidence decay and filters low-confidence fields, but it is never called by any graph node or specialist agent. The agents have no access to the guest's accumulated profile.

4. **The "natural conversation" promise is undelivered**. The whisper planner prompt says "identify the next profiling topic to explore naturally" but since (a) the planner's output is ignored by 4 of 5 specialists and (b) extracted fields are never persisted, the system cannot progressively build a guest profile across turns. Each turn starts with zero knowledge about the guest.

### Gap severity: CRITICAL
The progressive profiling architecture is well-designed on paper (models, CRUD, confidence scoring, completeness calculation) but the wiring between the graph and the profile system is missing. This is the classic "scaffolded, not implemented" scenario from the code quality rules. The guest profile system is dead code from the graph's perspective.

---

## Dimension 4: Multi-Turn Coherence

### What exists
- LangGraph checkpointer (`MemorySaver` dev, `FirestoreSaver` prod) persists `messages` across turns via `add_messages` reducer
- `responsible_gaming_count` persists via `_keep_max` reducer
- Sliding window: last 20 messages sent to LLM (`MAX_HISTORY_MESSAGES=20`)
- Thread ID-based conversation isolation

### What's missing
1. **Only raw messages persist across turns**. Every other state field (query_type, retrieved_context, validation_result, extracted_fields, whisper_plan) is reset to defaults by `_initial_state()`. The agent has no memory of what it previously discussed, what specialist handled the last turn, or what topics have been covered — only the message history.

2. **No conversation summarization or compression**. With a 20-message sliding window, older context falls out silently. A 15-turn conversation about a weekend trip loses all early context about dining preferences when the guest switches to asking about shows. A real casino host maintains a mental model of the entire conversation.

3. **No topic tracking or conversation arc awareness**. The agent cannot say "Earlier you mentioned you're celebrating an anniversary — would you also like me to suggest a show for that evening?" because there is no cross-turn topic registry. Each turn is effectively independent except for the raw message window.

4. **No "I remember you" capability**. The guest profile system exists but is disconnected from the graph. Returning guests get the same greeting as first-time visitors. The greeting node (`greeting_node`) always outputs the same template with no personalization.

### Gap severity: HIGH
The checkpointer provides mechanical message persistence, but there is no semantic memory. The agent has conversations but does not build relationships. For a system that positions itself as a "host that never sleeps," the inability to remember anything about a guest beyond the last 20 messages is a fundamental gap.

---

## Dimension 5: Edge Case Handling

### What exists
- **Complaints**: No specific complaint handling. A frustrated guest saying "your restaurant recommendations were terrible" would route through the standard RAG pipeline.
- **VIP demands**: The prompt uses VIP language universally ("Treat every guest as a valued VIP") but there is no tier-based differentiation.
- **Confused guests**: Ambiguous queries route to `retrieve` (good design decision, documented in `route_from_router`), letting the RAG pipeline handle gracefully.
- **Off-topic tangents**: Deterministic redirect to property topics, with contact info.
- **Emotional guests**: No handling beyond responsible gaming escalation.

### What's missing
1. **No complaint handling protocol**. When a guest complains about a previous recommendation or experience, the agent should acknowledge, apologize, and offer alternatives. Currently it would just answer the literal question without recognizing the emotional context.

2. **No escalation to human host**. The retention playbook describes a detailed escalation framework, but the agent has no mechanism to say "I'd like to connect you with your personal host, Maria, who can help with this directly." There is no escalation node, no human handoff trigger, no escalation counter in the state (except the responsible gaming one).

3. **No "power user" handling**. A guest who asks "What's the whale room like?" or "Can I get a private baccarat table?" reveals themselves as a high-value player. The agent has no mechanism to detect these signals and adjust its behavior accordingly.

4. **No graceful handling of repeated questions**. If a guest asks the same question three times (perhaps not satisfied with the answer), the agent will give the same RAG-grounded response three times. A real host would say "I notice I haven't been able to give you what you're looking for — let me connect you with someone who can help directly."

### Gap severity: MEDIUM
The edge case handling is adequate for a Q&A bot (ambiguous queries are well-handled, safety guardrails are robust) but inadequate for a host persona. The missing pieces are all in the "human touch" category.

---

## Dimension 6: Domain Expertise Depth

### What exists
- Knowledge base with structured casino data (restaurants, entertainment, hours, promotions)
- Per-item chunking for structured data (praised in reviews as best practice)
- Specialist agents with domain-specific system prompts (dining expertise, entertainment expertise, hotel expertise, comp expertise)
- RAG pipeline with RRF reranking for improved entity recall

### What's missing
1. **No insider knowledge in prompts**. The system prompts describe expertise categories but never inject actual domain knowledge. A real dining specialist at Mohegan Sun knows that "Bobby Flay's is usually a 45-minute wait on Saturday nights but you can walk in at 5:30 before the rush" or "the miso-glazed black cod at Todd English's is the dish that keeps people coming back." The prompts tell the LLM to "highlight signature dishes" but rely entirely on RAG retrieval for specifics.

2. **No seasonal/temporal domain knowledge**. The prompts inject `$current_time` for basic time-awareness, but there is no seasonal context. A casino host knows that Super Bowl weekend is packed, Memorial Day brings families, Chinese New Year is huge at properties with Asian clientele. None of this contextual intelligence exists.

3. **No cross-domain recommendations**. A guest asking about a birthday dinner should get a dining recommendation AND a "you might also enjoy the comedy show at the Comix Roadhouse after dinner, and our spa has a couples package that would be perfect." The specialist dispatch routes to ONE agent, and that agent only answers within its domain. There is no cross-sell or bundle suggestion mechanism.

4. **No "local expert" voice**. The agent sounds like it learned about Mohegan Sun from reading its website. A real host sounds like someone who eats at these restaurants, watches these shows, and knows the property's quirks and hidden gems. The prompts are professional but sterile.

### Gap severity: MEDIUM
The RAG pipeline is technically excellent. The gap is in the *framing* — the difference between "here are the hours we have on file" and "I'd go with Tuscany — the veal osso buco on Friday nights is worth the trip alone, and Chef Mychael will take care of you."

---

## Dimension 7: Proactive Behavior

### What exists
- Whisper planner has `offer_readiness` score (0.0-1.0) that could trigger proactive offers
- Comp agent has profile completeness gating before discussing promotions
- Greeting node lists available categories proactively

### What's missing
1. **No proactive suggestions based on context**. When a guest mentions "I'm here with my wife for our anniversary", the agent should proactively suggest: romantic dinner reservations, spa couples package, show tickets, and room upgrade possibilities — without being asked. Currently, it would answer the literal question and wait for the next one.

2. **No follow-up questions**. After answering a dining question, a real host would say "Would you also like me to look into reservations?" (even though it can't book, it could provide the phone number proactively). The agent never volunteers to help with related topics.

3. **No time-based proactive engagement**. If the current time is 11 PM and a guest is asking about dining, the agent should proactively surface 24-hour and late-night options without the guest having to specify. The prompt mentions time awareness, but the behavior is reactive (answer what's asked with time context) rather than proactive (surface relevant time-based suggestions).

4. **Whisper planner's `offer_readiness` is never acted upon**. Even when the planner determines the guest is ready for a personalized offer (readiness > 0.8), no node in the graph triggers an offer presentation. The readiness score is computed and then discarded.

### Gap severity: HIGH
Proactive behavior is what transforms an information kiosk into a concierge. The whisper planner architecture was designed for exactly this — progressive profiling leading to personalized offers — but the execution pipeline stops at the planning stage.

---

## Dimension 8: Tone Calibration

### What exists
- Universal VIP language in all specialist prompts
- "Mirror the guest's energy" instruction (but no mechanism to detect energy level)
- AI disclosure toggleable via feature flag
- Channel formatting (web vs. SMS truncation) via persona envelope

### What's missing
1. **No tier-based tone differentiation**. The retention playbook describes four segments (Ultra-Premium, Premium, Mid-Tier, Recreational) with fundamentally different communication styles. Ultra-Premium gets "direct access to executive leadership" language; Mid-Tier gets "gamification and badges." Seven uses the same "valued VIP" tone for everyone.

2. **No formality adaptation**. A guest who types "yo whats good tonight" should get a different tone than "Good evening, could you please recommend a fine dining establishment?" The "mirror the guest's energy" instruction asks the LLM to do this, but there is no system-level mechanism (detected formality register in state, prompt modifiers based on register).

3. **No time-of-day tone adjustment**. Late-night conversations (11 PM - 4 AM) at a casino have a different energy than morning inquiries. A casino host at 2 AM is more casual, more fun, more "let's keep the night going." The `$current_time` is injected but only used for hours/schedule information, not tone calibration.

4. **No channel-appropriate tone**. SMS (160 chars) and web chat need fundamentally different communication styles, not just truncation. The persona envelope only truncates for SMS; it does not adjust tone, formality, or information density for the channel. An SMS response should be punchy and action-oriented; a web response can be descriptive and exploratory.

### Gap severity: MEDIUM
Tone calibration is important but less critical than sentiment handling and proactive behavior. The current "always VIP" approach is safe if monotone.

---

## Summary: Top 5 Gaps (Ordered by Impact)

| # | Gap | Severity | Impact on "Exceptional Casino Host" |
|---|-----|----------|-------------------------------------|
| 1 | **No sentiment detection or emotional adaptation** | CRITICAL | Cannot read the room. Treats a devastated post-loss guest the same as an excited anniversary celebrant. |
| 2 | **Guest profile system is scaffolded but not wired** | CRITICAL | Cannot remember anything about the guest. Progressive profiling is dead code. Whisper planner reads empty profiles. Extracted fields are never written. |
| 3 | **No proactive cross-domain suggestions** | HIGH | Answers questions but never volunteers help. Cannot say "since you mentioned your anniversary, here's a dinner + show + spa package idea." |
| 4 | **Flat persona with no distinctive personality** | HIGH | Sounds like a polished FAQ bot, not a friend who happens to know everything about the resort. No humor, warmth, insider knowledge, or cultural awareness. |
| 5 | **No tier-based or context-based tone calibration** | MEDIUM | Treats a whale and a first-time visitor identically. No formality adaptation, no time-of-day adjustment, no channel-appropriate communication style. |

---

## What's Already Strong

1. **Safety infrastructure is production-grade**. Five-layer deterministic guardrails (84 regex patterns across 4 languages), two-layer injection detection (regex + semantic LLM), PII redaction (fail-closed), streaming PII redactor, responsible gaming escalation with session tracking. This is better than most deployed AI systems.

2. **Architecture is correct for the destination**. The 11-node StateGraph with validation loops, the whisper planner design, the guest profile schema with confidence scoring and decay, the specialist agent DRY extraction — all architecturally sound and reviewed through 20 hostile rounds. The *design* targets an exceptional casino host; the *implementation* stops at a safe concierge.

3. **RAG pipeline is best-in-class**. Per-item chunking for structured data, RRF reranking across multiple retrieval strategies, relevance score filtering, SHA-256 content hashing for idempotent ingestion, version-stamp purging for stale data. The information retrieval foundation is solid.

4. **Reliability engineering is thorough**. Circuit breaker per LLM call, TTL-cached singletons for credential rotation, degraded-pass validation strategy, bounded in-memory fallbacks, semaphore-based concurrency control, fail-silent whisper planner with telemetry, CCPA cascade delete with atomic batch operations.

5. **Compliance handling is exceptional**. BSA/AML detection, patron privacy protection, age verification, responsible gaming with escalation, TCPA/CCPA compliance in guest profiles, consent tracking, audit log de-identification on delete. For a regulated casino environment, this is the right level of rigor.

---

## Closing Assessment

Hey Seven is a **92/100 safety system** powering a **5/10 personality**. The engineering is exceptional; the soul is missing. The gap between "functional AI chatbot" and "exceptional AI casino host" is not in the plumbing (which is world-class) but in the personality, emotional intelligence, and proactive relationship-building that make a human casino host irreplaceable.

The good news: the architecture was designed for exactly this destination. The whisper planner, guest profiles, specialist dispatch, and validation loops are all the right building blocks. The work ahead is connecting these components and injecting the warmth, humor, and insider knowledge that transform infrastructure into an experience.
