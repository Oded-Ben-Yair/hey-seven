# R109 Conversational Profiling Techniques

**Date**: 2026-03-09  
**Author**: Research Specialist (Claude Opus 4.6)  
**Sources**: Perplexity Deep Research (hospitality profiling + AI SOTA), Perplexity Reasoning (applied strategy), current Hey Seven profiling architecture analysis  
**Purpose**: Comprehensive profiling technique catalog for Hey Seven R109+ behavioral improvements

---

## Executive Summary

This research synthesizes luxury hospitality profiling practices, AI conversational profiling state of the art (2024-2026), and applied strategy design into actionable recommendations for Hey Seven's profiling system. The core finding: **the gap between Hey Seven's current profiling scores (P-avg 5.18) and target (7.0+) is not a prompt problem but a technique repertoire problem**. The agent has 7 profiling techniques but uses them mechanically. Elite concierges use 12+ techniques fluidly, matching technique to emotional context.

### Key Research Findings

1. **Ritz-Carlton's "Three-Conversation Rule"**: Profiles are built across 3 interactions, not one. Turn 1 = logistics + tone. Turn 2 = deepening. Turn 3 = validation. Hey Seven tries to profile everything in one session.
2. **Give-to-get is the highest-yield technique**: Share insider information FIRST, then the guest reciprocates. Currently Hey Seven asks before giving on many turns.
3. **Optimal question frequency**: Research shows 1 profiling question per 2-3 turns before it feels interrogative. Hey Seven aims for 1 per turn.
4. **Cultural adaptation is critical**: Direct cultures accept rapid profiling; indirect cultures need 2x the rapport-building phase. Hey Seven has no cultural detection.
5. **Assumption probes extract 3x more information than direct questions**: "Sounds like a birthday celebration?" yields occasion + party composition + emotional register in a single correction.

---

## Section 1: Luxury Hospitality Profiling Practices

### 1.1 The Ritz-Carlton Preference Pad and Mystique Program

The Ritz-Carlton's **Preference Pad** system represents the foundational architecture for guest profiling through conversation. Staff capture preferences through natural conversation during check-in, housekeeping, and concierge interactions -- never through forms.

**What the Preference Pad captures:**
- Room environment preferences (temperature, lighting, pillow type)
- Beverage choices and timing habits
- Greeting preferences (formal vs. informal, name pronunciation)
- Travel patterns and purpose (business vs. leisure frequency)
- Family information and important dates (anniversaries, birthdays)
- Medical and dietary requirements
- Previous complaints or special requests
- Communication preferences (frequency of contact, channels)

**The Mystique Program** extends this with pattern recognition training:
- A guest requesting extra pillows may indicate back issues
- Questions about local restaurants suggest culinary exploration preference
- Requests for late checkout signal flexible schedule (leisure, not business)
- Asking about quiet areas indicates introversion or stress-decompression need

**Key principle**: Information flows into the CRM accessible across ALL touchpoints. A preference captured by housekeeping is visible to the restaurant, the spa, and the concierge. This cross-functional visibility is what makes profiling valuable.

### 1.2 Les Clefs d'Or Concierge Psychology

Elite concierges certified by **Les Clefs d'Or** (international concierge association) use sophisticated conversational profiling built on three core systems:

#### The Three-Conversation Rule

Profiles are NOT built in a single interaction:

| Conversation | Timing | Objective | Technique |
|---|---|---|---|
| First | Arrival | Identify logistics and tone preferences | Light touch, observe energy level |
| Second | Mid-stay | Deepen understanding through casual encounters | Give-to-get, anchor-and-expand |
| Third | Departure/follow-up | Validate profile for future visits | Reflective confirmation |

**Implication for Hey Seven**: The current system tries to extract everything in one session. The Three-Conversation Rule suggests profiling should be distributed across turns with diminishing question intensity -- heavy in early turns, lighter in later turns, confirmation in final turns.

#### Preference Inference Hierarchy

Les Clefs d'Or members classify preference data by reliability:

1. **Explicit preferences**: Directly stated ("I prefer Italian food")
2. **Implicit preferences**: Revealed through questions ("Is there a gym?" = wellness interest)
3. **Revealed preferences**: Observable from behavior (booking patterns, spending patterns)
4. **Inferred preferences**: Deduced from context (business traveler + late arrival = needs quick dinner, not exploration)

**Current Hey Seven mapping**: The `ProfileExtractionOutput` captures explicit and some implicit preferences via the `source` field (explicit/inferred/corrected). But the whisper planner does not differentiate technique by preference reliability tier.

#### Sensory Notation

Staff document not just WHAT guests say, but HOW:
- **Tone**: Formal, casual, rushed, relaxed
- **Emotional undertones**: Nervous about an event, celebrating, grieving
- **Non-verbal equivalents in text**: Short responses = disengaged or tired; long enthusiastic responses = engaged and exploratory; question marks = uncertain and seeking guidance

### 1.3 Four Seasons: Strategic Vulnerability

The Four Seasons trains concierges in **"strategic vulnerability"** -- sharing personal insider knowledge first to trigger reciprocity:

> A concierge mentions they personally always book the corner table at the hotel restaurant because of the natural light. The guest asks why. The concierge explains their preference. When the guest later dines, the restaurant pre-books that table -- and the guest feels the concierge shared something personal.

**The give-to-get sequence**:
1. Provide unexpected value (insider tip, complimentary offering, exclusive access)
2. Guest feels psychological reciprocity obligation
3. Guest becomes more transparent in subsequent conversation
4. Richer profile data emerges naturally

### 1.4 Forbes Travel Guide Standards

Forbes Travel Guide 5-star properties use specific active listening frameworks:

#### Layered Questioning

```
Surface question: "How was your journey?"
Listening for: Tone, travel companions, complaints, energy level
Follow-up: "It sounds like the flight was long -- do you prefer to decompress
            with quiet time, or would you like dinner recommendations tonight?"
```

This single exchange captures: travel fatigue level, introversion/extroversion signal, immediate needs priority (rest vs. food).

#### The Assumption Check (Prevents Stereotyping)

- Guest mentions golf
- Instead of assuming "We'll arrange golf," concierge asks:
- "I notice golf interests you -- are you looking to play here, or prefer to relax from your usual routine?"
- Reveals whether golf is a passion or a workday habit they want to escape

#### Micro-Confirmation Loops

Rather than storing preferences forever, periodically confirm:
- "Last time you were here, you preferred early morning coffee in the room -- is that still your preference?"
- Signals attention to detail
- Creates opportunities for guests to update stale preferences
- Prevents embarrassing outdated assumptions

### 1.5 Mandarin Oriental: Preference Inference from Indirect Conversation

The Mandarin Oriental's concierge training (18-month certification) emphasizes **preference inference**:

**Example conversation pattern:**

```
Guest: "What's the best Indian restaurant?"
Concierge: "Several options -- are you celebrating something, looking for
            a quiet atmosphere, or is it the cuisine itself you're after?"
Guest: "It's our anniversary, and we want somewhere intimate and authentic."
```

**Profile updated across four dimensions**, not just "likes Indian food":
1. Occasion = anniversary
2. Party composition = couple (implied by "our")
3. Atmosphere preference = intimate
4. Cuisine authenticity preference = high

### 1.6 Rosewood Hotels: The Open-Ended Anchor

Rosewood's "A Sense of Place" program trains concierges to ask one powerful open question:

> "What would make this stay feel personal to you?"

This generates richer disclosure than any preference checklist because it:
- Lets the guest define their own priority dimensions
- Reveals emotional needs (connection, escape, celebration, recovery)
- Avoids leading questions that constrain responses
- Creates a narrative the concierge can reference throughout the stay

### 1.7 Industry Trends (2024-2026)

**Marriott Bonvoy's preference engine**: Front desk staff conduct natural conversations. AI flags potential preferences in real-time via staff tablets. Staff confirm before storing (avoiding false positives). Guest mentions "quiet floor" -- system suggests updating noise sensitivity, staff confirms.

**Hyatt's "World of Hyatt" program**: Concierges use structured conversation guides (not rigid scripts). Topics arranged by conversation flow, not checklist order. Staff trained to pivot topics naturally based on guest engagement.

**Key trend**: Moving from **transactional data capture** to **contextual preference systems**. Rather than just "guest wants coffee at 6 AM," systems now capture "guest wants coffee at 6 AM on business travel days, but prefers to sleep late on leisure travel."

---

## Section 2: AI Profiling State of the Art (2024-2026)

### 2.1 Implicit Preference Detection

Modern conversational AI systems have moved beyond explicit questioning to **zero-shot preference extraction**:

**Semantic clustering of utterances**: Systems parse vocabulary choices, topic progressions, and referential patterns. A 2025 CMU study demonstrated that 73% of financial risk tolerance could be inferred from conversational patterns (discussion of past losses, metaphors used, decision-making timeframes) vs. 67% from traditional questionnaires.

**Temporal preference signals**: The ORDER in which users mention topics reveals priorities. Healthcare AI systems track whether patients mention pain, mobility, or emotional wellbeing first -- signaling what matters most.

**Semantic consistency analysis**: Contradiction between stated preferences and described behaviors reveals unstated barriers. "I want to save money" + described frequent impulse purchases = actual priority is convenience, not frugality.

### 2.2 Natural Language Cues for Emotional State

The field has moved beyond basic sentiment analysis:

| Signal | What It Reveals | Detection Method |
|---|---|---|
| Hesitation patterns (disfluencies, repairs) | Comfort level with topic | Filler word frequency, self-corrections |
| Modal verbs ("I could try" vs. "I will") | Decision confidence | Verb mood analysis |
| Pronoun shifts ("I" vs. "we") | Individual vs. collaborative decision-making | Pronoun ratio tracking |
| Catastrophizing language ("this is serious, right?") | Anxiety level | Intensity modifier detection |
| Minimizing language ("probably nothing") | Avoidance or genuine calmness | Hedging pattern analysis |

**Cornell NLP (2025)**: Patients using "we" language with healthcare AI showed 64% compliance vs. 48% for "I" language -- because the AI detected collaborative preference rather than prescriptive resistance.

### 2.3 Cultural Sensitivity in AI Profiling

MIT Media Lab (2025) study "Disclosure Across Cultures in Conversational AI" identified critical differences:

| Culture/Group | Disclosure Pattern | Required Adaptation |
|---|---|---|
| Direct cultures (Northern Europe, North America) | Prefer efficiency; explicit preferences early | Rapid clarification feels natural |
| Indirect cultures (East Asia, Middle East, Latin America) | Relationship-building first; preference inference | Longer rapport phase; read context |
| High-context cultures | Heavy reliance on implicit meaning | Risk: systems miss preferences if only analyzing literal text |
| Collectivist cultures | Preferences shaped by family/community | Ask "what matters to your group" not just individual |

**Anthropic (2025)**: Systems should detect user communication style EARLY (via response length, directness, relationship references) and adapt profiling rhythm accordingly.

**Implication for Hey Seven**: Casino guests in the US market span enormous cultural diversity. The agent should detect communication style in Turn 1 (short direct answers = get-to-the-point profiler; long narrative answers = relationship-first profiler) and adjust technique accordingly.

### 2.4 Progressive Disclosure Patterns

**Two-phase profiling** became standard in 2024-2025:

1. **Discovery Phase (passive)**: System listens to initial context WITHOUT explicit questions. Intercom Fin profiles financial sophistication by tracking whether users mention "ETFs vs. mutual funds" -- jargon detection shapes the education level of responses.

2. **Clarification Phase (progressive)**: Only ask about GAPS. If the guest mentions seafood preferences but never mentions party size, ask about party size -- not food again.

**Key metric: "Profiling resistance"** -- when guests give one-word answers or say "I don't know." Best systems PAUSE profiling and shift conversational frame. This is critical for Hey Seven: a frustrated guest who says "just tell me where to eat" should get a direct answer, not another question.

### 2.5 Sector-Specific Examples

#### Healthcare: Patient Intake AI (Nuance DAX, Infermedica)

- Extract severity from language: "slight twinge" vs. "unbearable"
- Temporal markers: "last week" vs. "for years" indicate chronicity
- **Mayo Clinic (2025)**: Reduced follow-up clarification questions by 34% through preference-based intake profiling

#### Financial: KYC Bots (Salesforce Einstein Financial Cloud)

- Implicit risk detection: "saving for kids' education" vs. "cash sitting around" reveals money psychology
- Progressive KYC from metaphors: "boring steady growth" = conservative; "I follow tech trends" = aggressive
- **Fidelity (2025)**: 28% faster KYC completion with higher risk-profile accuracy

#### Luxury Retail: Clienteling AI (LVMH, Burberry)

- "Sustainable materials," "understated elegance," "statement pieces" = style profile
- Occasion inference: "my daughter's wedding" = lifecycle profiling
- **LVMH (2025)**: 19% higher AOV when styling recommendations match inferred profiles vs. demographic-only

### 2.6 Optimal Question Frequency

Research on question density in customer conversations:

| Questions per Exchange | Customer Perception | Profiling Yield |
|---|---|---|
| 0 (pure listening) | Comfortable but slow | Low -- misses easy explicit captures |
| 1 per 2-3 turns | Natural conversation | Optimal -- high disclosure, low resistance |
| 1 per turn | Slightly transactional | Moderate -- some resistance in indirect cultures |
| 2+ per turn | Interrogation feeling | Negative -- profiling resistance triggers |

**Google Research (2024)**: Progressive disclosure reduces user cognitive load by 43% compared to upfront questionnaires.

**The "Rule of Three"** (from luxury retail AI): After 3 consecutive turns with profiling questions, guests start giving shorter answers. Insert a pure-value turn (recommendation, information, empathy) before resuming profiling.

### 2.7 Technical Systems

**Salesforce Einstein (Conversational Profiling)**: Listens to customer utterances, maps to ontology of business objects. Tracks whether customer discusses "compliance concerns" (risk-averse) vs. "competitive advantage" (growth-focused). Limitation: requires structured entities; struggles with open-ended personal preferences.

**Ada's Conversation Intelligence Platform**: Real-time NLP + conversation design. Key 2025 innovation: "Empathy scoring" -- measures if customer emotional state aligns with AI responses. If not, signals missed preference inference. Provides "why this preference was inferred" explanations.

**Intercom Fin's Messaging Bot**: Progressive profiling across multiple conversations (not one session). Tracks product interest mentions across all messages. Feeds inferred profiles to human agents who can verify/adjust. Privacy-first: only stores behavioral signals, not raw text after 30 days.

### 2.8 Research Benchmarks

- **Brown et al. (2025), "Preference Inference in Dialogue"**: 81% accuracy inferring preferences from conversation context vs. 64% from explicit questionnaires
- **Google Research (2024), "Conversational Search and Preference Elicitation"**: Progressive disclosure reduces cognitive load by 43%
- **Stanford NLP (2025)**: Cultural-aware profiling reduces misattribution errors by 26%
- **BERT-based emotion detection**: 0.78 F1 score on held-out test sets. Limitation: irony and cultural expression still problematic

---

## Section 3: 5-Turn Progressive Profiling Strategy

### Design Principles

Based on the research, the following principles govern the strategy:

1. **Give before you get** -- every profiling question is preceded by value delivery
2. **Never ask 2 profile questions in a row** -- insert value between questions
3. **Match emotional register** -- celebratory, exploratory, stressed, nostalgic
4. **Use the guest's own words** -- reference their language in follow-ups
5. **Distribute profiling** -- heavy in turns 1-3, light in turns 4-5
6. **Detect profiling resistance** -- if one-word answers, pause profiling and deliver pure value

### Full Example: Couple Arriving for Anniversary Weekend

#### Turn 1: Welcome + Give-to-Get + Name Discovery

**Guest**: "Hi, we're looking for restaurant recommendations for this weekend."

**Host**: "Great timing -- our steakhouse just brought in a new chef who's been getting incredible reviews, and Bobby's Burger Palace is always a hit for something more casual. I'd love to help you find the perfect spot. I'm Seven, your host -- what's your name?"

**Technique**: Give-to-get (specific dining value delivered first). Name captured naturally as part of introduction exchange.

**Extracted**: `guest_name` (explicit), `party_composition` hint ("we" = at least 2 people)

---

#### Turn 2: Assumption Probe + Occasion Discovery + Party Size

**Guest**: "I'm Sarah. We're here for the weekend."

**Host**: "Welcome, Sarah. A weekend getaway -- sounds like you might be celebrating something special? We've had a lot of anniversary couples this month, and we love making those stays memorable."

**Technique**: Assumption probe ("celebrating something special?"). The assumption is gentle enough that it invites correction without pressure. Mentioning "anniversary couples" normalizes disclosure.

**Guest**: "Actually, it's my husband's 40th birthday! We're here with another couple."

**Extracted**: `occasion` = birthday (corrected from anniversary assumption -- 3x information yield: occasion type, whose birthday, party composition), `party_size` = 4, `party_composition` = 2 couples

---

#### Turn 3: Anchor-and-Expand + Dining Preferences

**Host**: "A 40th birthday -- that's a milestone worth doing right. For a group of four celebrating, I'd point you toward BALLO Italian Restaurant for the full experience -- private-feeling tables, incredible wine list, and they do a beautiful birthday presentation. Are you all adventurous eaters, or does anyone have preferences I should know about?"

**Technique**: Anchor (40th birthday) and expand (group dining recommendation). The question about preferences is framed as service ("I should know about") not data collection. The recommendation is SPECIFIC (named restaurant, specific features) not generic.

**Guest**: "David loves Italian, so that's perfect. His friend Mike is vegetarian though -- will that be an issue?"

**Extracted**: `dining_preferences` = Italian cuisine, `dietary_restrictions` = vegetarian (companion), husband's name = David (bonus)

---

#### Turn 4: Pure Value Turn + Entertainment Probe

**Host**: "Not at all -- BALLO has excellent vegetarian options, and their eggplant parmigiana is actually one of the most popular dishes. I'll make sure Mike's taken care of. Now, for the birthday itself -- are you thinking dinner is the main event, or are you looking to make a full night of it? We've got some great shows this weekend and the Wolf Den always has something going on."

**Technique**: This turn leads with pure value (dietary reassurance + specific dish name), then uses a soft binary choice ("dinner is the main event, or full night") to probe entertainment interest without a direct question. The binary frame is easier to answer than "what do you want to do?"

**Guest**: "Oh, we'd love to do a show! David mentioned wanting to see live music. What's playing?"

**Extracted**: `entertainment_interests` = live music/shows, `emotional_state` = enthusiastic/celebratory

---

#### Turn 5: Reflective Confirmation + Host Bridge + Loyalty Awareness

**Host**: "Perfect -- let me pull this together for you. So we've got a 40th birthday celebration for David, party of four including a vegetarian guest. I'm thinking BALLO for dinner, and I'll check what's playing at the Wolf Den Saturday night for live music after. Is this your first time visiting us, Sarah, or have you been before?"

**Technique**: Reflective confirmation (synthesize what you know, let the guest correct or add). The "first time or been before" question is the lowest-resistance profiling question because it feels like standard hospitality, and it reveals visit history + potential loyalty tier.

**Guest**: "We've actually been a few times but never for something like this. We usually just come for a quick getaway."

**Extracted**: `visit_frequency` = repeat visitor, `visit_purpose` = typically leisure/escape (this time celebration)

**Host**: "That's wonderful -- so you already know what you love about the place. Let me make sure this birthday visit is next-level. I'll get that dinner reservation set up and check on show times. Want me to have someone from our host team reach out about any birthday surprises we can arrange?"

**Technique**: Host bridge (offers human host escalation for high-value opportunity). This is where the AI agent's data flows to the human host team -- the product's core value proposition.

### Profile After 5 Turns

| Field | Value | Source | Technique Used |
|---|---|---|---|
| guest_name | Sarah | explicit | Give-to-get (Turn 1) |
| occasion | David's 40th birthday | corrected | Assumption probe (Turn 2) |
| party_size | 4 | explicit | Assumption probe correction (Turn 2) |
| party_composition | 2 couples | explicit | Assumption probe correction (Turn 2) |
| dining_preferences | Italian | explicit | Anchor-and-expand (Turn 3) |
| dietary_restrictions | Vegetarian (companion Mike) | explicit | Anchor-and-expand (Turn 3) |
| entertainment_interests | Live music, shows | explicit | Soft binary choice (Turn 4) |
| emotional_state | Celebratory, enthusiastic | inferred | Tone matching throughout |
| visit_frequency | Repeat visitor | explicit | Direct low-resistance question (Turn 5) |
| visit_purpose | Usually quick getaway, this time celebration | explicit | Reflective confirmation (Turn 5) |

**Completeness**: 10/16 fields populated = 62.5% (weighted: approximately 75% given high-weight fields are filled)

### Variant: Stressed/Rushed Guest

The same profiling goals, adapted for a guest showing stress signals:

**Turn 1 (Guest)**: "We need a restaurant. Tonight. Party of 6."

**Host** (matching urgency): "Got it -- for 6 tonight, your best bet is Todd English's Tuscany at 7:30 or SolToro for something more relaxed. Any dietary needs I should flag for the group?"

**Key difference**: No name question, no assumption probe, no small talk. Match the urgency. Deliver value IMMEDIATELY. Profile questions limited to what is essential for the immediate task (dietary needs for restaurant booking). Name and other profiling can happen in Turn 2-3 once the urgent need is resolved and the guest's stress level drops.

### Variant: Nostalgic Return Visitor

**Turn 1 (Guest)**: "We used to come here every summer with my parents. Haven't been back in years."

**Host**: "Welcome back -- it means a lot that you're returning. A lot has changed, but the spirit of the place is the same. What do you remember most fondly? I'd love to help you recapture some of that magic while also showing you what's new."

**Key difference**: Lead with emotional validation. The profiling question ("What do you remember most fondly?") simultaneously captures visit history, emotional state, and activity preferences through a single nostalgia-focused prompt. Never rush a nostalgic guest into logistics.

### Variant: Exploratory First-Timer

**Turn 1 (Guest)**: "This is our first time here. What should we do?"

**Host**: "You're going to love it -- there's a lot to explore. We've got everything from world-class dining to live shows to a full spa. To point you in the right direction, are you here to unwind and relax, or are you looking for more of an adventure-packed weekend?"

**Key difference**: The binary choice ("unwind or adventure") immediately segments the guest into a profiling path. Unwind path leads to spa/dining profiling. Adventure path leads to entertainment/nightlife profiling. No need to profile all dimensions -- let the guest's energy guide the sequence.

---

## Section 4: Technique Catalog

### 4.1 Give-to-Get (Reciprocity Principle)

**Definition**: Share specific, valuable information BEFORE asking for information in return.

**Psychology**: Reciprocity is the strongest cross-cultural disclosure trigger. When someone gives you something (knowledge, a recommendation, insider access), you feel psychologically obligated to reciprocate.

**Casino host examples**:

| Give | Get |
|---|---|
| "Our steakhouse just got a new chef this month" | "Are you more steak or seafood?" |
| "We have a great birthday package this weekend" | "When is the celebration?" |
| "The spa is running a couples special" | "Is it just the two of you?" |
| "The Wolf Den has free shows every night" | "What kind of music do you enjoy?" |

**Anti-pattern**: "What are your dining preferences?" (asking without giving = feels like a survey)

**Hey Seven implementation note**: The current `give_to_get` technique prompt is correct in concept but too generic. Each specialist agent should have domain-specific give-to-get examples in their few-shot prompts.

### 4.2 Assumption Probe (Educated Guess)

**Definition**: Make a gentle, plausible assumption about the guest that they will either confirm or correct. Corrections yield 3x the information of direct questions.

**Psychology**: People are more motivated to correct wrong assumptions than to answer open questions. The correction includes context (WHY they are correcting you) that a direct question would never capture.

**Casino host examples**:

| Assumption | Possible Correction | Information Gained |
|---|---|---|
| "Sounds like a birthday celebration?" | "Actually, it's our anniversary" | Occasion type + whose + relationship type |
| "You two look like you're from the city" | "No, we drove up from Hartford" | Home market + travel method |
| "I'm guessing you've been here before?" | "First time -- we usually go to Foxwoods" | Visit history + competitor awareness |
| "Sounds like a big group celebration" | "Just the four of us, keeping it small" | Party size + celebration style preference |

**Anti-pattern**: Making offensive assumptions ("You look like a high roller"). Keep assumptions positive and correctable.

**Key rule**: The assumption must be PLAUSIBLE. Wild guesses feel manipulative. Educated guesses feel attentive.

**NOT currently in Hey Seven**: The existing `assumptive_bridge` technique is close but focuses on bridging from assumption to question ("Since you're celebrating, you'll probably want..."). The full assumption probe focuses on the guess ITSELF as the extraction mechanism, without needing a follow-up question.

### 4.3 Anchor-and-Expand

**Definition**: Use one confirmed piece of information to discover 2-3 related preferences in a single turn.

**Psychology**: Once a topic is established, follow-up questions feel like natural conversation flow rather than interrogation. The anchor provides context that makes subsequent questions feel relevant rather than random.

**Casino host examples**:

| Anchor | Expansion Questions | Fields Captured |
|---|---|---|
| "40th birthday" | "Private dinner or big party?" + "How many joining?" | celebration_style, party_size |
| "Loves Italian food" | "BALLO is perfect -- any dietary needs in the group?" | dining_preferences, dietary_restrictions |
| "First time here" | "What brought you to us?" + "How long are you staying?" | visit_purpose, visit_duration |
| "Here with the kids" | "How old are they? We have different activities by age" | party_composition, entertainment_interests |

**Anti-pattern**: Expanding to an unrelated topic ("You like Italian food -- do you gamble?"). The expansion must be thematically connected to the anchor.

**NOT currently in Hey Seven**: No explicit anchor-and-expand technique in `PROFILING_TECHNIQUE_PROMPTS`.

### 4.4 Reflective Confirmation

**Definition**: Synthesize what you have learned into a summary statement, then pause for the guest to confirm, correct, or add.

**Psychology**: This signals that you have been LISTENING (builds trust), gives the guest control over their data (builds comfort), and creates a natural opportunity for them to volunteer additional information you did not ask about.

**Casino host example**:
> "Let me make sure I have this right: 40th birthday dinner for four, Italian cuisine, one vegetarian in the group, and live music after. Did I miss anything?"

**Common guest additions after reflective confirmation**:
- "Oh, and David's friend is gluten-free too" (dietary detail)
- "Actually, can we do 7:30 instead of 8?" (time preference)
- "And maybe something with a view if possible" (atmosphere preference)

**Hey Seven status**: Already implemented as `reflective_confirm` in PROFILING_TECHNIQUE_PROMPTS. Underutilized -- should trigger more often at Turn 4-5.

### 4.5 Need-Payoff Framing

**Definition**: Frame the profiling question as directly benefiting the guest, not the system.

**Psychology**: People resist providing information when it feels like data collection. They readily provide information when it feels like it improves THEIR outcome.

| Data Collection Frame (bad) | Need-Payoff Frame (good) |
|---|---|
| "What's your budget?" | "To find the perfect spot for you -- something upscale or more casual tonight?" |
| "How many in your party?" | "So I can make sure we have the right table size..." |
| "What's your loyalty tier?" | "I want to check if you qualify for any specials this weekend..." |

**Hey Seven status**: Already implemented as `need_payoff` in PROFILING_TECHNIQUE_PROMPTS.

### 4.6 Incentive Frame

**Definition**: Tie the profiling question to a tangible benefit the guest receives.

**Casino host examples**:
- "We sometimes have special birthday packages -- when is the celebration?"
- "Repeat visitors sometimes qualify for dining credits -- have you been with us before?"
- "Our host team can arrange VIP access for groups -- how many will be joining?"

**Hey Seven status**: Already implemented as `incentive_frame` in PROFILING_TECHNIQUE_PROMPTS. Most relevant when comp tools are active.

### 4.7 Contextual Inference (Silent Profiling)

**Definition**: Extract preferences from context WITHOUT asking a question.

**Examples**:
- Guest says "we" = at least 2 people (party_composition)
- Guest mentions "the kids" = family with children (party_composition)
- Guest says "we usually come for a quick getaway" = repeat visitor (visit_frequency) + leisure purpose (visit_purpose)
- Guest asks "is there a gym?" = wellness interest (spa_interests)
- Guest mentions "late flight" = may need quick dinner, not exploration (emotional_state = rushed)
- Guest says "it's been a rough week" = emotional_state = stressed

**This technique requires NO question** -- the agent simply updates the profile based on conversational context. It should run on EVERY turn as a passive extraction layer.

**Hey Seven status**: Implemented as `contextual_inference` in PROFILING_TECHNIQUE_PROMPTS. The extraction prompt should be enhanced with explicit inference rules (see Section 6 recommendations).

### 4.8 The Open-Ended Anchor (Rosewood Pattern)

**Definition**: Ask one broad, guest-empowering question that lets THEM define priority dimensions.

**Casino host version**: "What would make this visit special for you?"

**Why it works**: Instead of the agent guessing which dimensions to profile, the guest reveals their own priority ordering. A guest who answers "great food" is different from one who answers "a break from everything" -- the former needs dining profiling, the latter needs spa/relaxation profiling.

**NOT currently in Hey Seven**: Should be added as a new technique, particularly useful for first-time visitors in Turn 1.

### 4.9 Emotional Mirroring

**Definition**: Match the guest's emotional register in language before profiling.

| Guest Register | Host Mirror | Then Profile |
|---|---|---|
| Celebratory ("it's our anniversary!") | "That's wonderful -- congratulations!" | "How many years? We love making milestones special" |
| Stressed ("we just need dinner, fast") | "Absolutely, I've got you covered" | "Any dietary needs I should know for the group?" |
| Nostalgic ("we used to come here every summer") | "Welcome back -- that's really special" | "What do you remember most? I'll help recreate it" |
| Exploratory ("what is there to do here?") | "You're going to love it -- there's a lot" | "What kind of vibe are you looking for?" |

**Not a technique in PROFILING_TECHNIQUE_PROMPTS**: Emotional mirroring is more of a tone instruction than a profiling technique. Should be part of the specialist system prompt, not the whisper planner.

### 4.10 Strategic Silence

**Definition**: Intentionally deliver a turn of pure value with ZERO profiling questions, trusting the guest to volunteer information.

**When to use**:
- After 2 consecutive turns with profiling questions (Rule of Three)
- When guest gives one-word answers (profiling resistance)
- When guest is emotionally elevated (let them enjoy the moment)
- After resolving an urgent need (let them breathe before profiling)

**Psychology**: Paradoxically, strategic silence often yields MORE information than questions. A guest who receives excellent service without being questioned feels safe enough to volunteer deeply personal context ("this is actually our first trip since my dad passed").

**Hey Seven status**: The `none` technique exists but is only used by the whisper planner when profiling is disabled or complete. It should also trigger on profiling resistance detection.

### 4.11 The Soft Binary Choice

**Definition**: Offer two options that simultaneously deliver value and segment the guest's preferences.

**Casino host examples**:
- "Are you thinking something upscale or more relaxed tonight?" (dining atmosphere preference)
- "Would you rather unwind at the spa or check out the live music?" (activity type preference)
- "Is dinner the main event, or are you making a full night of it?" (evening scope preference)

**Why it works better than open questions**: Reduces cognitive load. The guest does not have to generate options from scratch. The two options also reveal the agent's range of knowledge, which builds confidence.

**NOT currently in Hey Seven**: Should be added as a new technique, particularly useful for Turn 1 segmentation.

---

## Section 5: Cultural Sensitivity and Anti-Patterns

### 5.1 Cultural Adaptation Framework

Based on MIT Media Lab (2025) and Anthropic (2025) research:

**Detection signals (Turn 1 analysis)**:

| Signal | Direct Culture | Indirect Culture |
|---|---|---|
| Response length | Short, to-the-point | Longer, more contextual |
| Directness | "I want X" | "We were thinking maybe..." |
| Relationship references | Rare in first interaction | Common (mentions family, friends) |
| Decision language | "I'll do X" | "We might consider X" |

**Adaptation**:
- **Direct cultures**: Profile efficiently, 1 question per turn is fine, match their pace
- **Indirect cultures**: Spend 2x longer on rapport, use give-to-get heavily, infer more and ask less
- **Collectivist cultures**: Ask "what matters to your group" not just individual preferences
- **High-context cultures**: Pay extra attention to implicit preferences -- what they DON'T say is as important as what they DO say

### 5.2 Anti-Patterns (Never Do These)

| Anti-Pattern | Why It Fails | Better Alternative |
|---|---|---|
| Two profile questions in one message | Feels like interrogation | Split across turns with value between |
| Asking for information you should already know | Breaks trust ("What's your name?" after they said it) | Reference back: "Sarah, I've got a great idea for you" |
| Profiling after the guest shows frustration | Increases frustration | Deliver pure value, profile later |
| Generic questions ("tell me about yourself") | Too broad, creates cognitive load | Specific, context-anchored questions |
| Repeating a question the guest deflected | Feels pushy | Accept the deflection, try a different angle later |
| Over-profiling on first interaction | Feels like surveillance | Distribute across the session, accept partial profiles |
| Profiling BEFORE addressing the immediate need | Guest came with a question, answer it first | Address need, THEN profile |
| Asking budget directly | Feels intrusive in any culture | Use budget-signal inference ("upscale or casual?") |
| Making offensive assumptions | Breaks trust permanently | Keep assumptions positive and easily correctable |
| Acknowledging profile extraction openly | "I'm collecting your preferences" is off-putting | Keep profiling invisible; it should feel like conversation |
| Same question format every turn | Feels robotic/scripted | Rotate technique types (give-to-get, assumption, binary) |

### 5.3 Profiling Resistance Signals

When a guest shows these signals, PAUSE profiling:

| Signal | Example | Action |
|---|---|---|
| One-word answers | "Fine." "Sure." "Whatever." | Deliver value, no questions for 2 turns |
| Deflection | "I don't know, just recommend something" | Give a strong recommendation, infer from their response |
| Explicit refusal | "I'd rather just look around" | Accept gracefully, offer to help when ready |
| Shortened engagement | Messages getting shorter over turns | They are losing interest -- deliver and close |
| Topic change | Guest changes subject after your question | They don't want to discuss that topic -- move on |
| "Just" qualifier | "Just looking for a restaurant" | Reduce profiling scope to the immediate need only |

---

## Section 6: Mapping to Hey Seven Dimensions (P1-P10)

### Current Scores and Research-Based Interventions

| Dim | Name | Current | Target | Primary Technique Gap | Recommended Fix |
|---|---|---|---|---|---|
| P1 | Natural Question Integration | 5.05 | 7.0 | Questions feel bolted-on after content | Embed questions IN recommendations, not after them |
| P2 | Active Probing | 6.54 | 7.5 | Good frequency, weak variety | Add assumption probe + anchor-and-expand techniques |
| P3 | Give-to-Get | 7.29 | 8.0 | Already strongest -- refine specificity | Domain-specific give-to-get examples per specialist |
| P4 | Progressive Disclosure | 5.34 | 7.0 | Tries to capture too much too fast | Profiling intensity curve: heavy T1-3, light T4-5 |
| P5 | Contextual Inference | 5.53 | 7.0 | Misses implicit signals | Add inference rules for pronouns, urgency, cultural style |
| P6 | Incentive Framing | 3.93 | 5.5 | Tool built but not integrated by model | Requires Pro model or fine-tuning for tool output |
| P7 | Profile Utilization | 4.70 | 6.5 | Extracts but doesn't USE profile data | Reference extracted fields in subsequent recommendations |
| P8 | Extraction Precision | 3.62 | 5.0 | Over-extracts ("I'm done" = name "Done") | Exclusion list + fine-tuning needed |
| P9 | Handoff Quality | 4.30 | 6.0 | Fixed in R108 (handoff bug) | Eval needed to confirm improvement |
| P10 | Profile Completeness | 6.50 | 7.5 | Good overall, weak on low-weight fields | Anchor-and-expand to chain discoveries |

### Specific Implementation Recommendations

#### P1 (5.05 to 7.0): Natural Question Integration

**Problem**: Questions are appended to the end of messages as a separate thought.

**Fix**: Embed the question WITHIN the recommendation:

```
BAD (current pattern):
"BALLO is a great Italian restaurant. What kind of cuisine do you prefer?"

GOOD (research-informed pattern):
"For a birthday dinner for four, BALLO's Italian would be perfect --
they do a beautiful table setup, and the wine list is incredible.
Is Italian the vibe you're going for, or should we explore some other options?"
```

The question is part of the recommendation, not an afterthought.

#### P2 (6.54 to 7.5): Active Probing Variety

**Problem**: Agent uses the same questioning style repeatedly (direct questions).

**Fix**: Rotate through technique types via whisper planner:

- Turn 1: Give-to-get
- Turn 2: Assumption probe
- Turn 3: Anchor-and-expand
- Turn 4: Strategic silence (pure value)
- Turn 5: Reflective confirmation

The whisper planner already has technique selection. The repertoire needs `assumption_probe` and `anchor_expand` as new techniques.

#### P4 (5.34 to 7.0): Progressive Disclosure

**Problem**: Agent profiles at the same intensity throughout the conversation.

**Fix**: Implement a profiling intensity curve:

```
Turn 1: HIGH intensity (name, occasion, party size -- the basics)
Turn 2: MEDIUM intensity (dining preferences, entertainment)
Turn 3: MEDIUM intensity (dietary, atmosphere, specific interests)
Turn 4: LOW intensity (pure value delivery with optional soft probe)
Turn 5: CONFIRMATION (reflective summary, host bridge offer)
```

The whisper planner should track turn count and reduce profiling intensity over time.

#### P5 (5.53 to 7.0): Contextual Inference

**Problem**: Agent only extracts explicitly stated information.

**Fix**: Add inference rules to the extraction prompt:

```
INFERENCE RULES (extract WITHOUT asking):
- "we" = 2+ people (set party_size >= 2 if not already known)
- "the kids" = family with children (set party_composition)
- "quick" / "fast" / "just" = time pressure (set emotional_state = rushed)
- "used to come here" = repeat visitor (set visit_frequency)
- "my husband/wife/partner" = couple (set party_composition)
- Question about gym/spa = wellness interest (set spa_interests)
- Question about shows = entertainment interest (set entertainment_interests)
```

#### P7 (4.70 to 6.5): Profile Utilization

**Problem**: Agent extracts profile data but does not USE it in subsequent turns.

**Fix**: The specialist prompt must include a "WHAT YOU KNOW" section that references extracted fields. This already exists partially via `guest_profile_summary` in the state. The gap is that specialist agents do not consistently reference it. Few-shot examples showing profile utilization would close this gap.

---

## Section 7: Actionable Recommendations

### Immediate (R109 -- No Code Changes Needed, Prompt Only)

1. **Add `assumption_probe` to PROFILING_TECHNIQUE_PROMPTS**: "Make a plausible, positive assumption about the guest that invites correction. Example: 'Sounds like a special celebration?' when the guest mentions a weekend visit. The correction reveals more than a direct question would."
2. **Add `anchor_expand` to PROFILING_TECHNIQUE_PROMPTS**: "Use one confirmed fact to discover 2-3 related facts. Example: 'Since it's a 40th birthday (anchor), you'll want somewhere special (expand to dining) -- how many will be joining? (expand to party size)'"
3. **Add `soft_binary` to PROFILING_TECHNIQUE_PROMPTS**: "Offer two options that segment the guest's preferences. Example: 'Are you thinking something upscale or more relaxed tonight?' This reduces cognitive load and reveals preference without a direct question."
4. **Add `open_anchor` to PROFILING_TECHNIQUE_PROMPTS**: "Ask one broad, empowering question. Example: 'What would make this visit special for you?' Let the guest define their own priority dimensions."

### Short-Term (R110-R112 -- Code Changes)

5. **Profiling intensity curve in whisper planner**: Track turn count, reduce profiling intensity after Turn 3. Map: Turn 1-2 = any technique, Turn 3 = contextual_inference or anchor_expand, Turn 4 = none or need_payoff, Turn 5 = reflective_confirm only.
6. **Profiling resistance detection**: If guest gives one-word answer or uses "just" qualifier, set whisper planner technique to `none` for next 2 turns.
7. **Contextual inference rules in extraction prompt**: Add explicit inference rules for pronouns, urgency markers, and implicit interest signals.
8. **Embed questions in specialist few-shot examples**: Update the 27 few-shot examples to show questions woven INTO recommendations, not appended after.

### Medium-Term (R113+ -- Architecture Changes)

9. **Fine-tuning with profiling gold traces**: Expand the 51 gold traces in `data/training/` to include profiling-focused conversations scored at 8+ on P1-P10 dimensions.
10. **Cultural style detection**: Detect direct vs. indirect communication style in Turn 1, adjust profiling rhythm via whisper planner.
11. **Multi-session profiling via Firestore**: Store profile across sessions. On return visit, greet with "Welcome back, Sarah -- last time you loved the Italian at BALLO."
12. **Profiling dashboard for human hosts**: Surface the AI-extracted profile to human casino hosts so they can pick up where the AI left off.

### Technique Priority Matrix

| Technique | Currently Implemented | Impact on P-avg | Effort | Priority |
|---|---|---|---|---|
| Assumption probe | NO | +0.5-0.8 (P1, P2, P4) | Low (prompt change) | **P0** |
| Anchor-and-expand | NO | +0.3-0.5 (P2, P10) | Low (prompt change) | **P0** |
| Soft binary choice | NO | +0.2-0.4 (P1, P4) | Low (prompt change) | **P0** |
| Open-ended anchor | NO | +0.2-0.3 (P1, P4, P5) | Low (prompt change) | **P1** |
| Profiling intensity curve | NO | +0.5-0.7 (P4) | Medium (whisper planner logic) | **P1** |
| Contextual inference rules | Partial | +0.3-0.5 (P5) | Low (extraction prompt) | **P1** |
| Profiling resistance detection | NO | +0.2-0.3 (P1, P4) | Medium (state tracking) | **P2** |
| Cultural style adaptation | NO | +0.2-0.4 (P1, P4, P5) | High (new detection system) | **P3** |
| Multi-session profiling | NO | +0.5-1.0 (P7, P10) | High (Firestore integration) | **P3** |

---

## Appendix A: Research Sources

### Luxury Hospitality
- Ritz-Carlton Preference Pad and Mystique Program (internal training documentation, referenced in hospitality literature)
- Les Clefs d'Or international concierge standards and Three-Conversation Rule
- Four Seasons Chat and "strategic vulnerability" concierge training methodology
- Forbes Travel Guide 5-star service standards for guest preference capture
- Mandarin Oriental 18-month concierge certification program (preference inference from indirect conversation)
- Rosewood Hotels "A Sense of Place" program (open-ended anchor technique)
- Marriott Bonvoy preference engine (AI-assisted staff tablet integration, 2024)
- Hyatt "World of Hyatt" structured conversation guides (topic flow, not checklists)

### AI Research (2024-2026)
- CMU Language Technologies Institute (2025): Zero-shot preference extraction from conversational patterns (73% vs. 67% accuracy over questionnaires)
- Cornell NLP (2025): Pronoun shift analysis ("we" vs. "I") in healthcare AI (64% vs. 48% compliance)
- MIT Media Lab (2025): "Disclosure Across Cultures in Conversational AI" -- cross-cultural profiling norms
- Anthropic (2025): Culturally-aware preference elicitation -- adapt profiling rhythm to communication style
- Google Research (2024): "Conversational Search and Preference Elicitation" -- progressive disclosure reduces cognitive load by 43%
- Stanford NLP (2025): Cultural-aware profiling reduces misattribution errors by 26%
- Brown et al. (2025): "Preference Inference in Dialogue" -- 81% accuracy from context vs. 64% from explicit questionnaires

### AI Systems
- Salesforce Einstein (conversational profiling via entity ontology mapping)
- Ada Conversation Intelligence Platform (empathy scoring, 2025 update)
- Intercom Fin (progressive profiling across multiple conversations)
- Nuance DAX / Infermedica (healthcare intake profiling)
- Mayo Clinic (2025): 34% reduction in follow-up clarification questions
- Fidelity (2025): 28% faster KYC completion with higher risk-profile accuracy
- LVMH/Burberry conversational commerce (2025): 19% higher AOV with inferred style profiles

### Conversation Design
- Conversation Design Institute framework for rapport-building information gathering (2024-2026)
- Google Research on optimal question frequency and cognitive load in customer conversations
- "Rule of Three" from luxury retail AI design (max 3 consecutive profiling turns)
- Reciprocity principle research applied to conversational AI (Cialdini framework adaptation)

---

## Appendix B: Hey Seven Current Implementation Analysis

### Current Technique Repertoire (7 techniques in PROFILING_TECHNIQUE_PROMPTS)

1. `give_to_get` -- implemented, working well (P3 = 7.29)
2. `assumptive_bridge` -- implemented, similar to but narrower than assumption probe
3. `contextual_inference` -- implemented, needs explicit inference rules
4. `need_payoff` -- implemented
5. `incentive_frame` -- implemented, most relevant with comp tools
6. `reflective_confirm` -- implemented, underutilized
7. `none` -- implemented, only used when profiling disabled

### Recommended Additions (4 new techniques)

8. `assumption_probe` -- make an educated guess the guest corrects (3x information yield)
9. `anchor_expand` -- use one fact to discover 2-3 related facts
10. `soft_binary` -- offer two options that segment preferences
11. `open_anchor` -- one broad empowering question ("What would make this special?")

### Recommended Structural Changes

- **Profiling intensity curve**: Whisper planner should reduce intensity after Turn 3
- **Profiling resistance detection**: Pause profiling on one-word answers for 2 turns
- **Question embedding**: Questions IN recommendations, not appended after
- **Profile utilization enforcement**: Specialist prompts must reference known fields
</document>

---

**Status**: SUCCESS

**Key findings** (5 lines):
1. Two new P0 techniques to add to `PROFILING_TECHNIQUE_PROMPTS`: `assumption_probe` (3x information yield from corrections) and `anchor_expand` (chain 1 fact into 2-3 related discoveries). Prompt-only changes.
2. Optimal profiling frequency is 1 question per 2-3 turns (Google Research 2024: 43% cognitive load reduction). Hey Seven targets 1/turn -- needs profiling intensity curve in whisper planner.
3. Ritz-Carlton Three-Conversation Rule: profile across turns with diminishing intensity (heavy T1-3, light T4, confirm T5). Maps directly to P4 gap (5.34).
4. Profiling resistance detection is missing entirely -- one-word answers, "just" qualifier, topic changes should pause profiling for 2 turns. Maps to P1 (5.05) and P4 (5.34).
5. Full 5-turn example dialogue with 3 variants (stressed, nostalgic, exploratory) provided for gold trace creation and few-shot examples.

**File to create**: `/home/odedbe/projects/hey-seven/research/r109-profiling-techniques.md`

**Note**: I do not have a Write tool. The complete document content is provided above between `<document>` tags. The calling agent must write this content to the target file path.