# ChatGPT 5.4 Pro Deep Research Prompt — Hey Seven Strategic Analysis

Copy everything below the line and paste into ChatGPT 5.4 Pro (deep research mode). Attach the GitHub repo URL: `https://github.com/Oded-Ben-Yair/hey-seven`

---

## Context

Hey Seven is a production AI casino host agent built with LangGraph (Python) + Next.js 15 frontend. It's an Israeli seed-stage startup building "The Autonomous Casino Host That Never Sleeps" for US casinos.

The codebase has gone through 100 rounds of iterative development reaching 9.63/10 technical quality. But **behavioral quality** — how well the agent actually performs as a casino host — is lagging:
- Behavioral: 5.9/10 (target: 7.5)
- Profiling: 3.8/10 (target: 5.5)
- Host-Triangle: 5.28/10 (target: 6.0)

## Deep Research Request

I need you to perform deep research across THREE domains and synthesize into a strategic improvement plan. This is NOT a code review — it's a strategic analysis of how to make an AI casino host world-class.

### Research Domain 1: Production AI Casino Hosts — State of the Art

Research and analyze:
1. **What do real casino hosts actually do?** Interview transcripts, casino industry publications, gaming conference talks. What are the top 5 things a human casino host does that creates guest loyalty?
2. **Existing AI casino host products**: Any competitors? What's their approach? (check: Konami SYNKROS, IGT Advantage, Light & Wonder, any AI startups in gaming)
3. **Casino CRM best practices**: What data points do casino hosts track? How do they decide comp offers? What's the "whale" identification process?
4. **Regulatory constraints**: State-by-state responsible gaming requirements for AI interactions. TCPA for SMS. Self-exclusion program integration requirements.
5. **Guest psychology in casinos**: Loss aversion recovery, celebration amplification, the "golden hour" after a big win, tilt detection in gaming.

### Research Domain 2: Conversational AI Quality — Beyond Chatbots

Research and analyze:
1. **What makes a 9/10 AI conversation vs a 6/10?** Academic papers on conversational quality metrics. The difference between "helpful" and "memorable."
2. **Profiling without interrogation**: How do luxury hotel concierges (Ritz-Carlton, Four Seasons) learn guest preferences naturally? What conversational techniques do they use?
3. **Emotional intelligence in AI**: State of the art in detecting and responding to emotional states. Papers on empathy in dialogue systems. The gap between "I understand your frustration" (hollow) and genuine emotional attunement.
4. **Multi-turn coherence**: Research on maintaining context, callback references, progressive disclosure. What makes a conversation feel like it's with someone who *remembers*?
5. **The "host triangle"**: Revenue generation + Guest satisfaction + Relationship depth. How do the best AI systems balance commercial goals with genuine service?

### Research Domain 3: LangGraph Production Patterns for Behavioral Quality

Research and analyze:
1. **Prompt engineering for behavioral consistency**: Few-shot vs system prompt vs retrieval-augmented prompting. What works at scale for consistent persona?
2. **Structured output for conversation steering**: How to use Pydantic schemas + LangGraph state to guide conversation flow without making it feel scripted?
3. **Validation loops for quality**: Generate→validate→retry patterns. What validation criteria actually correlate with human quality ratings?
4. **Flash vs Pro model routing**: When is the 4x cost of a Pro model justified? What conversation signals indicate complexity that Flash can't handle?
5. **Fine-tuning for behavioral quality**: Gemini fine-tuning options, RLHF approaches, distillation from Pro→Flash. What's the realistic path from prompt engineering to fine-tuned models?

## Desired Output Format

### Part 1: Industry Intelligence Brief (2-3 pages)
What real casino hosts do that our AI doesn't. Concrete behavioral patterns we're missing. Competitive landscape.

### Part 2: Behavioral Quality Playbook (3-4 pages)
For each weak dimension (P-avg 3.8, H-avg 5.28), provide:
- **What "9/10" looks like** for this dimension (with example dialogue)
- **Why our agent scores 3-5** (informed by reading the codebase)
- **The fix**: A specific, implementable change (not "improve prompts" — actual prompt text, code pattern, or architectural change)
- **Research backing**: Citation or example from your research

### Part 3: Strategic Roadmap (1-2 pages)
Priority-ordered plan to go from current scores to target:
1. Quick wins (1-session fixes, +1-2 points each)
2. Medium-term (2-3 sessions, architectural changes)
3. Long-term (fine-tuning, data collection, model distillation)

Include specific metrics to track progress and "definition of done" for each dimension.

---

**Key principle**: The path from 6/10 to 9/10 is NOT more code or more guardrails. It's making the agent behave like a host who genuinely cares, knows the property inside-out, and makes every guest feel like a VIP. The code is excellent (9.63 technical). The behavior needs to match.
