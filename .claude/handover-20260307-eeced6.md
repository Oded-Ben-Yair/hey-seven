# Hey Seven Session Handover

**Session**: hey-seven-session-20260307-eeced6
**Date**: 2026-03-07
**Round**: R101
**Phase**: PARADIGM SHIFT — Root Cause Investigation Complete
**Commit**: ab783a4

---

## 30-Dimension Scorecard

### Technical (D1-D10): 9.63/10 — DONE (R75)

### Behavioral (B1-B10): 5.9/10 — Target 8.0
| Dim | Name | Score | Target | Status |
|-----|------|-------|--------|--------|
| B1 | Sarcasm & Tone | 8.0 | 8.0 | MET |
| B2 | Implicit Signals | 6.1 | 8.0 | BELOW |
| B3 | Engagement | 7.0 | 8.0 | BELOW |
| B4 | Agentic Behavior | 6.4 | 8.0 | BELOW |
| B5 | Emotional Intel | 6.6 | 8.0 | BELOW |
| B6 | Tone Calibration | 6.3 | 8.0 | BELOW |
| B7 | Coherence | 7.1 | 8.0 | BELOW |
| B8 | Cultural | - | 8.0 | NOT MEASURED |
| B9 | Safety | 6.8 | 8.0 | BELOW |
| B10 | Overall Quality | 6.5 | 8.0 | BELOW |

### Profiling (P1-P10): 3.8/10 — Target 7.0
| Dim | Name | Score | Target | Status |
|-----|------|-------|--------|--------|
| P1 | Natural Extraction | 4.3 | 7.0 | BELOW |
| P2 | Active Probing | 6.3 | 7.0 | BELOW |
| P3 | Give to Get | 7.8 | 7.0 | MET |
| P4 | Assumptive Bridging | 5.4 | 7.0 | BELOW |
| P5 | Progressive Sequence | 4.8 | 7.0 | BELOW |
| P6 | Incentive Framing | 3.1 | 7.0 | BELOW |
| P7 | Privacy Awareness | 5.5 | 7.0 | BELOW |
| P8 | Profile Completeness | 3.0 | 7.0 | BELOW |
| P9 | Host Handoff | 2.7 | 7.0 | BELOW |
| P10 | Behavioral Signals | 6.4 | 7.0 | BELOW |

### Host Triangle (H1-H10): 5.28/10 — Target 6.0
| Dim | Name | Score | Target | Status |
|-----|------|-------|--------|--------|
| H1 | Guest Development | 6.7 | 6.0 | MET |
| H2 | Guest Retention | 6.1 | 6.0 | MET |
| H3 | Experience Design | 5.4 | 6.0 | BELOW |
| H4 | Relationship Depth | 6.2 | 6.0 | MET |
| H5 | Revenue Protection | 5.6 | 6.0 | BELOW |
| H6 | Rapport Depth | 4.4 | 6.0 | BELOW |
| H7 | Cross-Selling | 5.8 | 6.0 | BELOW |
| H8 | Tribal Knowledge | 6.2 | 6.0 | MET |
| H9 | Comp Strategy | 2.0 | 6.0 | BELOW (routing fixed R100, re-judge needed) |
| H10 | Lifetime Value | 4.5 | 6.0 | BELOW |

---

## THE PARADIGM SHIFT — Root Cause Found

### Phase 0 Experiments (all 3 complete)

**Experiment 1 (Evaluation Ceiling Test)**:
- 3 judges scored agent's 5 best non-crisis conversations
- GPT-5.2: avg 5.8, all CHATBOT | Grok 4: avg 8.2, 4/5 HOST | DeepSeek: avg 5.0, 4/5 PROCESSED
- Rubric CAN produce 9.2 (gold trace Conv B). Agent behavior is the ceiling.

**Experiment 2 (Gold Traces)**:
- 5 exemplar conversations created (3 at 9/10, 1 at 3/10, 1 at 6/10)
- Gold trace Conv B scored 9.2/10 by GPT-5.2
- Key insight: "3→6 is TONE, 6→9 is AGENCY" (later corrected by Oded)

**Experiment 3 (Failure Taxonomy)**:
- 62 scenarios classified into 10 failure buckets
- #1: timeout_error 32.3% | #2: generic_response 11.3% | #3 tie: wrong_mode + didnt_take_lead 11.3%
- Score drag: timeout scenarios avg 3.99 vs clean 5.42 (1.43-pt gap)
- Distribution: 32% infra + 28% fixable code + 26% model capability + 14% harsh judging

### Oded's Root Cause Correction (Human Gate Review)

**The agent's PURPOSE is wrong.** It answers questions instead of building relationships.

Oded's corrected framework:
- 3→6: TONE (sound human) — SOLVED
- **6→8: RELATIONSHIP (profiling through conversation) — THE GAP**
- 8→9: AGENCY (decide, book, confirm) — FUTURE

Every turn the agent speaks is a MISSED OPPORTUNITY to learn about the guest:
- "What kind of food do you like?"
- "What are we celebrating?"
- "Where are you visiting from?"
- "Want me to have someone from my team show you around?"

The gathered data flows to the HUMAN HOST TEAM — that's the product's value proposition.

### Oded's Specific Feedback on Agent's Best Responses:
1. Missing profiling through small talk — every turn should include a natural question
2. Generic suggestions without customization — recommends without learning preferences
3. No human host bridge — should offer "want me to have someone come meet you?"
4. Empty promises — "upgrade" said twice with no specifics or action
5. Rushing to suggestions instead of using questions to (a) customize and (b) gather data

---

## This Session's Work

### Deliverables Completed
1. **Failure taxonomy** — 62 scenarios classified (`.claude/teams/r101-paradigm-shift/failure-taxonomy.md`)
2. **Gold traces** — 5 exemplar conversations with annotations (`.claude/teams/r101-paradigm-shift/gold-traces.md`)
3. **Domain research** — 7 MCP queries, comp heuristics, 7 AI rules (`.claude/teams/r101-paradigm-shift/research-findings.md`)
4. **3-judge ceiling test** — GPT-5.2, Grok 4, DeepSeek on 5 best conversations
5. **4 bug fixes** — word-boundary comp, dead code, 13-node log, stale companion_names (committed)
6. **Process rules** — `~/.claude/rules/hey-seven-process.md` with TONE→RELATIONSHIP→AGENCY framework
7. **R102 implementation plan** — `.claude/teams/r101-paradigm-shift/r102-implementation-plan.md`
8. **MEMORY.md** updated with all findings (199 lines)
9. **Memory MCP** — 2 entities persisted (architectural_decision + session_learnings)

### Tests
- 338 passed, 0 failures (on modified files)
- Full suite: 3674 collected (pending background run)

### Key Decisions
1. TONE→RELATIONSHIP→AGENCY framework replaces TONE→AGENCY (Oded's correction)
2. Agent purpose = relationship builder, not info kiosk
3. Failure taxonomy drives fix priority, not score optimization
4. Gold traces as calibration anchors (3/6/9) and eventual few-shot examples
5. Process shift: hypothesis-first, human judgment in loop, MCP research before code

### Learnings (persisted to Memory MCP + rules files)
- Hypothesis-first investigation found root cause in 1 session vs 25 rounds of fix-code-re-eval
- Human gates are CRITICAL — Oded corrected diagnosis from AGENCY to RELATIONSHIP
- Grok inflates scores 2.4 pts above GPT-5.2 — always use GPT-5.2 as primary judge
- code-worker teammate can go idle without output — check after 2nd idle, do work in lead
- Domain research via MCPs before code prevents building the wrong thing
- Substring matching for keywords = false positives (use set intersection)

---

## Blueprint Next Session: R102

### Phase: RELATIONSHIP-FIRST REWRITE

### Priority Order (from R102 implementation plan)

1. **Infrastructure fixes** (30 min) — timeout 60s→90s, fix canned "Glad I could help" closer, fix crisis exit conditions, fix Spanish crisis responses

2. **Rewrite agent purpose** (2 hours, CORE CHANGE) — System prompt for ALL specialists must encode:
   - Job 1: Address the immediate need
   - Job 2: Ask 1 profiling question naturally woven in
   - Job 3: Customize using gathered info
   - Job 4: Offer human host bridge when appropriate
   - Include profiling question bank by context (first contact, group, dining, after loss, returning)
   - Include human host bridge phrases

3. **Update gold traces** (1 hour) — Add profiling questions to existing gold trace conversations

4. **Update few-shot examples** (1 hour) — Current 27 focus on TONE. Need RELATIONSHIP examples showing curiosity

5. **Create longer eval scenarios** (1 hour) — 10 new 5-7 turn scenarios testing relationship building (does agent ask? does it USE gathered info? does it offer human host bridge?)

6. **Re-eval** (1 hour) — 30 targeted scenarios at 90s timeout. Target B-avg 7.0+ on clean scenarios

### Success Metrics (Product Outcomes)
- 80%+ of turns include a natural profiling question
- 50%+ of turn-2+ responses reference info from previous turns
- 20%+ of conversations include human host bridge offer
- B-avg 7.0+ on clean (non-timeout) scenarios
- Oded reads 5 new transcripts and says "host" for 3+

### Key Files for Next Session
- `~/.claude/teams/r101-paradigm-shift/failure-taxonomy.md` — what to fix
- `~/.claude/teams/r101-paradigm-shift/gold-traces.md` — what 9/10 looks like
- `~/.claude/teams/r101-paradigm-shift/research-findings.md` — domain grounding
- `~/.claude/teams/r101-paradigm-shift/r102-implementation-plan.md` — detailed plan
- `~/.claude/rules/hey-seven-process.md` — process rules (MANDATORY)
- `src/agent/prompts.py` — system prompts to rewrite
- `src/agent/agents/_base.py` — specialist execution logic
- `src/agent/nodes.py` — greeting/farewell node (canned closer fix)

---

## Optimal Execution Strategy

### Recommended Mode: SOLO (Lead + subagents for research)

The core work is system prompt rewriting + few-shot example design. This needs a single coherent vision, not parallel implementation. Use subagents only for:
- Research: `research-specialist` for additional domain grounding
- Review: `code-judge` to hostile-review the prompt changes

### Key Tools for Next Session

#### MCP Tools
- `azure_chat` (GPT-5.2) — judge scoring after changes
- `grok_chat` — write realistic guest dialogue for few-shots
- `azure_brainstorm` — design profiling question bank
- `perplexity_research` — additional casino host psychology if needed

#### Skills
- `/honest-answer` — "Did this session's changes actually improve relationship building?"
- `/ship-it` — if scope creep threatens delivery (DON'T rewrite everything, focus on purpose)
- `/pre-mortem` — before touching system prompts (affects all specialists)

#### Process (MANDATORY — from hey-seven-process.md)
- Read 5 transcripts before any behavioral code change
- State hypothesis before coding
- Run at least 1 MCP research query for domain grounding
- Include Oded in eval loop (binary host/chatbot verdict)

---

## Quick Resume Command

```
/go
```

### What /go Should Auto-Plan (for the receiving agent):

1. Read this handover + MEMORY.md + `hey-seven-process.md`
2. Read `r102-implementation-plan.md` for detailed steps
3. Read `gold-traces.md` to understand what 9/10 looks like
4. Read `research-findings.md` for domain grounding (especially the 7 AI rules)
5. Enter plan mode and create the implementation plan:
   - Step 1: Infrastructure fixes (timeout, canned closer, crisis exit, Spanish)
   - Step 2: Rewrite specialist system prompts with relationship-first purpose
   - Step 3: Design profiling question bank + human host bridge phrases
   - Step 4: Update gold traces and few-shot examples
   - Step 5: Create 10 longer eval scenarios (5-7 turns)
   - Step 6: Run 30-scenario eval at 90s timeout
   - Step 7: Present 5 transcripts to Oded for binary host/chatbot verdict
6. Execute the plan using the process rules (hypothesis-first, human gates, MCP research)

### CRITICAL CONTEXT FOR NEXT AGENT:
- The agent's PURPOSE must change from "answer questions" to "build relationships"
- Every specialist turn needs: answer + 1 profiling question + customization + optional host bridge
- The profiling questions must feel NATURAL ("What are we celebrating?") not CLINICAL ("Please provide occasion")
- Read Oded's specific feedback in `memory/r101-phase0-findings.md` under "Oded's Root Cause Diagnosis"
- The failure taxonomy shows 32% timeout — fix infra FIRST before measuring behavioral quality
- Gold traces define the target but need updating to include profiling questions (Oded's feedback)
- Process rules in `hey-seven-process.md` are MANDATORY — no behavioral code without failure taxonomy
