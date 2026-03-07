# ChatGPT Codex Prompt — Hey Seven R100 Code Review

Copy everything below the line and paste into ChatGPT Codex (connect to repo `Oded-Ben-Yair/hey-seven`, branch `main`).

---

You are reviewing a production AI casino host agent built with LangGraph (Python). The repo is `hey-seven` — a 13-node StateGraph with 72 source files, 123 test files, 3674 tests, and a Next.js 15 frontend.

## Your Task

Produce **4 separate outputs** as markdown files. Read the ENTIRE codebase before writing anything.

## Architecture Overview (read these files first)

1. `src/agent/graph.py` — 13-node StateGraph assembly
2. `src/agent/nodes.py` — LLM nodes, router, Flash→Pro model routing
3. `src/agent/dispatch.py` — specialist agent dispatch (just got H9 comp-intent fix)
4. `src/agent/agents/_base.py` — shared specialist execution (1422 LOC)
5. `src/agent/profiling.py` — guest profiling enrichment node
6. `src/agent/behavior_tools/` — 4 behavior tools (comp_strategy, handoff, ltv_nudge, rapport_ladder)
7. `src/agent/guardrails.py` — 5-layer pre-LLM deterministic guardrails
8. `src/agent/compliance_gate.py` — compliance validation
9. `src/api/app.py` — FastAPI with SSE streaming
10. `tests/scenarios/` — 250 behavioral evaluation scenarios (34 YAML files)
11. `frontend/src/` — Next.js 15 chat interface

## Evaluation Framework

The agent is measured on **40 dimensions** in 4 categories:

**Behavioral (B1-B10)**: B1=Knowledge Accuracy, B2=Contextual Understanding, B3=Engagement Depth, B4=Cross-Domain Awareness, B5=Conversation Flow, B6=Persona Consistency, B7=Safety Compliance, B8=Multi-Turn Coherence, B9=Emotional Intelligence, B10=Actionability

**Profiling (P1-P10)**: P1=Natural Extraction, P2=Active Probing, P3=Give-to-Get, P4=Assumptive Bridge, P5=Progressive Sequence, P6=Incentive Framing, P7=Privacy Respect, P8=Profile Completeness, P9=Host Handoff, P10=Graceful Closure

**Host Triangle (H1-H10)**: H1=Guest Recognition, H2=Anticipatory Service, H3=Emotional Attunement, H4=Trust Building, H5=Personalization Depth, H6=Rapport Depth, H7=Revenue Generation, H8=Retention Mechanics, H9=Comp Strategy, H10=Lifetime Value

**Technical (D1-D10)**: D1=Graph Architecture, D2=RAG Pipeline, D3=Data Model, D4=API Design, D5=Testing, D6=DevOps, D7=Guardrails, D8=Scalability, D9=Docs, D10=Domain Intelligence

## Current Scores (R100 eval, GPT-5.2 judge)
- Technical: 9.63/10 (done)
- Behavioral: B-avg 5.9 (target: 7.5+)
- Profiling: P-avg 3.8 (target: 5.5+)
- Host-Triangle: H-avg 5.28 (target: 6.0+)

## Weakest Dimensions
H9=2.0 (comp routing just fixed in R100), P9=2.7 (handoff threshold just lowered), P8=3.0, P6=3.1, H6=4.4, H10=4.5, P1=4.3, P2=6.3 (above target but can improve)

---

## OUTPUT 1: Architecture & Code Quality Audit

Read every file in `src/agent/` and `src/api/`. For each module:

1. **SRP compliance**: Is the module doing one thing? Flag functions >100 LOC.
2. **Dead code**: Any functions/classes with zero callers? Trace from `graph.py` entry point.
3. **Error handling**: Any broad `except Exception` that silently swallows? Any missing error paths?
4. **State correctness**: Do all state field updates use correct reducers? Any overwrite risks?
5. **Security**: Input normalization gaps, PII leakage paths, guardrail bypasses?

Format as a prioritized finding list: CRITICAL / MAJOR / MINOR with file:line references.

## OUTPUT 2: Behavioral Quality Gap Analysis

Read ALL 34 scenario YAML files in `tests/scenarios/`. Then read the agent code that handles each scenario type. For each behavioral dimension (B1-B10):

1. **What the scenario tests** (from YAML `expected_behavioral_quality`)
2. **What the code actually does** (trace the code path for that scenario type)
3. **The gap** between expected and actual behavior
4. **Specific code change** to close the gap (file, function, what to add/modify)

Focus especially on B3 (Engagement), B9 (Emotional Intelligence), and B4 (Cross-Domain) — these have the most room for improvement.

## OUTPUT 3: Profiling & Host-Triangle Fix Plan

For each dimension scoring below 5.0 (H9, P9, P8, P6, H6, H10, P1):

1. **Root cause**: Is it a routing issue, prompt issue, missing business logic, or model capability gap?
2. **Evidence**: Quote the specific code that handles this dimension
3. **Fix specification**: Exact code change with before/after snippets
4. **Expected impact**: How many points will this fix add? (be conservative)
5. **Dependencies**: Does this fix require Flash→Pro model switch?

Prioritize by impact-per-effort ratio. A 10-line dispatch fix > a 200-line new tool.

## OUTPUT 4: Test Coverage & Scenario Gaps

1. Read `tests/conftest.py` and 5 representative test files. Identify:
   - Tests that mock LLMs vs test real behavior
   - Coverage gaps (modules with <80% coverage)
   - Missing E2E tests (full graph invocation with mocked LLMs)

2. Read all 34 scenario YAML files. Identify:
   - Dimension coverage gaps (which B/P/H dimensions have <5 scenarios?)
   - Missing edge cases (cultural sensitivity, multi-language, accessibility)
   - Scenarios that don't actually test what they claim (expected_behavior vs turns mismatch)

3. Propose 10 new scenarios that would most improve weak dimensions (P8, P9, H9, H6).

---

**IMPORTANT**: Base every finding on actual code you read. No speculation. Quote file paths and line numbers. If you can't find evidence, say "NOT FOUND" — don't fabricate.
