# Production Rebrand + 10-Round Hostile Review — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform hey-seven from interview assignment to production MVP, then run 10 rounds of hostile review using real LLMs (Gemini 3 Pro, GPT-5.2, Grok 4, DeepSeek) to reach 95+ production-readiness score.

**Architecture:** Two phases — (1) Production rebrand: delete interview files, rewrite CLAUDE.md/README.md, fix .gitignore, clean references. (2) 10 sequential hostile review rounds, each using an agent team with 3 real LLM reviewers + 1 fixer.

**Tech Stack:** Claude Opus 4.6 (orchestrator), Gemini 3 Pro via `gemini-query`, GPT-5.2 via `azure_chat`, Grok 4 via `grok_reason`, DeepSeek via `azure_deepseek_reason`.

---

## Phase 1: Production Rebrand

### Task 1: Delete interview-era files

**Files:**
- Delete: `assignment/` (entire directory)
- Delete: `reviews/` (entire directory)
- Delete: `boilerplate/` (entire directory)
- Delete: `research/personas/` (entire directory)
- Delete: `Take-Home Assignment — Senior AI_Backend Engineer.pdf`
- Delete: `former session hey seven.txt`
- Delete: `deliverables/` (empty directory)

**Step 1: Verify files exist and list contents**

Run: `ls -la assignment/ reviews/ boilerplate/ research/personas/ deliverables/ && ls "Take-Home Assignment"* && ls "former session"*`
Expected: File listing of all targets.

**Step 2: Delete files**

```bash
rm -rf assignment/ reviews/ boilerplate/ research/personas/ deliverables/
rm "Take-Home Assignment — Senior AI_Backend Engineer.pdf"
rm "former session hey seven.txt"
```

**Step 3: Verify deletion**

Run: `ls assignment/ reviews/ boilerplate/ research/personas/ deliverables/ 2>&1`
Expected: "No such file or directory" for each.

**Step 4: Run tests to verify nothing broke**

Run: `cd /home/odedbe/projects/hey-seven && python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5`
Expected: All tests pass (1070+). No imports reference deleted files.

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: delete interview-era files (assignment, reviews, boilerplate, personas, deliverables)"
```

---

### Task 2: Fix .gitignore for production repo

**Files:**
- Modify: `.gitignore`

**Step 1: Read current .gitignore**

The current `.gitignore` has a section (lines 60-72) that hides production-relevant files:
```
# Internal prep work (not part of submission)
.claude/
CLAUDE.md
research/
boilerplate/
knowledge-base/
assignment/
reviews/
deliverables/
docs/
former session hey seven.txt
Take-Home Assignment*.pdf
```

**Step 2: Replace the interview-era section**

Remove the entire "Internal prep work" section. Replace with production gitignore:

```gitignore
# Claude Code config (dev only)
.claude/
CLAUDE.md
```

Keep `research/` and `knowledge-base/` tracked now — they have production value. `docs/` should be tracked. Deleted files (`assignment/`, `reviews/`, `boilerplate/`, etc.) don't need gitignore entries anymore.

**Step 3: Verify docs/ can be committed**

Run: `git add docs/plans/2026-02-20-production-review-rounds-design.md && git status`
Expected: File staged successfully (no longer ignored).

**Step 4: Commit**

```bash
git add .gitignore docs/
git commit -m "chore: update .gitignore for production repo — track docs, research, knowledge-base"
```

---

### Task 3: Rewrite CLAUDE.md for production

**Files:**
- Rewrite: `CLAUDE.md`

**Step 1: Write new CLAUDE.md**

Replace the entire file. Key changes:
- Title: "Hey Seven — Production AI Casino Host Agent"
- Remove: "Interview Assignment Infrastructure", "Oded's Strengths to Highlight", "Key People" interview context, "Assignment Reception Protocol", "Phase 3/4/5" interview phases, "Boilerplate Code Map", interview team templates
- Keep: Hard Rules (reworded), Tech Stack, Company Context (reworded), Directory Structure (updated), Review Round Protocol (updated for production), Known Limitations (updated)
- Add: Production priorities, deployment status, current architecture summary

New CLAUDE.md structure:
```
# Hey Seven — Production AI Casino Host Agent

## Project Identity
Production MVP for Hey Seven (heyseven.ai). AI agent handling digital casino host tasks 24/7.

## Hard Rules (Project-Specific)
1. ISOLATION — never reference other projects
2. GitHub ONLY — never push to Azure DevOps
3. LangGraph ONLY
4. GCP deployment target
5. NO mock data
6. API keys from Azure Key Vault (dev) / GCP service accounts (prod)
7. QUALITY BAR — production-grade, no shortcuts

## Current State
- 23K LOC, 1070+ tests, 32 test files
- 11-node LangGraph StateGraph v2.2
- 92/100 review score (20 rounds of hostile review)
- Tagged v1.0.0
- Live demo: https://hey-seven-180574405300.us-central1.run.app

## Tech Stack Decisions
[Keep existing table]

## Company Context
[Keep but remove interview references — just company/product/market]

## Directory Structure
[Updated — remove deleted dirs]

## Production Review Protocol
[Updated from interview review protocol]

## Known Limitations
[Updated — remove "waiting for assignment" language]
```

**Step 2: Verify no interview language remains**

Run: `grep -i "interview\|assignment\|hired\|hiring\|candidate\|career\|strengths to highlight\|wow factor" CLAUDE.md`
Expected: Zero matches.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "chore: rewrite CLAUDE.md for production — strip all interview language"
```

---

### Task 4: Update README.md for production

**Files:**
- Modify: `README.md`

**Step 1: Review and update README.md**

The README is already fairly production-oriented. Changes needed:
- Line 19: "What I Built & Why" → "Architecture Overview" (remove first-person interview framing)
- Line 290: "Built by Oded Ben-Yair | February 2026" → remove or update
- Line 131: Update test count if changed
- Remove any "Trade-offs I'd Revisit" framing that implies interview submission
- "Trade-offs I'd Revisit" → "Production Backlog" or merge with existing "Production Backlog" section

**Step 2: Verify no interview language**

Run: `grep -i "interview\|assignment\|submission\|I built\|I chose\|I'd revisit" README.md`
Expected: Zero matches (rewrite all first-person interview-framed language).

**Step 3: Commit**

```bash
git add README.md
git commit -m "chore: update README.md — remove interview framing, production language"
```

---

### Task 5: Fix targeted references in ARCHITECTURE.md and middleware.py

**Files:**
- Modify: `ARCHITECTURE.md:813`
- Modify: `src/api/middleware.py:180`

**Step 1: Fix ARCHITECTURE.md**

Line 813: Change:
```
The following features from the initial architecture specification (`assignment/architecture.md`) were consciously deferred.
```
To:
```
The following features were consciously deferred from the initial architecture specification.
```

**Step 2: Fix middleware.py**

Line 180: Change:
```python
    # interview demo where the frontend is secondary to the agent architecture.
```
To:
```python
    # current deployment where the frontend is a single-file static asset.
```

**Step 3: Final grep for any remaining interview language across entire codebase**

Run: `grep -ri "interview\|assignment\|hired\|hiring\|candidate\|career\|wow factor\|strengths to highlight" src/ tests/ *.md *.yaml *.yml data/ knowledge-base/ 2>/dev/null | grep -v "node_modules\|.git\|__pycache__"`
Expected: Zero matches (excluding false positives like "candidate" in non-interview context).

**Step 4: Run tests**

Run: `python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add ARCHITECTURE.md src/api/middleware.py
git commit -m "chore: remove last interview references from ARCHITECTURE.md and middleware.py"
```

---

### Task 6: Push rebrand and tag v2.0.0-alpha

**Step 1: Push to GitHub**

Run: `git push origin main`

**Step 2: Verify clean repo**

Run: `git status && grep -ri "interview\|assignment" src/ tests/ *.md 2>/dev/null | head -20`
Expected: Clean working tree, zero interview language.

**Step 3: Do NOT tag yet — tag v2.0.0 after review rounds complete**

---

## Phase 2: 10 Hostile Review Rounds

### Prerequisite: Reviewer Prompt Template

Before starting rounds, create a reusable reviewer prompt template at `docs/review-prompt-template.md`. This template is sent (with modifications) to each real LLM reviewer via MCP.

**Template contents:**

```markdown
# Hey Seven Production Code Review — Round {N}

## Repo
GitHub: https://github.com/Oded-Ben-Yair/hey-seven
Commit: {commit_hash}

## File Tree
{file_tree}

## Source Files to Review
{file_contents — key files only, ~50 files}

## Previous Round Scores
{previous_scores_table}

## Spotlight Focus: {spotlight_area}
This round has extra focus on {spotlight_description}. Flag issues in this area as +1 severity.

## Scoring Rubric (10 Dimensions, 0-10 each)

1. **Graph/Agent Architecture** (0-10): StateGraph structure, specialist dispatch, validation loop, state management, node wiring, conditional edges.
2. **RAG Pipeline** (0-10): Chunking strategy, retrieval quality, reranking, idempotent ingestion, multi-tenant safety, embedding model pinning.
3. **Data Model / State Design** (0-10): TypedDict fields, reducers, serialization safety, guest profiles, confidence decay, CCPA compliance.
4. **API Design** (0-10): Middleware correctness, SSE streaming, error taxonomy, auth, rate limiting, PII redaction.
5. **Testing Strategy** (0-10): Coverage (baseline 1070+), test quality, edge cases, integration tests, deterministic evals.
6. **Docker & DevOps** (0-10): Dockerfile security, CI/CD pipeline, health checks, Cloud Run config, dependency management.
7. **Prompts & Guardrails** (0-10): System prompt quality, deterministic guardrails (73+ patterns), injection defense, multilingual coverage.
8. **Scalability & Production** (0-10): Circuit breakers, caching (TTL), async patterns, error recovery, graceful degradation.
9. **Documentation & Code Quality** (0-10): README accuracy, inline docs, naming conventions, patterns consistency, dead code.
10. **Domain Intelligence** (0-10): Casino operations accuracy, regulatory compliance (TCPA, BSA/AML), SMS handling, comp system, guest profiles.

## Instructions

You are a hostile production code reviewer. Your job is to find real problems that would break in production. Do NOT be nice.

1. Score each dimension 0-10 with 1-sentence justification.
2. List findings in this format:
   ```
   ### Finding N (SEVERITY): Title
   - **Location**: `file.py:line`
   - **Problem**: What's wrong
   - **Impact**: What breaks in production
   - **Fix**: Specific code change
   ```
3. SEVERITY levels: CRITICAL (blocks production), HIGH (should fix before launch), MEDIUM (improvement), LOW (style/polish)
4. Minimum 5 findings per review. Do not rubber-stamp.
5. If the spotlight area has issues, those findings get +1 severity bump.
```

---

### Task 7: Create reviewer prompt template

**Files:**
- Create: `docs/review-prompt-template.md`

**Step 1: Write the template file**

Write the template above to `docs/review-prompt-template.md`.

**Step 2: Commit**

```bash
git add docs/review-prompt-template.md
git commit -m "docs: add hostile review prompt template for 10-round production review"
```

---

### Task 8-17: Review Rounds 1-10

Each round follows the SAME protocol. Variables per round: `{N}`, `{spotlight}`, `{models}`.

#### Round Protocol (repeat for each round)

**Step 1: Create team**

```
TeamCreate: "prod-review-r{N}"
```

**Step 2: Create tasks**

```
TaskCreate: "Gemini review round {N}" → assigned to gemini-reviewer
TaskCreate: "GPT-5.2 review round {N}" → assigned to gpt-reviewer
TaskCreate: "Grok/DeepSeek review round {N}" → assigned to grok-reviewer (or deepseek-reviewer for R5/R10)
TaskCreate: "Apply consensus fixes round {N}" → assigned to fixer, blocked by above 3
```

**Step 3: Spawn 4 teammates**

Each reviewer teammate:
1. Reads all source files in `src/` (key files: `graph.py`, `nodes.py`, `state.py`, `guardrails.py`, `compliance_gate.py`, `middleware.py`, `pipeline.py`, `config.py`, `app.py`)
2. Reads `README.md`, `ARCHITECTURE.md`, `Dockerfile`, `cloudbuild.yaml`, `docker-compose.yml`
3. Reads `tests/` directory listing + key test files
4. Constructs the review prompt from template
5. Calls their MCP tool with the full review prompt
6. Writes findings to `reviews/prod-r{N}/{model}-review.md`
7. Sends summary message to fixer

**Reviewer teammate prompts:**

For `gemini-reviewer`:
```
You are a hostile production code reviewer for a LangGraph casino host agent.

1. Read ALL files in src/ and tests/ (use Glob + Read)
2. Read README.md, ARCHITECTURE.md, Dockerfile, cloudbuild.yaml
3. Build the review prompt using docs/review-prompt-template.md
4. Call gemini-query with thinking_level="high" and the full review prompt + file contents
5. Parse the response and write to reviews/prod-r{N}/gemini-review.md
6. Send a 2-line summary to the fixer teammate

Round {N} spotlight: {spotlight}

OUTPUT: Write full review to reviews/prod-r{N}/gemini-review.md
```

For `gpt-reviewer`:
```
Same as above but use azure_chat with model="gpt-5.2"
Write to reviews/prod-r{N}/gpt-review.md
```

For `grok-reviewer` (R1-4, R6-9):
```
Same as above but use grok_reason
Write to reviews/prod-r{N}/grok-review.md
```

For `deepseek-reviewer` (R5, R10):
```
Same as above but use azure_deepseek_reason
Write to reviews/prod-r{N}/deepseek-review.md
```

**Fixer teammate prompt:**
```
You are a production code fixer. Your job is to apply consensus fixes from 3 hostile reviews.

1. Wait for all 3 review files to appear in reviews/prod-r{N}/
2. Read all 3 review files
3. Identify consensus findings (flagged by 2/3+ reviewers)
4. Fix CRITICAL findings first, then HIGH, then MEDIUM
5. Run pytest after each fix — must pass
6. Write reviews/prod-r{N}/summary.md with:
   - Score table (3 models × 10 dimensions)
   - Average score per dimension and total
   - Consensus findings fixed (list)
   - Single-model findings logged but not fixed (list)
   - Remaining LOW items (list)
7. Commit all changes with message "fix: prod-r{N} — {summary of fixes}"
8. Push to GitHub

OUTPUT: Write summary to reviews/prod-r{N}/summary.md
```

**Step 4: Main lead reads ONLY summary**

Read: `reviews/prod-r{N}/summary.md`
Report scores and trajectory to user.

**Step 5: Shut down team**

Send shutdown_request to all 4 teammates. Delete team.

---

### Round Variables

| Round | N | Spotlight | Third Reviewer | Third MCP Tool |
|-------|---|-----------|----------------|----------------|
| R1 | 1 | Production rebrand completeness | grok-reviewer | grok_reason |
| R2 | 2 | Security hardening | grok-reviewer | grok_reason |
| R3 | 3 | Error handling & resilience | grok-reviewer | grok_reason |
| R4 | 4 | Testing gaps | grok-reviewer | grok_reason |
| R5 | 5 | Scalability | deepseek-reviewer | azure_deepseek_reason |
| R6 | 6 | API contract & documentation | grok-reviewer | grok_reason |
| R7 | 7 | RAG & domain data quality | grok-reviewer | grok_reason |
| R8 | 8 | Deployment readiness | grok-reviewer | grok_reason |
| R9 | 9 | Code simplification | grok-reviewer | grok_reason |
| R10 | 10 | Final adversarial | deepseek-reviewer | azure_deepseek_reason |

---

## Phase 3: Final Validation

### Task 18: Post-round-10 validation and tagging

**Step 1: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short 2>&1 | tail -20`
Expected: All tests pass (1070+ baseline, likely higher after fix rounds).

**Step 2: Docker build**

Run: `docker build -t hey-seven:v2.0.0 . 2>&1 | tail -10`
Expected: Build succeeds.

**Step 3: Final interview language check**

Run: `grep -ri "interview\|assignment\|hired\|hiring\|candidate\|career\|wow factor" src/ tests/ *.md data/ knowledge-base/ 2>/dev/null`
Expected: Zero matches.

**Step 4: Verify README accuracy**

Read README.md. Verify test counts, file counts, and architecture description match current reality.

**Step 5: Score trajectory check**

Read all 10 summary files. Verify scores trend toward 95+.

**Step 6: Tag v2.0.0**

```bash
git tag -a v2.0.0 -m "Production MVP — 10 rounds of hostile review, 95+ score"
git push origin v2.0.0
```

---

## MCP Tool Quick Reference (for reviewer teammates)

| Tool | Load First? | Call Example |
|------|-------------|--------------|
| `gemini-query` | Yes: `ToolSearch("select:mcp__gemini__gemini-query")` | `mcp__gemini__gemini-query(prompt=review_text, thinking_level="high")` |
| `azure_chat` | No (always active) | `mcp__azure-ai-foundry__azure_chat(prompt=review_text, model="gpt-5.2")` |
| `grok_reason` | No (always active) | `mcp__grok__grok_reason(prompt=review_text)` |
| `azure_deepseek_reason` | No (always active) | `mcp__azure-ai-foundry__azure_deepseek_reason(prompt=review_text)` |

**Important**: Gemini is lazy-loaded — must call `ToolSearch` first. Azure and Grok are always active.

---

## Context Budget Per Round

| Component | Est. Tokens |
|-----------|-------------|
| Team creation + task setup | ~2K |
| 3 reviewer teammates (each reads files + calls MCP) | ~15K each = ~45K |
| Fixer reads 3 reviews + applies fixes | ~20K |
| Summary read by lead | ~1K |
| Team shutdown | ~1K |
| **Total per round** | **~70K** |
| **10 rounds** | **~700K** (spread across teams, not main context) |

Main lead context: ~10K per round (task setup + summary read + shutdown). 10 rounds = ~100K main context. Well within limits.
