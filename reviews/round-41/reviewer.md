# R41 Deep Review: D6 Docker & DevOps + D1 Graph Architecture + D9 Trade-off Docs

**Date**: 2026-02-23
**Reviewer**: Claude Opus 4.6 (deep review)
**Commit**: 4f4ae2d
**Cross-validated with**: GPT-5.2 Codex (azure_code_review, security focus), Gemini 3 Pro (gemini-query, thinking=high)
**Focus**: D6 (50% effort, weakest at 7.0), D1 (30% effort, highest weight 0.20), D9 (20% effort)

---

## D6: Docker & DevOps (Current: 7.0, Weight: 0.10)

### D6-C001: --require-hashes NOT enforced (CRITICAL, carried from R37)

**File**: `Dockerfile:20`, `requirements-prod.txt`
**Evidence**: Dockerfile line 20 says `pip install --no-cache-dir --target=/build/deps -r requirements-prod.txt` WITHOUT `--require-hashes`. The comment on line 15-19 references TODO HEYSEVEN-58 but requirements-prod.txt contains zero hash entries (checked: no `--hash=sha256:` lines in the file). This has been flagged since R37 and carried through R38, R39, R40 with zero progress.

**Impact**: Without hash-pinned requirements, a dependency confusion attack or compromised PyPI mirror can inject malicious packages into the Docker build. The Trivy scan (Step 3) catches known CVEs but NOT tampered packages with valid names. GPT-5.2 Codex confirmed this as the highest-priority security fix.

**Fix**:
1. Generate hashed requirements: `pip-compile --generate-hashes --output-file=requirements-prod.txt requirements-prod.in`
2. Change Dockerfile line 20 to: `RUN pip install --no-cache-dir --require-hashes --target=/build/deps -r requirements-prod.txt`

**Score impact**: +1.0 if fixed (addresses the single most-cited D6 gap across R37-R40).

---

### D6-M001: No SBOM generation in CI pipeline (MAJOR)

**File**: `cloudbuild.yaml` (missing step)
**Evidence**: Pipeline has Trivy vulnerability scan (Step 3) but no Software Bill of Materials (SBOM) generation. SBOM is a software supply chain best practice (NIST SP 800-218, Executive Order 14028) and is increasingly required for enterprise casino clients.

**Fix**: Add after Step 3:
```yaml
# Step 3b: Generate SBOM (CycloneDX format)
- name: 'aquasec/trivy:0.58.2'
  timeout: '120s'
  args:
    - 'image'
    - '--format=cyclonedx'
    - '--output=/workspace/sbom.json'
    - 'us-central1-docker.pkg.dev/$PROJECT_ID/hey-seven/hey-seven:$COMMIT_SHA'
```

**Score impact**: +0.25

---

### D6-M002: No image signing (cosign) (MAJOR)

**File**: `cloudbuild.yaml` (missing step)
**Evidence**: Images are pushed to Artifact Registry without cryptographic signing. Any actor with Artifact Registry write access can push a modified image. Cosign signing provides tamper-evidence.

**Fix**: Add after Step 4 (push):
```yaml
# Step 4b: Sign image with cosign
- name: 'gcr.io/cloud-builders/docker'
  timeout: '120s'
  entrypoint: 'bash'
  args:
    - '-c'
    - |
      cosign sign --yes us-central1-docker.pkg.dev/$PROJECT_ID/hey-seven/hey-seven:$COMMIT_SHA
```
(Requires cosign key setup in GCP KMS)

**Score impact**: +0.25

---

### D6-M003: curl installed for health check when Python is available (MAJOR)

**File**: `Dockerfile:27`
**Evidence**: The production stage installs `curl` (via `apt-get`) solely for the `HEALTHCHECK` instruction. Since Cloud Run ignores Dockerfile HEALTHCHECK (as documented on line 67-68), curl is only used for local Docker health monitoring. Installing an extra binary increases attack surface.

**Fix**: Replace curl-based HEALTHCHECK with Python:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1
```
Then remove `curl` from `apt-get install`. GPT-5.2 Codex confirmed this recommendation.

**Score impact**: +0.25

---

### D6-M004: .dockerignore missing CLAUDE.md and .claude/ (MAJOR)

**File**: `.dockerignore`
**Evidence**: `.dockerignore` excludes `*.md` (line 29) which catches `CLAUDE.md`, but `.claude/` directory is NOT explicitly excluded. If `.claude/` contains session data, status.json, or decisions.log, these are copied into the Docker image. Also `.hypothesis/` (from property-based tests) is not excluded.

**Fix**: Add to `.dockerignore`:
```
.claude/
.hypothesis/
```

**Score impact**: +0.15

---

### D6-m001: No Cloud Build failure notification (MINOR)

**File**: `cloudbuild.yaml`
**Evidence**: Pipeline has no failure notification mechanism. When the smoke test (Step 7) fails and auto-rollback occurs, no alert is sent. The runbook references Slack #hey-seven-alerts but no Cloud Build notification is configured.

**Fix**: Add to cloudbuild.yaml options:
```yaml
options:
  logging: CLOUD_LOGGING_ONLY
  # Add Pub/Sub notification for failure alerting
```
Or configure Cloud Build trigger with failure notification to Slack via Cloud Functions.

---

### D6-m002: Step 1 installs full dev requirements (MINOR)

**File**: `cloudbuild.yaml:22`
**Evidence**: Step 1 installs `requirements-dev.txt` which includes chromadb (~200MB), pytest, ruff, mypy, and all dev dependencies. This is correct (needed for testing in CI) but adds ~3-5 minutes to the pipeline. Consider using a pre-built builder image with dev deps cached.

---

### D6 Score Assessment

| Finding | Severity | Fixable? | Score Impact |
|---------|----------|----------|-------------|
| D6-C001 | CRITICAL | Yes (generate hashes) | +1.0 |
| D6-M001 | MAJOR | Yes (SBOM step) | +0.25 |
| D6-M002 | MAJOR | Yes (cosign step) | +0.25 |
| D6-M003 | MAJOR | Yes (Python healthcheck) | +0.25 |
| D6-M004 | MAJOR | Yes (.dockerignore) | +0.15 |
| D6-m001 | MINOR | Low priority | +0.05 |
| D6-m002 | MINOR | Low priority | +0.05 |

**Current D6 score: 7.0** | **Potential after fixes: 8.5-9.0** (the +1.0 from --require-hashes alone would be the single highest-impact change)

---

## D1: Graph Architecture (Current: 8.5, Weight: 0.20)

### D1-M001: retrieved_context not cleared before END (MAJOR)

**File**: `src/agent/graph.py` (missing cleanup), `src/agent/state.py:129`
**Evidence**: `retrieved_context` (a list of `RetrievedChunk` dicts with content, metadata, score) is populated by `retrieve_node` but never cleared before terminal nodes (respond, fallback, greeting, off_topic). With `FirestoreSaver` (production), the FULL list of retrieved chunks is serialized and written to Firestore on every checkpoint.

Typical `retrieved_context` size: 5 chunks * ~500 chars each = ~2.5KB per turn. Over a 20-turn conversation, this accumulates in the Firestore checkpoint document (~50KB of stale context data). While `_initial_state()` resets it per-turn, the checkpoint still serializes the full state INCLUDING retrieved_context from the current turn.

Gemini 3 Pro flagged this as a "production bottleneck for the Checkpointer" when scaling.

**Fix**: In `respond_node` or `persona_envelope_node`, add `"retrieved_context": []` to the return dict to clear it before the final checkpoint write. Same for `fallback_node`, `greeting_node`, `off_topic_node`.

**Score impact**: +0.25

---

### D1-M002: generate node name masks specialist dispatch observability (MAJOR)

**File**: `src/agent/graph.py:452`
**Evidence**: The `generate` node name is `graph.add_node(NODE_GENERATE, _dispatch_to_specialist)`. In LangSmith/LangFuse traces, the entire specialist dispatch + execution appears as a single `generate` node taking 5-15 seconds. There's no visibility into WHICH specialist was dispatched or how long the dispatch LLM call took vs. the specialist execution.

The SSE streaming explicitly filters by `langgraph_node == NODE_GENERATE` (graph.py:742), so renaming the node would break SSE compatibility. However, the specialist name IS logged (line 314-317) but not emitted in SSE graph_node events or structured traces.

Gemini 3 Pro flagged this as "defeating the primary purpose of LangGraph" for observability. While this is overstated (logging IS present), the lack of specialist name in SSE `graph_node` events and structured traces IS a gap.

**Fix**: Add specialist name to the node metadata extraction in `_extract_node_metadata()`:
```python
if node == NODE_GENERATE:
    return {"specialist": output.get("specialist_name")}
```

**Score impact**: +0.15

---

### D1-M003: ARCHITECTURE.md guardrail pattern count stale (MAJOR, cross-ref D9)

**File**: `ARCHITECTURE.md:104`, `src/agent/compliance_gate.py:8`
**Evidence**: ARCHITECTURE.md says "84 total patterns" (line 104). compliance_gate.py docstring says "~185 compiled regex patterns across 11 languages" (line 8). Actual count from `grep -c 'compile(' guardrails.py` = 185. The ARCHITECTURE.md count is stale from pre-R35 when Hindi/Tagalog patterns were added.

**Fix**: Update ARCHITECTURE.md line 104 to "185 total patterns across 11 languages".

**Score impact**: +0.10 (D9 score, not D1)

---

### D1-m001: Compliance gate handles intent routing (greeting, off_topic) in addition to safety (MINOR)

**File**: `src/agent/compliance_gate.py:105-108`
**Evidence**: The compliance gate routes empty messages to `greeting` and determines `off_topic` from guardrail triggers. Gemini 3 Pro flagged this as an SRP concern -- compliance should strictly pass/block, not classify intent. However, this is a DELIBERATE design decision (documented in `route_from_compliance` docstring, graph.py:378-398) for defense-in-depth. The compliance gate catches safety-critical queries BEFORE any LLM call; the router adds nuanced classification on top. Removing intent from compliance_gate would require an extra LLM call for every empty/greeting message.

**Assessment**: This is a documented trade-off, not a bug. No fix needed. The existing documentation in `route_from_compliance` is thorough.

---

### D1-m002: _dispatch_to_specialist is 195 lines (MINOR)

**File**: `src/agent/graph.py:176-370`
**Evidence**: The `_dispatch_to_specialist` function is ~195 lines. It handles: retry-reuse, LLM dispatch, keyword fallback, feature flags, guest profile injection, specialist execution, timeout, key collision warnings, unknown key filtering, and specialist name persistence. While each section is well-documented with R-fix references, the function does many things.

**Assessment**: The function's complexity is inherent (dispatch IS complex with fallback paths, feature flags, and safety guards). Previous rounds extracted `execute_specialist` into `_base.py` for the specialist execution logic itself. Further splitting would likely increase cognitive load (more files to trace). Low priority.

---

### D1 Score Assessment

| Finding | Severity | Fixable? | Score Impact |
|---------|----------|----------|-------------|
| D1-M001 | MAJOR | Yes (clear before END) | +0.25 |
| D1-M002 | MAJOR | Yes (specialist metadata) | +0.15 |
| D1-M003 | MAJOR | Yes (doc update) | +0.10 (D9) |
| D1-m001 | MINOR | No (deliberate design) | 0.0 |
| D1-m002 | MINOR | Low priority | +0.05 |

**Current D1 score: 8.5** | **Potential after fixes: 8.9**

---

## D9: Trade-off Documentation (Current: 8.0, Weight: 0.05)

### D9-M001: ARCHITECTURE.md guardrail pattern count is 84, actual is 185 (MAJOR)

**File**: `ARCHITECTURE.md:98-104`
**Evidence**: See D1-M003 above. The ARCHITECTURE.md references "84 total patterns" which was accurate before R35 (pre-sprint research added Hindi, Tagalog, French, Vietnamese, Arabic, Japanese, Korean patterns). Now 185 patterns across 11 languages. Also, line 22 says "84 regex patterns" in the compliance_gate section -- should read 185.

**Fix**: Update both occurrences in ARCHITECTURE.md.

---

### D9-M002: ARCHITECTURE.md says "compliance_gate (84 regex patterns)" -- stale language count (MAJOR)

**File**: `ARCHITECTURE.md:22`
**Evidence**: Same root cause as D9-M001. The system overview ASCII diagram says "84 regex patterns" but the compliance_gate docstring says "11 languages". ARCHITECTURE.md line 98 says "31 patterns (17 EN + 8 ES + 3 PT + 3 ZH)" for responsible gaming alone, but the actual file now has Hindi and Tagalog patterns too.

**Fix**: Update ARCHITECTURE.md to reflect 185 patterns, 11 languages. Update the per-category breakdown (line 98) to include Hindi/Tagalog counts.

---

### D9-M003: Missing ADR for SIGTERM graceful drain (R40 feature) (MAJOR)

**File**: `src/api/app.py:47-54`, `docs/runbook.md`
**Evidence**: R40 added SIGTERM graceful drain with `_active_streams` tracking, `_shutting_down` event, and 30s drain timeout. This is a significant production behavior change. The runbook mentions graceful shutdown in the "Cloud Run Service Configuration" section (line 25) but doesn't document the SIGTERM drain mechanism, the `_DRAIN_TIMEOUT_S` parameter, or the interaction with Cloud Run's `--timeout=180s`.

**Fix**: Add to runbook.md under a new "Graceful Shutdown" section:
- SIGTERM handler sets `_shutting_down` event
- New `/chat` requests return 503 during drain
- Active SSE streams have 30s to complete
- After drain timeout, pending streams are force-cancelled
- Cloud Run's --timeout=180s is the outer bound; uvicorn's --timeout-graceful-shutdown=15s is for non-SSE connections

---

### D9-M004: Missing ADR for TTL jitter (R40 feature) (MAJOR)

**File**: `src/agent/nodes.py:127-133`, `src/config.py`
**Evidence**: R40 added TTL jitter (`+ _random.randint(0, 300)`) to all 8 singleton caches. The `import random` at line 127 of nodes.py has a comment referencing "R40 fix D8-C001" but there's no ADR explaining: (a) why jitter range is 0-300s, (b) why random.randint is acceptable (non-cryptographic RNG for timing jitter), (c) the thundering herd failure mode it prevents.

**Fix**: Add inline ADR comment or entry in ARCHITECTURE.md explaining the jitter strategy.

---

### D9-m001: cloudbuild.yaml staging strategy ADR is thorough but references nonexistent future state (MINOR)

**File**: `cloudbuild.yaml:1-11`
**Evidence**: The staging strategy ADR at the top of cloudbuild.yaml is well-written and explains why staging doesn't exist yet. However, it references "Branch-based staging via Cloud Build triggers" as "PLANNED" without a timeline or conditions. Adding a concrete trigger condition (e.g., "Implement when: second developer joins or first enterprise client onboards") would strengthen it.

---

### D9-m002: Inline ADRs scattered across files, no central ADR index (MINOR)

**File**: Various
**Evidence**: ADRs are in: `nodes.py` (i18n), `_base.py` (LLM concurrency), `memory.py` (checkpointer), `middleware.py` (rate limiting), `cloudbuild.yaml` (staging). There's no central ADR index or `docs/adrs/` directory. For a 51-module codebase, inline ADRs work but discoverability suffers.

**Assessment**: Low priority for MVP. A central `docs/adrs/README.md` linking to inline ADRs would improve discoverability but is cosmetic.

---

### D9 Score Assessment

| Finding | Severity | Fixable? | Score Impact |
|---------|----------|----------|-------------|
| D9-M001 | MAJOR | Yes (doc update) | +0.25 |
| D9-M002 | MAJOR | Yes (doc update) | (same fix as M001) |
| D9-M003 | MAJOR | Yes (runbook section) | +0.25 |
| D9-M004 | MAJOR | Yes (inline ADR) | +0.15 |
| D9-m001 | MINOR | Low priority | +0.05 |
| D9-m002 | MINOR | Low priority | +0.05 |

**Current D9 score: 8.0** | **Potential after fixes: 8.7-9.0**

---

## Cross-Validation Summary

### GPT-5.2 Codex (Security Focus)
- Confirmed D6-C001 (--require-hashes) as highest priority
- Confirmed D6-M003 (curl removal) and D6-M004 (.dockerignore)
- Flagged filesystem permissions hardening (not in my findings -- low-priority hardening)
- Recommended SBOM + cosign (aligned with D6-M001, D6-M002)

### Gemini 3 Pro (Architecture Focus)
- Flagged generate node as "observability black box" (aligned with D1-M002 but overstated severity -- specialist name IS logged, just not in SSE events)
- Flagged state serialization bloat from retrieved_context (aligned with D1-M001)
- Flagged compliance_gate intent routing as SRP violation (rejected -- deliberate design with documentation)
- Flagged _merge_dicts lack of deletion sentinel (overstated -- R37/R38 fixes already handle None/empty filtering; actual deletion not needed for guest profiling use case)
- Flagged concurrent user double-click scenario (valid concern for MemorySaver but mitigated by FirestoreSaver in production which uses optimistic concurrency)

---

## Score Proposal

| Dimension | R40 Score | Findings | Proposed Score | Delta |
|-----------|-----------|----------|---------------|-------|
| D1: Graph Architecture | 8.5 | 2 MAJOR, 2 MINOR | 8.5 (no change pre-fix) | 0.0 |
| D6: Docker & DevOps | 7.0 | 1 CRIT, 4 MAJOR, 2 MINOR | 7.0 (no change pre-fix) | 0.0 |
| D9: Trade-off Docs | 8.0 | 4 MAJOR, 2 MINOR | 8.0 (no change pre-fix) | 0.0 |

**Post-fix potential:**

| Dimension | Current | Post-Fix Target | Delta |
|-----------|---------|----------------|-------|
| D1 | 8.5 | 8.9 | +0.4 |
| D6 | 7.0 | 8.5-9.0 | +1.5-2.0 |
| D9 | 8.0 | 8.7-9.0 | +0.7-1.0 |

**Weighted score impact of fixes**: D6 at 0.10 weight with +2.0 = +0.20; D1 at 0.20 weight with +0.4 = +0.08; D9 at 0.05 weight with +0.9 = +0.045. Total potential: **+0.325 weighted** (93.5 -> ~96.8 with ALL fixes).

---

## Finding Count Summary

| Severity | D6 | D1 | D9 | Total |
|----------|---:|---:|---:|------:|
| CRITICAL | 1 | 0 | 0 | 1 |
| MAJOR | 4 | 2 | 4 | 10 |
| MINOR | 2 | 2 | 2 | 6 |
| **Total** | **7** | **4** | **6** | **17** |

---

## Priority Fix Order

1. **D6-C001**: --require-hashes (highest impact, longest carried)
2. **D9-M001/M002**: ARCHITECTURE.md pattern count update (quick doc fix)
3. **D6-M003**: Remove curl, use Python healthcheck
4. **D6-M004**: .dockerignore additions
5. **D1-M001**: Clear retrieved_context before END nodes
6. **D1-M002**: Specialist name in graph_node metadata
7. **D9-M003**: Runbook SIGTERM drain section
8. **D9-M004**: TTL jitter ADR
9. **D6-M001/M002**: SBOM + cosign (v2 scope if time permits)
