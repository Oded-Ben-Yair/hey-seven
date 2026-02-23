# R41 Fixes: D6 DevOps + D1 Graph Architecture

**Date**: 2026-02-23
**Fixer**: fixer-alpha (Claude Opus 4.6)

---

## D6 Fixes (5 findings addressed)

### D6-C001: --require-hashes enforced (CRITICAL -> RESOLVED)

- Created `requirements-prod.in` as source-of-truth input file
- Generated `requirements-prod.txt` with SHA-256 hashes via `pip-compile --generate-hashes`
- Updated Dockerfile line 19: `pip install --no-cache-dir --require-hashes --target=/build/deps -r requirements-prod.txt`
- Bumped `langgraph-checkpoint-firestore` from 0.1.0 (yanked) to 0.1.7 (latest compatible)
- 2325 hash entries across all direct + transitive dependencies

**Files**: `Dockerfile`, `requirements-prod.txt`, `requirements-prod.in` (new)

### D6-M001: SBOM generation added (MAJOR -> RESOLVED)

- Added Step 3b to `cloudbuild.yaml`: Trivy CycloneDX SBOM generation
- Uses same pinned Trivy image (0.58.2) as vulnerability scan step
- Output: `/workspace/sbom.json` in CycloneDX format per NIST SP 800-218

**Files**: `cloudbuild.yaml`

### D6-M002: Cosign image signing documented as ADR (MAJOR -> DOCUMENTED)

- Added inline ADR in `cloudbuild.yaml` between Steps 4 and 5
- Documents: why not yet (no KMS key, single developer), when (enterprise client onboard), how (GCP KMS)
- Not implemented as active step -- requires KMS infrastructure provisioning

**Files**: `cloudbuild.yaml`

### D6-M003: curl removed from production image (MAJOR -> RESOLVED)

- Removed `apt-get install curl` from Stage 2
- Replaced curl-based HEALTHCHECK with Python urllib: `python -c "import urllib.request; urllib.request.urlopen(...)"`
- Reduces attack surface (no curl binary in final image)

**Files**: `Dockerfile`

### D6-M004: .dockerignore gaps fixed (MAJOR -> RESOLVED)

- Added `.claude/` and `.hypothesis/` to `.dockerignore`
- Prevents session data and property-based test artifacts from leaking into Docker image

**Files**: `.dockerignore`

---

## D1 Fixes (2 findings addressed)

### D1-M001: retrieved_context cleared before END nodes (MAJOR -> RESOLVED)

- Added `"retrieved_context": []` to return dicts of:
  - `respond_node` (main path: persona_envelope -> respond -> END)
  - `fallback_node` (validation failure path: fallback -> END)
  - `greeting_node` (defensive: never follows retrieve, but ensures clean state)
  - `off_topic_node` (defensive: same rationale)
- Prevents stale chunk data (~2.5KB/turn) from accumulating in Firestore checkpoints

**Files**: `src/agent/nodes.py`

### D1-M002: specialist_name in SSE metadata (MAJOR -> RESOLVED)

- Added `NODE_GENERATE` case to `_extract_node_metadata()` in `graph.py`
- Returns `{"specialist": output.get("specialist_name")}` for generate node
- Updated test: `test_generate_node_returns_specialist` added, `test_unknown_node_returns_empty` updated
- Specialist name now visible in SSE `graph_node` events for LangSmith/Langfuse traces

**Files**: `src/agent/graph.py`, `tests/test_agent.py`

---

## Test Results

- **2169 passed**, 0 failed, 66 warnings
- Coverage: **90.11%** (above 90% gate)
- Runtime: 329s

---

## Score Impact Estimate

| Finding | Severity | Status | Score Impact |
|---------|----------|--------|-------------|
| D6-C001 | CRITICAL | Resolved | +1.0 |
| D6-M001 | MAJOR | Resolved | +0.25 |
| D6-M002 | MAJOR | Documented (ADR) | +0.10 |
| D6-M003 | MAJOR | Resolved | +0.25 |
| D6-M004 | MAJOR | Resolved | +0.15 |
| D1-M001 | MAJOR | Resolved | +0.25 |
| D1-M002 | MAJOR | Resolved | +0.15 |

**D6 estimated post-fix: 7.0 -> 8.5-8.75**
**D1 estimated post-fix: 8.5 -> 8.9**
