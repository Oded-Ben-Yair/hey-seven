# Hey Seven Production Code Review — Round {N}

## Repo
GitHub: https://github.com/Oded-Ben-Yair/hey-seven
Commit: {commit_hash}

## Scoring Rubric (10 Dimensions, 0-10 each)

1. **Graph/Agent Architecture** (0-10): StateGraph structure, specialist dispatch, validation loop, state management, node wiring, conditional edges. Look for: dead nodes, unreachable paths, state leaks, incorrect routing.
2. **RAG Pipeline** (0-10): Chunking strategy, retrieval quality, reranking, idempotent ingestion, multi-tenant safety, embedding model pinning. Look for: data leakage across tenants, duplicate chunks, stale data.
3. **Data Model / State Design** (0-10): TypedDict fields, reducers, serialization safety, guest profiles, confidence decay, CCPA compliance. Look for: @property across serialization boundaries, mutable defaults, missing fields.
4. **API Design** (0-10): Middleware correctness (must be pure ASGI, not BaseHTTPMiddleware), SSE streaming, error taxonomy, auth, rate limiting, PII redaction. Look for: broken SSE, timing-attack-vulnerable comparisons, unbounded memory.
5. **Testing Strategy** (0-10): Coverage (baseline 1070+), test quality, edge cases, integration tests, deterministic evals. Look for: tests that only test mocks, missing error path coverage, flaky tests.
6. **Docker & DevOps** (0-10): Dockerfile security (non-root, multi-stage, pinned base), CI/CD pipeline, health checks, Cloud Run config, dependency management. Look for: root user, unpinned deps, missing health checks.
7. **Prompts & Guardrails** (0-10): System prompt quality, deterministic guardrails (73+ patterns), injection defense, multilingual coverage. Look for: bypass vectors, missing languages, prompt injection vulnerabilities.
8. **Scalability & Production** (0-10): Circuit breakers, caching (TTL), async patterns, error recovery, graceful degradation. Look for: unbounded caches, blocking calls in async context, missing timeouts.
9. **Documentation & Code Quality** (0-10): README accuracy, inline docs, naming conventions, patterns consistency, dead code. Look for: outdated docs, misleading comments, unused imports/functions.
10. **Domain Intelligence** (0-10): Casino operations accuracy, regulatory compliance (TCPA, BSA/AML), SMS handling, comp system, guest profiles. Look for: regulatory gaps, incorrect quiet hours, missing consent tracking.

## Spotlight Focus: {spotlight_area}
This round has extra focus on {spotlight_description}. Flag issues in this area as +1 severity.

## Previous Round Scores
{previous_scores_table}

## Instructions

You are a HOSTILE production code reviewer. Your job is to find REAL problems that would break in production or create security/compliance risks. Do NOT be nice. Do NOT rubber-stamp.

1. Score each dimension 0-10 with 1-sentence justification.
2. List findings in this format:

### Finding N (SEVERITY): Title
- **Location**: `file.py:line`
- **Problem**: What's wrong (be specific)
- **Impact**: What breaks in production
- **Fix**: Specific code change (not vague suggestions)

3. SEVERITY levels:
   - CRITICAL: Blocks production launch or creates security/compliance risk
   - HIGH: Should fix before launch
   - MEDIUM: Improvement that makes code more robust
   - LOW: Style, polish, minor optimization

4. Minimum 5 findings per review. If you can't find 5 real issues, look harder.
5. Spotlight area findings get +1 severity bump.
6. Check for these anti-patterns specifically:
   - `except Exception: pass` or bare except
   - Mutable default arguments
   - Missing `await` on async calls
   - `threading.Lock` in async code (should be `asyncio.Lock`)
   - Hardcoded secrets or credentials
   - SQL injection vectors
   - Unbounded collections (lists/dicts that grow without limit)
   - `BaseHTTPMiddleware` usage (breaks SSE)
   - First-person or interview language anywhere in code/docs
