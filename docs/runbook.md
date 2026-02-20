# Hey Seven — Production Operations Runbook

## Service Overview

| Component | URL | Health |
|-----------|-----|--------|
| Hey Seven API | https://hey-seven-XXXXX.run.app | GET /health |
| Cloud Run Console | console.cloud.google.com/run | Dashboard |
| LangFuse Dashboard | cloud.langfuse.com | Traces |

## Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Retrieval relevance | < 80% | < 70% | Review query patterns, re-chunk data |
| Validation pass rate | < 90% | < 85% | Investigate prompt drift, check LLM changes |
| "I don't know" rate | > 20% | > 30% | Add data to knowledge base |
| Response latency P95 | > 3s | > 5s | Check LLM latency, review prompt length |
| Error rate | > 1% | > 2% | Check Gemini API status page |
| Circuit breaker opens | > 1/hour | > 3/hour | Gemini API degradation — check status |

## Escalation Matrix

| Severity | Response Time | Who | Channel |
|----------|--------------|-----|---------|
| P0 (service down) | 15 min | On-call engineer | Slack #hey-seven-alerts |
| P1 (degraded) | 1 hour | Engineering team | Slack #hey-seven-ops |
| P2 (non-urgent) | 4 hours | Engineering team | Jira ticket |
| P3 (cosmetic) | Next sprint | Product team | Jira backlog |

## Common Failure Scenarios

### 1. Gemini API Outage

**Symptoms**: Circuit breaker open, all chat requests return fallback responses.

**Diagnosis**:
1. Check Gemini API status: https://status.cloud.google.com/
2. Check circuit breaker state in logs: `severity=WARNING msg="Circuit breaker OPEN"`
3. Check error rate spike in Cloud Run metrics

**Remediation**:
1. Circuit breaker auto-recovers after 60s cooldown (half-open probe)
2. If persistent: check API key validity in Secret Manager
3. If API is down: wait for Google resolution; fallback responses serve gracefully

### 2. High "I Don't Know" Rate

**Symptoms**: > 30% of responses are fallback/no-info responses.

**Diagnosis**:
1. Check LangFuse traces for low retrieval scores
2. Check if knowledge base was recently updated
3. Check embedding model version (must match ingestion model)

**Remediation**:
1. Re-run data ingestion: restart Cloud Run instance (triggers lifespan ingestion)
2. Check `RAG_MIN_RELEVANCE_SCORE` threshold — may need lowering
3. Add missing content to knowledge-base/ JSON files

### 3. Cold Start Latency > 10s

**Symptoms**: First request after idle period takes > 10s.

**Diagnosis**:
1. Check Cloud Run instance count: scale-to-zero means cold start on first request
2. Check container startup logs for ingestion time

**Remediation**:
1. Set `min-instances=1` in Cloud Run config (prevents scale-to-zero)
2. Enable `--cpu-boost` for startup CPU allocation
3. Pre-build ChromaDB in Docker image (requires API key at build time — trade-off)

### 4. Rate Limit Exhaustion

**Symptoms**: Legitimate users getting 429 responses.

**Diagnosis**:
1. Check rate limit logs: `severity=WARNING msg="Rate limit exceeded"`
2. Check for bot traffic patterns (same IP, rapid requests)

**Remediation**:
1. Increase `RATE_LIMIT_CHAT` env var (default: 20 req/min)
2. Block abusive IPs at Cloud Run ingress (Armor/IAP)
3. Add user-agent filtering for known bots

### 5. Validation Loop Stuck

**Symptoms**: High latency, graph reaches recursion limit.

**Diagnosis**:
1. Check LangFuse traces for generate->validate->generate loops
2. Check `retry_count` in state — should max at 1

**Remediation**:
1. `GRAPH_RECURSION_LIMIT=10` prevents infinite loops (default)
2. Check validator prompt for overly strict criteria
3. Review recent prompt changes in LangFuse prompt versioning

### 6. PII in Logs

**Symptoms**: PII detected in Cloud Logging output.

**Diagnosis**:
1. Search Cloud Logging for phone patterns: `textPayload=~"\+1\d{10}"`
2. Check PII redaction middleware is active

**Remediation**:
1. Verify `pii_redaction.redact_pii()` is called in logging middleware
2. Add missing PII patterns to `_PATTERNS` list
3. Purge affected log entries from Cloud Logging

## Deployment Checklist

### Pre-Deploy
- [ ] All tests pass: `pytest tests/ --cov=src --cov-fail-under=90`
- [ ] Ruff clean: `ruff check src/`
- [ ] No PII in code: `grep -rn '@\|password\|secret' src/ --include='*.py'`
- [ ] Docker builds: `docker build -t hey-seven .`
- [ ] Health endpoint works: `curl localhost:8080/health`

### Post-Deploy
- [ ] Health endpoint returns 200: `curl https://hey-seven-XXXXX.run.app/health`
- [ ] Chat endpoint responds: Send test message via /chat
- [ ] Circuit breaker closed: No "OPEN" warnings in logs
- [ ] LangFuse traces appearing (if enabled)
- [ ] Response latency P95 < 5s

### Rollback
```bash
# Revert to previous revision
gcloud run services update-traffic hey-seven \
  --to-revisions=PREVIOUS_REVISION=100 \
  --region=us-central1

# Or redeploy specific commit
gcloud run deploy hey-seven \
  --image=us-central1-docker.pkg.dev/PROJECT/hey-seven/hey-seven:PREV_SHA \
  --region=us-central1
```

## Operational Contacts

| Role | Contact | Backup |
|------|---------|--------|
| Engineering Lead | TBD | TBD |
| GCP Admin | TBD | TBD |
| LangFuse Admin | TBD | TBD |

## Responsible Gaming Escalation

These are handled by guardrails (NO human judgment needed):
- **Self-exclusion mentions**: Auto-provides DMHAS 1-888-789-7777 (Connecticut)
- **Problem gambling**: Auto-provides 1-800-522-4700 (NCPG)
- **BSA/AML suspicious**: Auto-redirects to appropriate authorities
- **Underage mentions**: Auto-provides age verification info

## Key Environment Variables

| Variable | Default | Production |
|----------|---------|------------|
| ENVIRONMENT | development | production |
| LOG_LEVEL | INFO | WARNING |
| GOOGLE_API_KEY | (from Key Vault) | Secret Manager |
| RATE_LIMIT_CHAT | 20 | 30 |
| SSE_TIMEOUT_SECONDS | 60 | 60 |
| CB_FAILURE_THRESHOLD | 5 | 5 |
| CB_COOLDOWN_SECONDS | 60 | 60 |
| GRAPH_RECURSION_LIMIT | 10 | 10 |
| LANGFUSE_PUBLIC_KEY | (empty) | Set in Secret Manager |
