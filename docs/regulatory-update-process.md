# Regulatory Update Process

Process for propagating state gaming regulation changes to the Hey Seven platform.

## Monitoring

| Source | Frequency | Responsible |
|--------|-----------|-------------|
| State gaming commission websites | Monthly | Operations team |
| NCPG (National Council on Problem Gambling) | Quarterly | Operations team |
| Legal counsel alerts | As received | Legal |
| Client notifications | As received | Account management |

## Update Workflow

### 1. Identify Change

Document the regulatory change:
- State affected
- Effective date
- Summary of change (new requirement, modified threshold, repealed rule)
- Source URL / citation

### 2. Impact Assessment

Determine which components are affected:

| Component | When Affected |
|-----------|---------------|
| `src/casino/config.py` | Helpline numbers, age minimums, self-exclusion authorities |
| `src/agent/guardrails.py` | New prohibited topics, modified patterns |
| `knowledge-base/regulations/` | Regulatory reference content for RAG |
| `src/agent/prompts.py` | AI disclosure requirements, disclaimers |
| `src/sms/compliance.py` | TCPA rules, quiet hours, consent requirements |

### 3. Implement Changes

1. Update the affected files (see Impact Assessment)
2. Add `_regulatory_version` date stamp to the casino profile if regulations section changes
3. Update `knowledge-base/regulations/state-requirements.md` with the new rule

### 4. Test

```bash
# Run full test suite
make test-ci

# Run guardrail-specific tests
python -m pytest tests/test_guardrails.py tests/test_compliance_gate.py -v

# Run regulatory invariant tests
python -m pytest tests/test_regulatory_invariants.py -v
```

### 5. Deploy

Follow standard deployment process. Regulatory updates are treated as
high-priority changes — deploy within 30 days of effective date.

## SLA

| Priority | SLA | Example |
|----------|-----|---------|
| Critical | 7 days | Self-exclusion list changes, new prohibited content |
| High | 30 days | Helpline number changes, AI disclosure law amendments |
| Normal | 90 days | Documentation updates, formatting changes |

## Tracking

Each regulatory update is tracked as a commit with prefix `fix(regulatory):` and
references the state + effective date in the commit message. Example:
```
fix(regulatory): NJ SB 3401 push notification ban (effective 2026-03-01)
```
