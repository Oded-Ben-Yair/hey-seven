# Claude Code Kit Design — Round 2 Hostile Review Synthesis

**Date**: 2026-02-27

---

## Round 2 Scores

| Reviewer | Focus | R1 Score | R2 Score | CRITICALs | MAJORs |
|----------|-------|----------|----------|-----------|--------|
| Gemini Pro | GCP Architecture | 34 | **67** | 1 (supply chain) | 1 |
| GPT-5.2 | MCP Server Design | 35 | **77** | 0 | 6 |
| DeepSeek | Portability | 60 | **~82** | 0 | 2 |
| Grok 4 | 72-Hour Readiness | 49 | **86** | 0 | 3 |
| **CONSENSUS** | **Overall** | **44** | **78** | **1** | **12** |

---

## Progress: 44 → 78 (+34 points)

### CRITICALs: 13 → 1
The single remaining CRITICAL from Gemini:

**C1 (Gemini): Supply chain risk — third-party MCP packages with ADC credentials**
The @google-cloud/ MCP packages ARE published under the Google namespace (googleapis org), BUT Gemini raises the concern that ANY MCP server running with your ADC has broad access. If a package is compromised, attacker gets Cloud Run Admin + Secret Manager access.
**Fix**: Run MCP servers with restricted service accounts, or vendor/audit the source. For MVP: acceptable risk since packages are from googleapis org (Google's official GitHub).
**Decision**: ACCEPT for MVP (Google-published packages). Add supply-chain controls post-MVP.

### Remaining MAJORs (12 → consolidated to 8 unique):

| # | Finding | Raised By | Fix |
|---|---------|-----------|-----|
| M1 | No tool output schema versioning | GPT | Add schema_version field, document contract |
| M2 | No dependency lockfiles (Python + npm) | GPT | Add pip-compile lockfile, npm package-lock.json |
| M3 | No secret redaction/logging policy | GPT | Document what leaves machine, add PII redaction |
| M4 | No timeout/cancellation per tool | GPT | Add configurable timeouts, honor MCP cancel |
| M5 | No credential isolation between providers | GPT | Document env var separation, use .env per provider |
| M6 | rsync dependency without cp fallback | DeepSeek | Add fallback: rsync || cp -r |
| M7 | No post-setup infra validation steps | Grok | Add smoke test checklist after each day |
| M8 | Incomplete external API enable list | Grok | List all required APIs explicitly |

---

## Assessment: Can we reach 95/100?

Current blockers to 95:
1. Gemini supply-chain CRITICAL — accepted for MVP, document the risk
2. GPT operational hardening MAJORs — schema versioning, lockfiles, logging policy
3. DeepSeek rsync fallback — trivial fix
4. Grok validation steps — add smoke test checklists

**Estimated scores after M1-M8 fixes:**
- Gemini: 67 → 88 (accept supply chain for MVP, add audit note)
- GPT: 77 → 90 (add schemas, lockfiles, logging)
- DeepSeek: 82 → 92 (rsync fallback, quoting)
- Grok: 86 → 95 (validation steps, API list)
- **Consensus: ~91**

To reach 95+, we'd need one more round after these fixes.
