# Claude Code Kit Design — Round 1 Hostile Review Synthesis

**Date**: 2026-02-27
**Reviewers**: Gemini 3.1 Pro, GPT-5.2, DeepSeek-V3.2-Speciale, Grok 4

---

## Round 1 Scores

| Reviewer | Focus | Score | CRITICALs | MAJORs |
|----------|-------|-------|-----------|--------|
| Gemini Pro | GCP Architecture | 34/100 | 3 | 2 |
| GPT-5.2 | MCP Server Design | 35/100 | 6 | 5 |
| DeepSeek | Portability | 60/100 | 2 | 5 |
| Grok 4 | 72-Hour Readiness | 49/100 | 2 | 4 |
| **CONSENSUS** | **Overall** | **44/100** | **13** | **16** |

---

## CRITICALs (Must Fix — 13 total, 8 unique after dedup)

### C1: Google MCP packages are HALLUCINATED (Gemini + GPT confirmed)
`@google-cloud/gcloud-mcp`, `@google-cloud/observability-mcp`, `@google-cloud/cloud-run-mcp` DO NOT EXIST as npm packages.
**Fix**: Remove from design. Use `gcloud` CLI directly via Bash tool, or build custom lightweight wrappers. Research actual available GCP MCP servers.

### C2: DeepSeek-R1 cannot use Vertex AI `GenerativeModel` class (Gemini + GPT confirmed)
GenerativeModel only works for Gemini/PaLM. DeepSeek on Model Garden requires OpenAI-compatible API endpoint with ADC bearer token.
**Fix**: Use `openai.AsyncOpenAI(base_url="https://<region>-aiplatform.googleapis.com/...", api_key=get_bearer_token())` for DeepSeek. OR ship Gemini-only first.

### C3: Vertex AI does NOT accept API keys (Gemini + GPT confirmed)
"API key fallback" only works for Google AI Studio (generativelanguage.googleapis.com), NOT for Vertex AI.
**Fix**: Split auth cleanly: Vertex AI = ADC only. Google AI Studio = API key. Different endpoints, different clients.

### C4: `docs-langchain` MCP endpoint is unverified (GPT flagged)
`https://docs.langchain.com/mcp` has not been confirmed as a working MCP JSON-RPC endpoint.
**Fix**: Remove from default config. Replace with verified alternatives (context7, mcpdoc with llms.txt).

### C5: settings.json $HOME expansion is unreliable (GPT + DeepSeek confirmed)
Claude Code may not expand `$HOME` or `~` in JSON config. Must use absolute paths.
**Fix**: install.sh writes absolute paths into settings.json at generation time. No env var placeholders.

### C6: Symlinks may break Claude Code (GPT + DeepSeek confirmed)
Symlinked directories in ~/.claude/ may confuse file watchers, relative path resolution, and plugin systems.
**Fix**: Default to COPY mode (rsync). Offer --link mode as optional for dev workflow. Test both modes.

### C7: pip install -e without venv (GPT + DeepSeek confirmed)
Pollutes system Python, version conflicts, permissions issues.
**Fix**: Create dedicated venv at `~/.claude/mcp-venvs/multi-provider-ai/` and reference the venv's Python in MCP launcher.

### C8: GCP project must be pre-provisioned (Grok flagged HIGH)
Without a GCP project on day 1, nothing else works. Not under developer's control.
**Fix**: Add pre-arrival checklist item: confirm GCP project access, billing, and required IAM roles before start date.

---

## MAJORs (Should Fix — 16 total, 10 unique after dedup)

### M1: Rollback CLI uses fake "PREV" keyword (Gemini)
`gcloud run services update-traffic --to-revisions=PREV=100` — PREV is not valid.
**Fix**: Programmatically fetch previous revision ID via `gcloud run revisions list`.

### M2: Cloud Build is synchronous by default (Gemini)
120s wait window in cloud-build-gate.sh is unnecessary — `gcloud builds submit` blocks until complete.
**Fix**: Just check exit code. Remove arbitrary wait.

### M3: Manual revision cleanup is anti-pattern (Gemini)
Cloud Run auto-garbage-collects revisions.
**Fix**: Remove cleanup step from /gcp-deploy skill.

### M4: Missing canary --tag in deploy flow (Gemini)
Must tag revision during deploy for clean canary URL routing.
**Fix**: Add `--tag=canary` to deploy command.

### M5: Tool I/O contracts underspecified (GPT)
gcp_code_review should return structured findings, not free text.
**Fix**: Define Pydantic schemas for tool inputs/outputs.

### M6: Missing venv for MCP Python server (GPT + DeepSeek)
No isolated Python environment specified.
**Fix**: Create venv, pin all deps, reference in launcher.

### M7: Node npx versions not pinned (GPT + DeepSeek)
`npx @google-cloud/...` without version is non-reproducible.
**Fix**: Pin all npm packages to exact versions.

### M8: Missing credential handoff checklist (Grok)
External API keys (Telnyx, Langfuse, Sheets) need pre-arrangement.
**Fix**: Add credential sourcing checklist to 72-hour plan.

### M9: DNS/SSL not addressed for production (Grok)
Default *.run.app URLs aren't production-ready for a casino app.
**Fix**: Add domain mapping step to Day 2 plan.

### M10: macOS bash is v3 (DeepSeek)
#!/bin/bash on macOS gets bash 3.2 which lacks bash 4+ features.
**Fix**: Use `#!/usr/bin/env bash` and test with bash 3.2, or require brew bash.

---

## Design Fixes Required Before Round 2

1. **Remove hallucinated Google MCP packages** — replace with gcloud CLI direct or verified alternatives
2. **Fix DeepSeek-R1 access pattern** — OpenAI SDK with Vertex AI endpoint, not GenerativeModel
3. **Split auth: Vertex AI (ADC) vs AI Studio (API key)** — different clients, endpoints
4. **Remove docs-langchain MCP** until verified
5. **install.sh: write absolute paths, not $HOME** in settings.json
6. **Default to COPY mode** (rsync), optional symlink mode
7. **Create venv for custom MCP** in ~/.claude/mcp-venvs/
8. **Fix all gcloud CLI commands** (rollback, deploy --tag, no revision cleanup)
9. **Add pre-arrival GCP checklist** (project, IAM, billing, credentials)
10. **Add DNS/SSL step** to 72-hour plan
11. **Pin all dependency versions**
12. **Use #!/usr/bin/env bash** in all scripts

---

## Path to 95/100

After fixing all 8 CRITICALs and 10 MAJORs:
- GCP Architecture: 34 → ~80 (remove hallucinations, fix CLI)
- MCP Design: 35 → ~80 (fix auth, DeepSeek, venv, schemas)
- Portability: 60 → ~85 (copy mode, absolute paths, venv)
- 72-Hour Readiness: 49 → ~80 (pre-arrival checklist, DNS, credentials)

Round 2 will need: refined MCP tool schemas, verified Google MCP alternatives, tested install.sh on clean environment.
