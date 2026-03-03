# Claude Code Portable Kit — Complete Design Document

**Date**: 2026-02-27
**Author**: Oded Ben-Yair + Claude Opus 4.6
**Goal**: Portable Claude Code environment for Hey Seven development on GCP
**Success Criteria**: Clone repo, run install.sh, start Claude Code → fully productive in < 30 minutes
**72-Hour Target**: From kit installation to Hey Seven MVP live on Cloud Run

---

## 1. Problem Statement

Oded is transitioning from a company using Azure to Hey Seven (a startup using GCP). The current Claude Code environment at `~/.claude/` contains 4+ months of accumulated knowledge, skills, hooks, rules, MCP servers, and agent definitions — all deeply integrated with Azure services.

**What must transfer**: All universal development knowledge, LangGraph patterns, RAG production patterns, hostile review protocols, agent team architecture, multi-model debate capabilities, and hey-seven project-specific knowledge.

**What must change**: Azure-specific tooling → GCP-native equivalents. Azure DevOps → GitHub. Azure Key Vault → GCP Secret Manager. Azure AI Foundry MCP → multi-provider AI MCP with Vertex AI backend.

**What must be added**: Official Google Cloud MCP servers, LangChain Docs MCP, GCP-specific deploy/ops skills and hooks.

---

## 2. Repository Structure

```
claude-code-kit/                    # Git repo: github.com/Oded-Ben-Yair/claude-code-kit
├── install.sh                      # Bootstrap (cross-platform, idempotent)
├── uninstall.sh                    # Clean removal of all symlinks
├── sync.sh                         # Validate symlinks, detect drift, report status
├── .gitattributes                  # Enforce LF line endings
├── .gitignore                      # Exclude secrets, caches, node_modules
│
├── core/                           # Core config files (copied, not symlinked)
│   ├── CLAUDE.md                   # Master Claude Code config (GCP-native)
│   ├── MEMORY.md                   # Starter memory (hey-seven learnings baked in)
│   └── settings.json.template      # Settings template with $HOME placeholders
│
├── rules/                          # Always-loaded rules → symlinked as ~/.claude/rules/
│   ├── code-quality.md             # Universal code quality (Azure refs removed)
│   ├── gcp-deploy.md               # Cloud Run, Cloud Build, Secret Manager patterns
│   ├── gcp-safety.md               # Firestore safety, IAM, billing, cost control
│   ├── langgraph-patterns.md       # LangGraph production patterns (universal)
│   ├── rag-production.md           # RAG pipeline patterns (universal)
│   ├── orchestration-patterns.md   # Agent orchestration (universal)
│   ├── hostile-review-protocol.md  # Multi-model hostile review (adapted MCP names)
│   ├── team-templates.md           # Agent team compositions (adapted MCP names)
│   ├── diagramming.md              # D2/Mermaid diagramming (universal)
│   ├── cleanup-safety.md           # File deletion safety (universal)
│   └── project-config.md           # Single-project config (Hey Seven only)
│
├── skills/                         # Custom skills → symlinked individually
│   ├── go/                         # Session resume with GCP context
│   ├── end-of-session/             # Session close with GitHub push
│   ├── multi-model-debate/         # 6-LLM debate (gcp_* + grok_* + gemini_*)
│   ├── pre-mortem/                 # Risk assessment (universal)
│   ├── ship-it/                    # Anti-perfectionism (universal)
│   ├── frontend/                   # Frontend design (universal)
│   ├── create-diagram/             # Diagram generation (universal)
│   ├── image-asset-studio/         # Multi-engine image gen (Gemini + Grok)
│   ├── gemini3-pro/                # Gemini advanced features guide
│   ├── b2b-copy-writer/            # B2B copy (adapted tool names)
│   ├── browser-control/            # Cross-platform browser automation
│   ├── architecture-doc/           # Architecture documentation
│   ├── team-deploy/                # Agent team deployment
│   ├── learning-loop/              # Session learning extraction
│   ├── morning-update/             # Daily briefing
│   ├── gcp-deploy/                 # NEW: Cloud Run zero-downtime deploy
│   ├── gcp-rollback/               # NEW: Instant Cloud Run rollback
│   ├── gcp-status/                 # NEW: GCP project health dashboard
│   ├── gcp-logs/                   # NEW: Structured log analysis
│   └── gcp-compliance/             # NEW: GCP resource audit (naming, tags, IAM)
│
├── hooks/                          # Event hooks → symlinked as ~/.claude/hooks/
│   ├── auto-router.py              # Intent detection → routing (GCP-adapted)
│   ├── session-start-enhanced.sh   # Git context + gcloud context + status
│   ├── pre-compact-save.sh         # Save critical state before compaction
│   ├── post-compact-recover.sh     # Re-inject state after compaction
│   ├── quality-validation.sh       # Security pattern validation (universal)
│   ├── pre-tool-file-guard.sh      # Block writes to secrets/cross-project
│   ├── stop-verify.sh              # Proof-before-done (universal)
│   ├── notification.sh             # Cross-platform: notify-send (Linux) / osascript (macOS)
│   ├── cloud-build-gate.sh         # NEW: Wait for Cloud Build before verification
│   ├── gcp-adc-check.sh            # NEW: Validate ADC before gcloud commands
│   ├── schema-verify.sh            # Adapted: Firestore awareness
│   ├── dead-code-check.sh          # Universal ($HOME not hardcoded path)
│   ├── debug-first.sh              # Universal ($HOME not hardcoded path)
│   ├── test-result-tracker.sh      # Universal
│   ├── post-tool-autoformat.sh     # Universal
│   ├── periodic-commit-check.sh    # Adapted: GitHub push instead of Azure DevOps
│   ├── teammate-idle-verify.sh     # Universal
│   ├── task-completed-verify.sh    # Adapted: gcloud instead of az
│   ├── subagent-stop-tracker.sh    # Universal
│   ├── tool-failure-tracker.sh     # Universal
│   ├── config-change-audit.sh      # Universal ($HOME)
│   └── worktree-audit.sh           # Universal
│
├── agents/                         # Agent definitions → symlinked as ~/.claude/agents/
│   ├── architect-planner.md        # Design plans (gcp_* MCP tools)
│   ├── code-worker.md              # Execute plans (gcp_* MCP tools)
│   ├── code-judge.md               # Hostile code review (Read-only)
│   ├── code-simplifier.md          # Code simplification (universal)
│   ├── gemini-specialist.md        # Vision, docs, images (universal)
│   ├── research-specialist.md      # Web research (universal)
│   ├── realtime-specialist.md      # Social media/X (universal)
│   ├── reasoning-specialist.md     # Math/algorithms (gcp_* + gemini_*)
│   └── gcp-specialist.md           # NEW: Cloud Run, Firestore, Build ops
│
├── docs/                           # On-demand docs → symlinked as ~/.claude/docs/
│   ├── cloud-run.md                # NEW: Cloud Run patterns (replaces azure-functions.md)
│   ├── gcp-cli-reference.md        # NEW: Comprehensive gcloud command reference
│   ├── fastapi-streaming.md        # SSE/streaming patterns (universal)
│   ├── sse-chat-patterns.md        # SSE chat implementation (universal)
│   ├── visual-validation.md        # Screenshot validation (universal)
│   ├── voice-agent-tuning.md       # ElevenLabs voice (universal)
│   ├── ml-production.md            # ML patterns (adapted)
│   ├── pipeline-safety.md          # CI/CD safety (Cloud Build adapted)
│   ├── deliverable-quality.md      # Workbook/strategy quality (universal)
│   └── iterative-review-protocol.md # Review protocol (universal)
│
├── scripts/                        # Utility scripts → symlinked as ~/.claude/scripts/
│   ├── render-mermaid.js           # Mermaid rendering (universal)
│   ├── generate-interactive-diagram.sh  # Interactive diagram gen (universal)
│   ├── generate-macro-diagram.py   # Macro diagram gen (universal)
│   ├── skill-audit.sh              # Skill validation ($HOME adapted)
│   ├── rotate-telemetry.sh         # Telemetry cleanup (universal)
│   └── gcp-cost-query.sh           # NEW: BigQuery billing query helper
│
├── mcp-servers/                    # Custom MCP server source code
│   ├── multi-provider-ai/          # NEW: Replaces azure-ai-foundry
│   │   ├── pyproject.toml          # Python package config
│   │   ├── src/
│   │   │   └── gcp_ai_foundry/
│   │   │       ├── __init__.py
│   │   │       ├── server.py       # FastMCP 2.0 instance + tool definitions
│   │   │       ├── models.py       # Model registry (Gemini, DeepSeek, Claude)
│   │   │       └── auth.py         # ADC + API key fallback
│   │   ├── tests/
│   │   │   └── test_tools.py       # In-memory transport tests
│   │   ├── start-with-adc.sh       # Launcher: ADC credentials
│   │   └── start-with-apikeys.sh   # Fallback: direct API keys
│   ├── gemini-mcp/                 # Existing Gemini MCP (copy)
│   ├── grok/                       # Existing Grok MCP (copy, API key based)
│   └── playwright-cdp/             # Browser MCP (cross-platform launcher)
│       └── start-browser-mcp.sh    # Auto-detect Chrome/Edge/Chromium
│
├── configs/
│   ├── gcp-compliance-rules.json   # GCP naming/tagging rules
│   └── hey-seven-env.example       # Example .env for hey-seven project
│
├── templates/
│   ├── interactive-diagram.html    # React Flow template (universal)
│   ├── session-start.md            # Session start template (GCP adapted)
│   └── handover.md                 # Session handover template
│
└── themes/
    └── hey-seven.d2                # D2 theme for project diagrams
```

---

## 3. Multi-Provider AI MCP Server Design

### 3.1 Architecture

Replaces `azure-ai-foundry` (616 LOC Node.js) with Python FastMCP 2.0 server (~500 LOC).

**Why Python over TypeScript**:
- `google-cloud-aiplatform` Python SDK is more mature than JS equivalent
- FastMCP 2.0 provides decorator-based tools (less boilerplate)
- In-memory test transport for pytest (matches our test patterns)
- Hey Seven backend is Python — consistent ecosystem

**Why FastMCP 2.0**:
- MCP protocol version `2025-11-25` (latest stable)
- Decorator-based tool definition: `@mcp.tool()`
- Built-in `Client` class for in-memory testing
- `fastmcp.json` declarative config support
- stdio transport (lowest latency, ~1ms overhead)

### 3.2 Tool Interface

| Tool | Backend Model | Purpose | Replaces |
|------|--------------|---------|----------|
| `gcp_chat` | Gemini 2.5 Flash/Pro | General chat, Q&A | `azure_chat` |
| `gcp_code_review` | Gemini 2.5 Pro (thinking=high) | Code review with structured findings | `azure_code_review` |
| `gcp_brainstorm` | Gemini 2.5 Pro | Creative ideation, 5-7 ideas | `azure_brainstorm` |
| `gcp_research` | Gemini 2.5 Pro + Google Search grounding | Research with web citations | `azure_research` |
| `gcp_reason` | Gemini 2.5 Pro (thinking=high) | Complex step-by-step reasoning | `azure_reason` |
| `gcp_deepseek_reason` | DeepSeek-R1 via Vertex AI Model Garden | Math, algorithms, proofs | `azure_deepseek_reason` |
| `gcp_generate_image` | Imagen 3 via Vertex AI | Image generation | `azure_generate_image` |
| `list_models` | — | List available models and capabilities | `list_models` |

### 3.3 Authentication (R1 Fix: Split Vertex AI vs AI Studio)

**IMPORTANT**: Vertex AI and Google AI Studio use DIFFERENT auth mechanisms and endpoints. Never mix them.

**Vertex AI models** (Gemini via Vertex, DeepSeek via Model Garden): **ADC only**
```python
# Vertex AI requires OAuth2/ADC — NO API keys
import google.auth
credentials, project = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
# ADC resolution order:
# 1. GOOGLE_APPLICATION_CREDENTIALS env var (service account key file)
# 2. User credentials from `gcloud auth application-default login`
# 3. Metadata server (Cloud Run, GCE — automatic)
```

**Google AI Studio models** (Gemini via Developer API — simpler, no project needed): **API key**
```python
# AI Studio uses API keys, endpoint: generativelanguage.googleapis.com
import google.generativeai as genai
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
```

**Design decision**: The MCP server supports BOTH backends with explicit provider selection:
- `provider: "vertex"` → ADC auth, `aiplatform.googleapis.com` endpoint
- `provider: "ai_studio"` → API key, `generativelanguage.googleapis.com` endpoint
- The launcher script (`start-with-adc.sh` or `start-with-apikeys.sh`) sets the auth mode

### 3.4 Model Registry (R1 Fix: Correct DeepSeek access pattern)

```python
MODELS = {
    # Vertex AI native — uses vertexai SDK, ADC auth
    "gemini-flash": {
        "model_id": "gemini-2.5-flash-001",
        "provider": "vertex",
        "sdk": "vertexai.GenerativeModel",
        "thinking": False,
        "temperature": True,
        "context_window": "1M",
    },
    "gemini-pro": {
        "model_id": "gemini-2.5-pro-001",
        "provider": "vertex",
        "sdk": "vertexai.GenerativeModel",
        "thinking": True,  # thinking_config supported
        "temperature": True,
        "context_window": "1M",
    },
    # Vertex AI Model Garden — uses OpenAI-compatible SDK with ADC bearer token
    # IMPORTANT: GenerativeModel does NOT work for Model Garden third-party models
    "deepseek-r1": {
        "model_id": "deepseek-ai/deepseek-r1",
        "provider": "vertex_garden",
        "sdk": "openai.AsyncOpenAI",  # OpenAI SDK pointing to Vertex AI endpoint
        "base_url": "https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{region}/endpoints/openapi",
        "auth": "bearer_token",  # ADC token, NOT API key
        "thinking": True,
        "temperature": False,
        "context_window": "128k",
    },
    # Google AI Studio fallback — uses google-generativeai SDK, API key auth
    "gemini-flash-studio": {
        "model_id": "gemini-2.5-flash",
        "provider": "ai_studio",
        "sdk": "google.generativeai",
        "auth": "api_key",
        "thinking": False,
        "temperature": True,
        "context_window": "1M",
    },
    # Image generation — Vertex AI native
    "imagen-3": {
        "model_id": "imagegeneration@006",
        "provider": "vertex",
        "sdk": "vertexai.ImageGenerationModel",
        "type": "image",
    },
}
```

**DeepSeek-R1 access pattern** (via Vertex AI Model Garden):
```python
import google.auth
import google.auth.transport.requests

# Get ADC bearer token
credentials, project = google.auth.default()
credentials.refresh(google.auth.transport.requests.Request())
token = credentials.token

# Use OpenAI SDK pointing to Vertex AI endpoint
from openai import AsyncOpenAI
client = AsyncOpenAI(
    base_url=f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT}/locations/{REGION}/endpoints/openapi",
    api_key=token,  # Bearer token, refreshed hourly via TTL cache
)
response = await client.chat.completions.create(
    model="deepseek-ai/deepseek-r1",
    messages=[{"role": "user", "content": problem}],
)
```

### 3.5 Key Design Decisions (R1 Fix: Added structured output, health check)

1. **TTL-cached model clients** with jitter (per langgraph-patterns.md) — cache the client, not the model object. TTL=3600s for ADC token refresh.
2. **Timeout protection** via `asyncio.wait_for()` — 120s default, per-tool override
3. **Structured error handling** — `ResourceExhausted`, `PermissionDenied`, `InvalidArgument` → user-friendly messages
4. **Progress reporting** via `ctx.report_progress()` for long-running inference
5. **stderr logging only** — stdout reserved for JSON-RPC in stdio transport
6. **Structured tool outputs** (R1 fix) — `gcp_code_review` returns `findings[]` with `{file, line, severity, category, message, suggestion}`, not free text
7. **`diagnose_models` health tool** (R1 fix) — validates ADC, lists accessible models, checks quotas. Run on first session to catch auth issues.
8. **Provider-aware client factory** (R1 fix) — separate client creation paths for Vertex AI (ADC), Model Garden (OpenAI SDK + bearer), and AI Studio (API key). No mixing.
9. **Tool output schema versioning** (R2 fix) — every tool response includes `schema_version: "1.0"`. Breaking changes bump version. Documented in MCP server README.
10. **Cancellation support** (R2 fix) — long-running tools (gcp_research, gcp_deepseek_reason) check for MCP cancellation between API calls. `asyncio.wait_for()` with per-tool configurable timeout (default 120s, gcp_deepseek_reason: 300s).
11. **Secret redaction in logs** (R2 fix) — all prompts/code snippets logged to stderr are truncated to first 200 chars. Bearer tokens, API keys never logged. Explicit "what leaves the machine" contract: user prompts go to Vertex AI/AI Studio only, never to third-party unless explicitly configured.
12. **Supply chain controls** (R2 fix) — MCP packages from Google's `googleapis` org (verified: `npm view @google-cloud/gcloud-mcp repository.url` → `github.com/googleapis/gcloud-mcp`). Post-MVP: vendor source, `npm audit` in CI, Docker isolation with restricted service accounts.
13. **Schema compatibility policy** (R3 fix) — Tool output schemas follow semver. `schema_version: "1.x"` is backward-compatible (additive fields only). Breaking changes = `"2.0"` with 2-week deprecation window. Validation tests in `tests/test_tool_schemas.py`.
14. **Troubleshooting guide** (R3 fix) — README includes FAQ:
    - "ADC expired" → `gcloud auth application-default login`
    - "Model not found" → check Vertex AI API enabled, correct region
    - "MCP server won't start" → check venv, `pip install -e .`, stderr logs
    - "Hooks not firing" → check `chmod +x`, verify settings.json hook paths
    - "rsync not found" → install.sh falls back to `cp -r`, or install rsync
15. **Post-MVP hardening roadmap** (R3 fix) — documented as non-blocking items:
    - SLSA provenance for supply chain
    - Load/stress testing for Cloud Run
    - IAM role auditing automation
    - Regex-based secret redaction (replace truncation)
    - Subprocess kill + cleanup on cancellation
    - Native OS dependency coverage in lockfiles
    - Docker isolation for MCP servers with restricted SAs

---

## 4. MCP Server Configuration

### 4.1 Always-Active MCP Servers

| Server | Command | Purpose |
|--------|---------|---------|
| `memory` | Built-in | Cross-session persistence |
| `gcp-ai-foundry` | `$HOME/.claude/mcp-servers/multi-provider-ai/start-with-adc.sh` | Multi-model AI gateway |
| `grok` | `$HOME/.claude/mcp-servers/grok/start-with-env.sh` | Grok 4 (social, code, reasoning) |

### 4.2 Lazy-Loaded MCP Servers (R1 Fix: Pinned versions, verified packages)

All npm packages verified on npmjs.com as of 2026-02-27. Versions pinned for reproducibility.

| Server | Command | Version | Purpose |
|--------|---------|---------|---------|
| `perplexity` | Built-in | — | Web research (deep) |
| `gemini` | `{CLAUDE_HOME}/mcp-servers/gemini-mcp/start.sh` | — | Gemini advanced (vision, image edit) |
| `context7` | `npx -y @upstash/context7-mcp@latest` | latest | Library documentation |
| `docs-langchain` | `npx -y mcp-remote https://docs.langchain.com/mcp` | — | LangChain/LangGraph/LangSmith docs (verified MCP endpoint) |
| `playwright` | `{CLAUDE_HOME}/mcp-servers/playwright-cdp/start-browser-mcp.sh` | — | Browser automation |
| `gcloud` | `npx -y @google-cloud/gcloud-mcp@0.5.3` | 0.5.3 | All gcloud CLI commands (verified: googleapis/gcloud-mcp) |
| `gcp-observability` | `npx -y @google-cloud/observability-mcp@0.2.3` | 0.2.3 | Logs, metrics, traces (verified: googleapis/gcloud-mcp monorepo) |
| `cloud-run` | `npx -y @google-cloud/cloud-run-mcp@1.9.0` | 1.9.0 | Cloud Run management (verified: GoogleCloudPlatform/cloud-run-mcp) |
| `claude-mermaid` | `npx claude-mermaid` | latest | Live Mermaid preview |
| `superdesign` | Built-in | — | Image design |

Note: `{CLAUDE_HOME}` is replaced with absolute path by install.sh (e.g., `/home/user/.claude`). Never use `$HOME` in settings.json — Claude Code does not expand environment variables in JSON config.

### 4.3 Dropped MCP Servers (Not Needed)

| Server | Reason |
|--------|--------|
| `azure-ai-foundry` | Replaced by `gcp-ai-foundry` |
| `elevenlabs-creative` | Add back only if voice features needed |
| `lunarcrush` | Crypto domain — not relevant to casino |

---

## 5. install.sh Bootstrap Design (R1 Fix: Copy mode, absolute paths, venv)

### 5.1 Requirements

- **Cross-platform**: WSL2 (Ubuntu/Debian) and macOS (Intel/Apple Silicon)
- **Idempotent**: Safe to run multiple times
- **Non-destructive**: Backs up existing `~/.claude/` content before overwriting
- **Validates dependencies**: gcloud, Python 3.12+, Node.js 18+, Docker (warns, doesn't hard-fail on Docker)
- **Default: COPY mode** (rsync). Optional `--link` flag for developer workflow (symlinks).

### 5.2 Installation Flow (R1 Fix: Copy default, venv, absolute paths)

```
1. Detect platform (Linux/macOS) via `uname -s`
2. Parse flags: --link (symlink mode), --force (skip backup)
3. Check prerequisites:
   - REQUIRED: python3 (3.12+), node (18+)
   - RECOMMENDED: gcloud, docker (warn if missing, don't fail)
4. Backup existing ~/.claude/ → ~/.claude.backup.{timestamp}
5. Create ~/.claude/ directory structure
6. Install content (COPY mode by default, SYMLINK if --link):
   - Copy/link directories: rules/, hooks/, agents/, docs/, scripts/, configs/, templates/, themes/
   - Copy/link individual skills (preserves local additions in COPY mode)
   - Copy core files: CLAUDE.md, MEMORY.md
7. Set executable permissions: chmod +x on all hooks/*.sh, scripts/*.sh
8. Generate settings.json from template:
   - Replace {CLAUDE_HOME} with ABSOLUTE path (e.g., /home/user/.claude)
   - Replace {KIT_DIR} with absolute path to cloned repo
   - Never use $HOME, ~, or env vars in the generated JSON
9. Create Python venv for custom MCP server:
   - python3 -m venv ~/.claude/mcp-venvs/multi-provider-ai
   - ~/.claude/mcp-venvs/multi-provider-ai/bin/pip install -e {KIT_DIR}/mcp-servers/multi-provider-ai
   - MCP launcher references this venv's Python explicitly
10. Configure platform-specific settings:
    - Linux: notification via notify-send
    - macOS: notification via osascript, Chrome for Playwright
11. Verify gcloud CLI and ADC (if gcloud present)
12. Print setup summary, next steps, and any warnings
```

### 5.3 Cross-Platform Handling

| Component | WSL2/Linux | macOS | Notes |
|-----------|-----------|-------|-------|
| Notifications | `notify-send` | `osascript -e 'display notification'` | Auto-detected in hook |
| Browser (Playwright) | Edge + WSLg CDPEndpoint | Chrome + CDP | Browser auto-detected |
| Shell | bash (default) | zsh (default) | All scripts use `#!/usr/bin/env bash` |
| Home dir | `/home/username` | `/Users/username` | All paths absolute, set by install.sh |
| Package manager | apt | brew | install.sh doesn't install — just validates |
| Python | `python3` / `pip3` | `python3` / `pip3` (via brew) | Venv isolates MCP deps |
| gcloud install | Manual / snap | Manual / brew | Documented, not automated |
| sed | GNU sed | BSD sed | install.sh uses `sed` portably (no `-i ''` vs `-i`) |
| bash version | 5.x | 3.2 (default) | Scripts avoid bash 4+ features |

### 5.4 Path Handling (R1 Fix: Absolute paths, no env vars)

**CRITICAL**: Claude Code does NOT expand `$HOME`, `~`, or `${env:...}` in `settings.json` paths.

All paths in settings.json are written as **absolute paths** by install.sh at generation time:
```json
{
  "mcpServers": {
    "gcp-ai-foundry": {
      "command": "/home/jdoe/.claude/mcp-venvs/multi-provider-ai/bin/python",
      "args": ["-m", "gcp_ai_foundry.server"]
    }
  }
}
```

For hooks and scripts that run in bash, `$HOME` expansion works normally since bash processes it. But JSON config files must contain literal absolute paths.

### 5.5 Copy vs Symlink Modes

| Mode | Command | Pros | Cons |
|------|---------|------|------|
| **Copy** (default) | `./install.sh` | Reliable, no path confusion, survives repo deletion | Must re-run install.sh to update |
| **Symlink** (dev) | `./install.sh --link` | Instant updates, edit-in-place | May confuse file watchers, breaks if repo moves |

Copy mode uses `rsync -a --delete` for directories (with `cp -r` fallback if rsync not available) and `cp` for files. Symlink mode uses `ln -sf` with absolute paths.

### 5.6 Dependency Lockfiles (R2 Fix)

**Python**: `mcp-servers/multi-provider-ai/requirements.lock` generated by `pip-compile --generate-hashes`. The venv installs from this lockfile, not pyproject.toml directly.

**npm**: Not applicable — we use `npx` with pinned versions for Google MCP packages. No local node_modules except for the kit's own MCP servers.

### 5.7 Post-Install Validation (R2 Fix)

install.sh runs automatic validation after setup:
```
1. Verify all directories exist in ~/.claude/
2. Verify settings.json is valid JSON (python3 -m json.tool)
3. Verify custom MCP venv works (python3 -c "import gcp_ai_foundry")
4. Verify hooks are executable (test -x on each .sh/.py file)
5. Print summary: ✓ installed, ✗ missing, ⚠ warnings
```

---

## 6. CLAUDE.md Design (Master Config)

### 6.1 Sections to Keep (Universal)

- Identity ("Senior full-stack developer. Direct, concise, actionable.")
- 13 Hard Rules (Rule 8 changes: "Azure DevOps only" → "GitHub only")
- Bug Fix Protocol
- Sub-Agent Output Contract
- Opus 4.6 Features
- Mode Selection table
- Agent Teams architecture and rules
- Hook Reference (adapted hook names)
- On-Demand Docs triggers
- Codebase Search Strategy

### 6.2 Sections to Rewrite

| Section | Current | New |
|---------|---------|-----|
| Project Map | 8 projects | 1 project: Hey Seven |
| Post-Deploy Verification | Azure pipeline + `az` CLI | Cloud Build + `gcloud` CLI |
| MCP Quick Reference | `azure_*` tools | `gcp_*` tools |
| MCP Servers table | Azure-centric | GCP-native |
| Production Apps | Azure Static Web Apps URLs | Cloud Run URLs |
| Active Agents | azure-compliance agent | gcp-specialist agent |
| Git Remote rules | Azure DevOps | GitHub |
| DB Safety triggers | PostgreSQL + `az` | Firestore + `gcloud` |

### 6.3 Sections to Add

- GCP Infrastructure map (project ID, region, service accounts)
- Cloud Run deployment patterns
- Secret Manager access patterns
- Firestore safety rules (no information_schema — use gcloud indexes)
- ADC credential management
- Cost monitoring via BigQuery billing export

---

## 7. Rules Adaptation Details

### 7.1 gcp-deploy.md (NEW — replaces azure-deploy.md)

Key content:
- Git remote: GitHub ONLY (never Azure DevOps)
- Commit format: same (`<type>(<scope>): <description>`)
- Post-push: Cloud Build trigger verification via `gcloud builds list`
- GCP infrastructure map: project ID, region, service accounts, KMS key
- Deploy commands: `gcloud run deploy` with `--no-traffic` → canary → rollout
- Rollback: `gcloud run services update-traffic --to-revisions=PREV=100`
- Version assertion: health endpoint must return deployed version
- Secret safety: `gcloud secrets`, never hardcode, `--set-secrets` flag
- Cost: Budget alerts via `gcloud billing budgets create`
- End-of-session deploy check: all modified files committed + pushed

### 7.2 gcp-safety.md (NEW — replaces db-safety.md)

Key content:
- Firestore safety: export before destructive ops, index verification
- No information_schema — use `gcloud firestore indexes composite list`
- Secret Manager: version pinning in prod, `latest` only in dev
- IAM: least-privilege service accounts, no project-wide editor role
- Cost control: billing export to BigQuery, budget alerts, label all resources
- ADC: validate before every `gcloud` command (hook enforced)
- VPC: don't expose Redis Memorystore publicly

### 7.3 code-quality.md (Adapted)

Changes:
- Remove: Azure-specific Key Vault patterns, `az keyvault` commands
- Remove: PostgreSQL `information_schema` queries
- Add: Firestore index checks
- Change: "Azure DevOps" → "GitHub" in commit/push references
- Change: `az pipelines` → `gcloud builds` in pipeline verification
- Keep: All universal patterns (async migration, rename propagation, SRP, etc.)

---

## 8. New GCP Hooks Design (R1 Fix: Correct Cloud Build behavior)

### 8.1 cloud-build-gate.sh (replaces deploy-gate.sh)

**Event**: PreToolUse (Bash)
**Trigger**: Commands containing `curl` to Cloud Run URLs after a `gcloud run deploy`

**Logic** (R1 Fix: `gcloud builds submit` is synchronous by default — blocks until complete):
1. Track when the last `gcloud run deploy` completed (write timestamp to /tmp/.gcp-last-deploy)
2. If a `curl` to a Cloud Run URL happens within 30s of deploy completion, warn that the new revision may still be starting
3. Block premature verification only if deploy exit code was non-zero
4. For `gcloud builds submit` (synchronous): no gate needed — it blocks and returns exit code

**Note**: Unlike Azure DevOps async pipelines, `gcloud builds submit` blocks your terminal and streams logs. The gate is only needed for the Cloud Run cold-start delay AFTER a successful build+deploy, not for waiting on the build itself.

### 8.2 gcp-adc-check.sh (NEW)

**Event**: PreToolUse (Bash)
**Trigger**: Commands containing `gcloud` (first invocation per session only, cached result)

**Logic**:
1. Verify `gcloud auth application-default print-access-token` succeeds (ADC valid)
2. Verify correct project: `gcloud config get-value project`
3. Verify quota project matches core project (R1 fix: check both)
4. Compare against expected project from `.gcp-project` file in project root
5. Warn (not block) on project mismatch — different project might be intentional
6. Cache result for session to avoid repeated slow `gcloud` calls

### 8.3 periodic-commit-check.sh (Adapted)

**Event**: Stop
**Change**: Replace `git push azure` with `git push origin` (GitHub)

### 8.4 task-completed-verify.sh (Adapted)

**Event**: TaskCompleted
**Change**: Replace `az pipelines` with `gcloud builds list` for deploy verification

---

## 9. New GCP Skills Design

### 9.1 /gcp-deploy (R1 Fix: Correct CLI, no manual revision cleanup)

**Purpose**: Zero-downtime Cloud Run deployment with canary testing.

**Steps**:
1. Verify ADC and correct project (`gcloud config get-value project`)
2. Build Docker image: `docker build -t {AR_IMAGE}:{VERSION} .`
3. Push to Artifact Registry: `docker push {AR_IMAGE}:{VERSION}`
4. Deploy with `--no-traffic --tag=canary`: (R1 fix: tag during deploy for clean canary URL)
   ```bash
   gcloud run deploy SERVICE --image={AR_IMAGE}:{VERSION} \
     --no-traffic --tag=canary --region=REGION
   ```
5. Smoke test canary URL (R3 fix: use OIDC identity token, not ADC bearer):
   ```bash
   # Cloud Run requires OIDC identity token with audience = service URL
   CANARY_URL="https://canary---$(gcloud run services describe SERVICE \
     --region=REGION --format='value(status.url)' | sed 's|https://||')"
   TOKEN=$(gcloud auth print-identity-token --audiences="$CANARY_URL")
   curl -H "Authorization: Bearer $TOKEN" "$CANARY_URL/health"
   ```
6. Gradual traffic shift:
   ```bash
   # Get new revision name
   NEW_REV=$(gcloud run revisions list --service=SERVICE --region=REGION \
     --sort-by=~metadata.creationTimestamp --limit=1 --format="value(name)")
   # 10% → 50% → 100%
   gcloud run services update-traffic SERVICE --to-revisions=$NEW_REV=10 --region=REGION
   # Monitor error rate via Cloud Monitoring...
   gcloud run services update-traffic SERVICE --to-revisions=$NEW_REV=100 --region=REGION
   ```
7. Verify version assertion: health endpoint returns deployed version
8. (R1 fix: NO manual revision cleanup — Cloud Run auto-garbage-collects, keeping up to 1000)

### 9.2 /gcp-rollback (R1 Fix: Programmatic revision lookup)

**Purpose**: Instant rollback to previous stable Cloud Run revision.

**Steps**:
1. List recent revisions:
   ```bash
   gcloud run revisions list --service=SERVICE --region=REGION \
     --sort-by=~metadata.creationTimestamp --limit=5 --format="table(name,active,creationTimestamp)"
   ```
2. Identify previous revision (R1 fix: programmatic lookup, NOT "PREV" keyword):
   ```bash
   PREV_REV=$(gcloud run revisions list --service=SERVICE --region=REGION \
     --sort-by=~metadata.creationTimestamp --limit=2 --format="value(name)" | tail -1)
   ```
3. Shift 100% traffic: `gcloud run services update-traffic SERVICE --to-revisions=$PREV_REV=100 --region=REGION`
4. Verify health endpoint returns previous version
5. (Do NOT delete bad revision — keep it for debugging, Cloud Run handles cleanup)

### 9.3 /gcp-status

**Purpose**: Comprehensive project health dashboard.

**Output**:
- Cloud Run: active revision, traffic split, instance count, error rate
- Cloud Build: last 3 builds status
- Secret Manager: rotation status for each secret
- Monitoring: active alerts, recent incidents
- Artifact Registry: image count, latest tag
- Cost: today's spend vs budget (if BigQuery billing enabled)

### 9.4 /gcp-logs

**Purpose**: Structured log analysis for debugging.

**Capabilities**:
- Tail real-time logs: `gcloud run services logs tail`
- Filter by severity: ERROR, WARNING
- Parse structured JSON payloads
- Correlate with revision deployment times
- Extract request IDs for tracing

---

## 10. Cross-Platform Compatibility Matrix

| Component | WSL2 | macOS | Notes |
|-----------|------|-------|-------|
| install.sh | bash | bash/zsh | Platform detection via `uname -s` |
| notification.sh | notify-send | osascript | Auto-detected in hook |
| Playwright MCP | Edge + WSLg CDPEndpoint | Chrome + CDP | Browser auto-detected |
| gcloud CLI | apt/snap | brew | install.sh validates |
| Python 3.12 | apt / pyenv | brew / pyenv | install.sh validates |
| Node.js 18+ | nvm / apt | nvm / brew | install.sh validates |
| Docker | Docker Desktop WSL2 | Docker Desktop macOS | install.sh validates |
| D2 diagrams | `~/.local/bin/d2` | brew install d2 | Manual install noted |
| File paths | /home/user | /Users/user | All use $HOME |
| Line endings | LF | LF | .gitattributes enforced |
| Shell default | bash 5.x | zsh (bash 3.2 via brew) | Hooks use `#!/usr/bin/env bash`, avoid bash 4+ features |

---

## 11. Migration Checklist (What Transfers, What Doesn't)

### 11.1 Transfers As-Is

- All LangGraph patterns (langgraph-patterns.md) — 100% universal
- All RAG production patterns (rag-production.md) — 100% universal
- Hostile review protocol — universal process, model names unchanged
- Diagramming rules and tools — D2, Mermaid, React Flow
- Orchestration patterns — Planner → Implementer → Verifier
- Agent team architecture — team creation, task management, delegation
- Code quality rules — async safety, rename propagation, SRP, etc.
- Memory MCP entities — cross-session architectural decisions

### 11.2 Transfers With Adaptation

| Component | Change Required |
|-----------|----------------|
| 24 custom skills | MCP tool name references: `azure_*` → `gcp_*` |
| 9 agent definitions | MCP tool name references in tool lists |
| 22 hooks | Path: `/home/odedbe` → `$HOME`, Azure → GCP commands |
| 16 scripts | Path fixes, Azure → GCP where applicable |
| 11 rules | Azure patterns → GCP patterns, same structure |
| 12 docs | Azure Functions → Cloud Run, DevOps → Cloud Build |
| CLAUDE.md | Full rewrite of cloud-specific sections |

### 11.3 Does NOT Transfer

- Project configs for: Sentimark, QC Analyzer, CS Agents, Training, Compliance, Phone Spam, Khaleeji
- Azure DevOps repository configurations
- Azure Key Vault secret names and access patterns
- Azure-specific compliance rules (azure-compliance-rules.json)
- Session history, file history, debug logs, shell snapshots
- `.credentials.json` (machine-specific auth)
- `settings.local.json` (machine-specific overrides)

---

## 12. 72-Hour MVP Sprint Plan (R1 Fix: Pre-arrival checklist, DNS, credentials)

### 12.0 Pre-Arrival Checklist (CRITICAL — Do BEFORE Day 1)

These items require Hey Seven company action and may take days. Start immediately:

- [ ] **GCP Project access**: Confirm project ID, get `roles/editor` or specific roles
- [ ] **IAM Roles**: Request — Cloud Run Admin, Secret Manager Accessor, Artifact Registry Writer, Cloud Build Editor, Firestore User, Monitoring Viewer
- [ ] **Billing**: Confirm billing is enabled on the project
- [ ] **API Keys handoff**: Get from Hey Seven team:
  - GOOGLE_API_KEY (Gemini)
  - API_KEY (app auth)
  - CMS_WEBHOOK_SECRET (if Sheets CMS needed)
  - TELNYX_API_KEY + TELNYX_PUBLIC_KEY + TELNYX_MESSAGING_PROFILE_ID (if SMS needed)
  - LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY (if observability needed)
- [ ] **Machine setup**: Confirm OS (WSL2 / macOS), ensure Python 3.12, Node 18+, Docker available
- [ ] **GitHub access**: Confirm access to hey-seven repo
- [ ] **Domain**: If custom domain needed, start DNS verification early (propagation takes 24-48h)
- [ ] **Redis**: Confirm if Cloud Memorystore already provisioned (or if using InMemoryBackend for MVP)

### 12.1 Pre-Arrival (Build the Kit)

1. Build claude-code-kit repo with all adapted content
2. Build multi-provider-ai MCP server with tests
3. Write all new GCP skills, hooks, docs
4. Test install.sh on clean WSL2 environment (docker container or fresh user)
5. Validate with 4-model hostile review (this document)
6. Push to GitHub: github.com/Oded-Ben-Yair/claude-code-kit

### 12.2 Day 1 (Setup — Target: 4 hours)

1. Clone claude-code-kit: `git clone ... && cd claude-code-kit && ./install.sh`
2. Install gcloud CLI if not present (brew/apt)
3. Authenticate: `gcloud auth login` + `gcloud auth application-default login`
4. Set project: `gcloud config set project {PROJECT_ID}`
5. Verify MCP servers start: open Claude Code, check status
6. Clone hey-seven: `git clone ... && cd hey-seven`
7. Set up Python: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt`
8. Run tests: `pytest tests/ -v --tb=short -x` (expect 2537 pass)
9. Create `.env` from `.env.example`, populate with keys from pre-arrival checklist
10. Run `docker-compose up` — verify local dev works
11. **Checkpoint**: Local dev environment fully working

### 12.3 Day 1-2 (Infrastructure — Target: 8 hours)

1. Enable GCP APIs:
   ```bash
   gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
     secretmanager.googleapis.com firestore.googleapis.com \
     monitoring.googleapis.com cloudbuild.googleapis.com \
     redis.googleapis.com
   ```
2. Create secrets in Secret Manager (from keys in .env)
3. Create Artifact Registry repo: `gcloud artifacts repositories create hey-seven --repository-format=docker --location=us-central1`
4. Set up Cloud Build trigger for GitHub repo
5. Provision Redis Memorystore (if not exists) + VPC connector
6. First deploy: `git push origin main` → verify Cloud Build triggers → watch logs
7. (R1 fix) If DNS/custom domain needed: `gcloud run domain-mappings create --service=SERVICE --domain=DOMAIN`
8. (R2 fix) **Day 1-2 Smoke Test Checklist**:
   - [ ] `gcloud run services describe hey-seven-agent --region=us-central1` → shows ACTIVE
   - [ ] `curl $SERVICE_URL/health` → 200 + correct version
   - [ ] `gcloud builds list --limit=1` → SUCCESS
   - [ ] `gcloud secrets versions access latest --secret=hey-seven-api-key` → returns value
   - [ ] `gcloud firestore indexes composite list` → indexes exist

### 12.4 Day 2-3 (MVP Live — Target: 12 hours)

**Required GCP APIs** (R2 fix: explicit list):
```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  firestore.googleapis.com \
  monitoring.googleapis.com \
  cloudbuild.googleapis.com \
  redis.googleapis.com \
  cloudkms.googleapis.com \
  sheets.googleapis.com \
  aiplatform.googleapis.com
```

1. Verify Cloud Run service healthy: `curl https://SERVICE-URL/health`
2. Version assertion: health response includes correct version
3. Test SSE streaming end-to-end (browser + curl)
4. Verify Firestore checkpointer: multi-turn conversation persists
5. Verify RAG pipeline: ingest knowledge base, query, validate relevance
6. (R1 fix) Set up Langfuse observability (if keys available)
7. (R1 fix) Set up budget alert: `gcloud billing budgets create` (or console if permissions lacking)
8. Demo to stakeholders with real casino data
9. **Checkpoint**: MVP live, monitoring enabled, stakeholder approval

---

## 13. Risk Assessment (R1 Fix: Added pre-arrival mitigations)

| Risk | Impact | Mitigation | Pre-Arrival? |
|------|--------|------------|-------------|
| GCP project not provisioned | **HIGH** | Pre-arrival checklist item. Confirm access + billing before start. | YES |
| Missing GCP IAM permissions | **HIGH** | Pre-request: Cloud Run Admin, Secret Manager Accessor, AR Writer, Cloud Build Editor, Firestore User | YES |
| External API keys not handed off | **HIGH** | Credential sourcing checklist. Get Telnyx, Langfuse, Sheets keys from team. | YES |
| DeepSeek-R1 not available on Vertex AI Model Garden | Medium | Fall back to Gemini Pro thinking=high. Ship Gemini-only first. | No |
| ADC token expiry during long sessions | Medium | TTL-cached clients with 1-hour refresh + jitter | No |
| macOS Playwright differences | Medium | Chrome CDP fallback, tested on both platforms | No |
| macOS bash 3.2 incompatibility | Medium | All scripts use `#!/usr/bin/env bash`, avoid bash 4+ features | No |
| FastMCP breaking changes | Low | Pin exact version in pyproject.toml | No |
| DNS propagation for custom domain | Medium | Start DNS verification on day 1, use *.run.app for initial demo | No |
| gcloud CLI version differences | Low | Document minimum version, install.sh validates | No |
| Cloud Build quota limits | Low | Manual `gcloud builds submit` as fallback | No |
| Redis not provisioned | Medium | Use InMemoryBackend for MVP, migrate to Redis later | No |

---

## 14. Success Metrics

| Metric | Target |
|--------|--------|
| Time from clone to first Claude Code session | < 30 minutes |
| MCP servers all healthy | 100% on first start |
| All hooks fire correctly | 0 errors in first session |
| Hey Seven tests pass locally | 2537 tests, 0 failures |
| First Cloud Run deployment | Within 8 hours of starting |
| MVP demo-ready | Within 72 hours |
| Multi-model hostile review score | 95/100 minimum, 0 CRITICALs |

---

**End of Design Document**
