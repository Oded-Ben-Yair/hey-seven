# LangGraph Latest Intelligence (Feb 2026)

## Sources
- Grok X/Twitter search (40+ posts, Dec 2025 - Feb 2026)
- Perplexity deep research (60 citations)
- Context7 official docs
- PyPI release history

---

## 1. Version Status

| Version | Date | Key Changes |
|---------|------|-------------|
| **1.0.8** | Feb 6, 2026 | Fixed Pydantic messages double streaming, shallow-copied futures, connection pool lock optimization |
| 1.0.7 | Jan 22, 2026 | Resolved aiosqlite breaking change. Dynamic tool calling via `tool` override |
| 1.0.6 | Jan 12, 2026 | Fixed default base cache, recursion limit, compile-time checkpointer validation |
| 1.0.0 | Oct 17, 2025 | **Major release**. Stability commitment: zero breaking changes until v2.0 |
| **0.2.60** | ~Aug 2025 | **Our pinned version** — stable, tested, compatible with our boilerplate |

### Ecosystem Versions (Feb 2026)
| Package | Latest |
|---------|--------|
| langgraph | 1.0.8 |
| langgraph-sdk | 0.3.4 |
| langgraph-checkpoint-sqlite | 3.0.3 |
| langchain-core | 1.2.11 |
| langchain-google-genai | latest (supports Gemini 2.5 + 3.0) |

### Breaking Changes Since 0.2.60
- Python 3.9 dropped (requires 3.10+)
- `state_modifier` removed — use `prompt` parameter
- `create_react_agent` deprecated in `langgraph.prebuilt` — migrating to `langchain.agents.create_agent`
- `MemorySaver` renamed to `InMemorySaver` (both still work)
- No breaking changes to core graph primitives (StateGraph, nodes, edges remain stable)

---

## 2. `create_react_agent` Migration Path

### Deprecation Timeline
- **~v0.1.9**: `state_modifier` deprecated, `prompt` introduced
- **v0.2.46 (JS)**: JS unified `stateModifier` to `prompt`
- **v1.0.0**: `state_modifier` fully removed. `prompt` is standard
- **v1.0+ migration**: `prompt` becomes `system_prompt` (string only) in `create_agent`

### Migration Table
| Aspect | Old (Our Code) | Deprecated (0.2.70+) | New (1.0+) |
|--------|----------------|----------------------|------------|
| Import | `langgraph.prebuilt.create_react_agent` | same | `langchain.agents.create_agent` |
| Prompt | `state_modifier=` | `prompt=` (str/SystemMessage/Callable) | `system_prompt=` (string only) |
| Dynamic | Callable in state_modifier | Callable in prompt | `@dynamic_prompt` middleware |

### Our Strategy
Our pinned `langgraph==0.2.60` uses `state_modifier=`. This is correct for our version. If assignment requires 1.0+ features, bump version and use `prompt=` or `system_prompt=`.

---

## 3. Middleware System (New in v1.0)

Composable hooks replacing ad-hoc state modification patterns.

| Hook | When | Use Cases |
|------|------|-----------|
| `before_model` | Before LLM call | Trim messages, load context, auth, dynamic routing |
| `after_model` | After LLM response | HITL approvals, guardrails, moderation |
| `before_agent` | Pre-agent input | Input validation, file loading |
| `after_agent` | Final post-agent | Logging, persistence, cleanup |
| `wrap_model_call` | Wraps full invocation | Dynamic model swapping, caching |
| `wrap_tool_call` | Intercepts tool calls | Retries, error handling, rate limiting |

### Casino Host Use Cases
- `before_model`: Inject player context, check compliance flags
- `after_model`: Validate comp calculations, enforce regulatory limits
- `wrap_tool_call`: Rate-limit CRM API calls, retry on timeout

---

## 4. New APIs (v1.0)

### Functional API (`@entrypoint`)
Alternative to StateGraph for simpler workflows:
```python
from langgraph.func import entrypoint, task

@entrypoint(checkpointer=checkpointer)
def my_agent(inputs, *, previous, store, writer, config):
    # previous = state from last invocation
    # store = cross-thread memory
    # writer = streaming output
    pass
```

### Command API (Multi-Agent Handoffs)
```python
from langgraph.types import Command

def agent_node(state):
    return Command(
        update={"messages": [response]},
        goto="next_agent"  # Dynamic routing
    )
```

### Send API (Dynamic Fan-Out)
```python
from langgraph.types import Send

def route_to_workers(state):
    return [Send("worker", {"task": t}) for t in state["tasks"]]
```

### Store API (Cross-Thread Memory)
```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(index={"embed": embeddings_model})
# Stores persist across threads — player preferences, learned patterns
```

---

## 5. Streaming Modes

| Mode | What Streams | Best For |
|------|-------------|----------|
| `"values"` | Full state after each node | Dashboards, debugging |
| `"updates"` | State deltas only | Production UIs (efficient) |
| `"messages"` | Chat tokens as generated | Chatbot UIs, token-by-token |
| `"custom"` | Arbitrary data via writer | Progress indicators |
| `"debug"` | Full execution traces | Development |

Multiple modes can be combined: `stream_mode=["updates", "messages"]`

---

## 6. Production Gotchas (Community-Reported)

### Checkpointer Issues (Critical)
| Checkpointer | Status | Issues |
|--------------|--------|--------|
| `MemorySaver` / `InMemorySaver` | **Dev only** | Unbounded RAM growth, no persistence |
| `SqliteSaver` | Local/single-instance | File locking in clusters; aiosqlite breaking change fixed in 3.0.3 |
| `PostgresSaver` | Production-ready | Connection pool race conditions (`PoolClosed`) in FastAPI — use PgBouncer |
| `AsyncPostgresSaver` | Async production | **Hangs** with sync `invoke()` — always match sync/async |
| `FirestoreSaver` | Community package | 1MB doc limit, NOT official LangGraph |

### Security
- **CVE-2025-64439**: Deserialization RCE in checkpoint. Fixed in checkpoint v3.0+
- **CVE-2025-67644**: SQL injection in SQLite checkpointer. Update to latest.

### Other Gotchas
- `langgraph dev` ignores custom checkpointers (forces in-memory)
- ~50 rows per execution with checkpointing is normal
- Non-JSON state objects fail checkpointing — use RunnableConfig for non-serializable
- Prune old threads periodically (unbounded checkpoint growth)

---

## 7. Gemini Integration

### Critical Limitation
**Gemini 2.5 cannot combine `bind_tools()` + `with_structured_output()` in a single call.**

#### Workaround 1: Bind Pydantic as Tool
```python
class ComplianceResult(BaseModel):
    status: str
    flags: list[str]

tools = [check_player, calculate_comp, ComplianceResult]
llm = llm.bind_tools(tools, parallel_tool_calling=False)
# Parse ComplianceResult "tool call" args as structured output
```

#### Workaround 2: Two-Node Graph
```python
# Node 1: Tool-calling agent (tools only)
# Node 2: Structured output (with_structured_output only)
# ~2x cost but strict schema compliance
```

### Tips
- Temperature=0 for tool calling consistency
- `thinking_level` for complex reasoning
- Reference repo: google-gemini/gemini-fullstack-langgraph-quickstart
- Phil Schmid's LangChain+Gemini cheatsheet is community gold

---

## 8. Human-in-the-Loop (Interrupts)

```python
from langgraph.types import interrupt, Command

def compliance_review(state):
    if state["comp_value"] > 5000:
        human_decision = interrupt({
            "type": "comp_approval",
            "player": state["player_id"],
            "value": state["comp_value"]
        })
        return Command(
            update={"approval": human_decision},
            goto="finalize" if human_decision == "approved" else END
        )
```

Key points:
- Interrupts pause entire nodes (break complex nodes into smaller ones)
- State saved to checkpointer, resumes exactly where paused
- Use `Command(resume=value)` to provide human input

---

## 9. Deployment Options

| Option | Best For |
|--------|----------|
| LangGraph Cloud (SaaS) | Fastest to production |
| Self-Hosted (Docker/K8s) | Data sovereignty |
| BYOC | Enterprise, custom infra |
| Google Cloud Marketplace | GCP-native teams |
| Cloud Run + FastAPI | **Our approach** — full control, GCP-aligned |

---

## 10. Casino/Gaming Intelligence

**Zero public LangGraph implementations found in casino/gaming.**

- No evidence of adoption in iGaming or regulated gambling
- Casino industry uses certified RGS, not public AI experiments
- Hey Seven would be a FIRST MOVER in this space
- Closest regulated-industry examples: financial agents, legal contract analysis
- Potential use cases: compliance agents, player risk assessment, AML checks

---

## 11. Community Sentiment

### What Works
- v1.0 stability commitment (zero breaking changes until v2.0)
- Middleware system for clean separation of concerns
- Checkpointing + time-travel for debugging
- Streaming modes for flexible UIs

### What Hurts
- Steep learning curve vs plain LangChain
- MemorySaver memory leaks in production
- PostgresSaver + FastAPI connection pooling
- Gemini tools + structured output limitation
- Documentation gaps for advanced middleware

### Notable Production Deployments
- **Vodafone/Fastweb**: 90% correctness on 9.5M customers (telecom)
- **Monte Carlo Data**: Data observability agents
- **Tradestack**: WhatsApp MVP in 6 weeks (construction)

---

## Quick Decision Matrix

| If You Need... | Use |
|-----------------|-----|
| Simple chatbot | `create_react_agent` / `create_agent` with system prompt |
| Dynamic prompts | `before_model` middleware (v1.0+) |
| Durable execution | `PostgresSaver` checkpointer |
| Token-by-token UI | `stream_mode="messages"` |
| Progress indicators | `astream_events(version="v2")` |
| Gemini + tools + structured | Two-node graph or Pydantic-as-tool |
| Multi-agent orchestration | Subgraphs + supervisor pattern |
| Production observability | LangSmith (non-negotiable) |
| Cross-thread memory | InMemoryStore / persistent Store |
