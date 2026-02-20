# LangGraph + GCP Technical Reference

## LangGraph 1.0 GA — Key Patterns

### StateGraph Architecture
```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    player_id: str | None
    context: dict

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")
graph.add_edge(START, "agent")
app = graph.compile(checkpointer=checkpointer)
```

### Tool Calling with ToolNode
```python
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

@tool
def check_player(player_id: str) -> dict:
    """Check player status and loyalty tier."""
    return {"tier": "Platinum", "adt": 5000}

tools = [check_player]
tool_node = ToolNode(tools)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")  # Or "gemini-3-flash" (Dec 2025)
model = model.bind_tools(tools)
```

### Conditional Routing
```python
def route_after_agent(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    if state.get("escalation_needed"):
        return "escalate"
    return "end"

graph.add_conditional_edges("agent", route_after_agent, {
    "tools": "tools",
    "escalate": "escalation_handler",
    "end": END
})
```

### Checkpointing with Firestore
```python
# FirestoreSaver — community package (NOT official LangGraph)
# Install: pip install langgraph-checkpoint-firestore
# Note: 1MB Firestore document size limit — long conversations will need truncation
from langgraph_checkpoint_firestore import FirestoreSaver
from google.cloud import firestore

db = firestore.Client(project="hey-seven-prod")
checkpointer = FirestoreSaver(db)
# Official LangGraph checkpointers: MemorySaver, PostgresSaver, SqliteSaver, CosmosDBSaver

# Compile with checkpointer
app = graph.compile(checkpointer=checkpointer)

# Invoke with thread_id for session persistence
config = {"configurable": {"thread_id": f"player_{player_id}"}}
result = app.invoke({"messages": [("user", message)]}, config=config)
```

### Human-in-the-Loop (Interrupt)
```python
# Interrupt before executing sensitive actions
graph.add_node("compliance_check", compliance_node)
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["compliance_check"]
)
```

### Fan-out/Fan-in Parallel
```python
# Parallel tool execution
graph.add_node("parallel_lookup", parallel_node)
# Fan-out to multiple tools, fan-in to collect results
```

## GCP Stack for Casino Host Agent

### Cloud Run Deployment
- Container-based, auto-scaling, pay-per-use
- Port 8080 (default)
- Min instances: 1 (avoid cold start for demo)
- Max instances: 10
- Memory: 2Gi (LLM client + embeddings)
- CPU: 2 cores

### Vertex AI (LLM Inference)

**Note**: Gemini 3 Pro (Nov 2025) and Gemini 3 Flash (Dec 2025) are now available. Hey Seven may have migrated from 2.5 to 3.0 — the patterns remain the same, only the model name string changes. Below shows 2.5 (from job posting era) with 3.0 alternatives.

```python
from langchain_google_genai import ChatGoogleGenerativeAI

# Primary: Fast, cost-effective
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Or "gemini-3-flash" (latest)
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Complex reasoning: When needed
pro_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",  # Or "gemini-3-pro" (latest)
    temperature=0.1,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
```

### Firestore (State + Checkpoints)
- Collection: `conversations/{thread_id}/checkpoints`
- Collection: `players/{player_id}` (player profiles)
- Collection: `comps/{comp_id}` (comp calculations)
- Real-time listeners for live dashboard updates

### Vertex AI Vector Search (RAG)
```python
from langchain_google_vertexai import VertexAIEmbeddings

embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
# Or for local dev: ChromaDB with same embeddings
```

### AlloyDB AI (Alternative: Combined Vector + Relational)
- PostgreSQL-compatible with vector extensions
- Player data + embeddings in same DB
- Good for complex queries (JOIN player history with vector search)

## Cost Estimates (100K monthly executions)

| Component | Monthly Cost |
|-----------|-------------|
| Gemini 2.5 Flash (primary) | ~$300 (1M tokens/day) — Gemini 3 pricing may differ |
| Gemini 2.5 Pro (complex) | ~$200 (100K tokens/day) — Gemini 3 pricing may differ |
| Cloud Run | ~$150 (2 instances avg) |
| Firestore | ~$100 (reads + writes) |
| Vertex AI Vector Search | ~$200 |
| **Total** | **~$950/month** |

## RAG Patterns for Casino Domain

### Agentic RAG
Agent decides when to retrieve — not every query needs RAG.
```python
@tool
def search_regulations(query: str) -> str:
    """Search gaming regulations database."""
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])
```

### Corrective RAG
Grade retrieved documents, rewrite query if quality is low.

### Self-RAG
Agent reflects on whether it needs more information.

## Production Patterns

### LangSmith Tracing
```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "..."
os.environ["LANGCHAIN_PROJECT"] = "hey-seven-casino-host"
```

### Error Handling
```python
from langchain_core.runnables import RunnableConfig

async def agent_node(state: AgentState, config: RunnableConfig):
    try:
        response = await model.ainvoke(state["messages"], config=config)
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return {"messages": [AIMessage(content="I'm having trouble processing that. Let me connect you with a host.")]}
```

### Rate Limiting
- Gemini 2.5 Flash: 1000 RPM free tier, 4000 RPM paid
- Implement exponential backoff
- Queue management for burst traffic

## Key API Differences (LangGraph 1.0 vs Earlier)

- Use `StateGraph` not `MessageGraph` (deprecated)
- Use `add_messages` annotation for message lists
- Use `ToolNode` from `langgraph.prebuilt`
- Use `tools_condition` for standard tool routing
- Checkpointers are passed to `compile()`, not `__init__()`
- Thread config via `{"configurable": {"thread_id": "..."}}`
- `create_react_agent` from `langgraph.prebuilt` handles standard agent loop (LLM -> tools -> LLM) out of the box
- LangGraph Platform / Agent Server handles checkpointing automatically (alternative to self-managed)

## Version Compatibility Notes (Feb 2026)

- **Latest stable**: LangGraph 1.0.6 (as of Feb 2026)
- **Our pinned version**: langgraph==0.2.60
- **`state_modifier` vs `prompt`**: In `create_react_agent`, the parameter to pass a system prompt was originally called `state_modifier`. It was aliased to `prompt` in a later version (likely ~0.2.70+). The latest 1.0.x uses `prompt=`. Our boilerplate uses `state_modifier=` for compatibility with 0.2.60.
- **`MemorySaver` vs `InMemorySaver`**: In newer LangGraph versions (1.0+), the in-memory checkpointer was renamed from `MemorySaver` to `InMemorySaver`. Both import from `langgraph.checkpoint.memory`.
- **Functional API**: LangGraph 1.0 introduced the `@entrypoint` decorator for a functional style (alternative to StateGraph). Uses injectable params like `previous`, `store`, `writer`, `config`.
- **InMemoryStore**: New long-term memory store (cross-thread) available alongside checkpointer (per-thread).
- **When to upgrade**: If the assignment requires features from 1.0+ (functional API, InMemoryStore, etc.), bump to `langgraph>=1.0.3`. Otherwise, 0.2.60 is stable and tested.

## Gemini 3 Models (Nov-Dec 2025)

Google released Gemini 3 Pro (November 2025) and Gemini 3 Flash (December 2025). These are now the latest production models on Vertex AI. For Hey Seven's purposes:
- Migration from 2.5 to 3.0 only requires changing the model name string
- LangChain integration via `langchain-google-genai` supports both generations
- Gemini 3 models have improved reasoning, tool calling, and structured output capabilities
