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
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)
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
# FirestoreSaver for GCP-native persistence
from langgraph.checkpoint.firestore import FirestoreSaver
from google.cloud import firestore

db = firestore.Client(project="hey-seven-prod")
checkpointer = FirestoreSaver(db)

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
```python
from langchain_google_genai import ChatGoogleGenerativeAI

# Primary: Fast, cost-effective
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Complex reasoning: When needed
pro_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
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
| Gemini 2.5 Flash (primary) | ~$300 (1M tokens/day) |
| Gemini 2.5 Pro (complex) | ~$200 (100K tokens/day) |
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
