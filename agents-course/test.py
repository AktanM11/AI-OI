import os
from dotenv import load_dotenv

from typing import TypedDict, Annotated

from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI

# =====================================================
# ENV
# =====================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found")

# =====================================================
# TOOLS
# =====================================================

@tool
def get_time() -> str:
    """Return current time."""

    print("TOOL CALLED: get_time")

    return "12:00"

# =====================================================
# LLM
# =====================================================

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-5.4-nano-2026-03-17"
)

tools = [get_time]

llm_with_tools = llm.bind_tools(tools)

# =====================================================
# STATE
# =====================================================

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# =====================================================
# ASSISTANT
# =====================================================

def assistant(state: AgentState):

    print("\n=== ASSISTANT NODE ===")

    response = llm_with_tools.invoke(
        state["messages"]
    )

    print("MODEL RESPONSE:")
    print(response)

    return {
        "messages": [response]
    }

# =====================================================
# GRAPH
# =====================================================

builder = StateGraph(AgentState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")

builder.add_conditional_edges(
    "assistant",
    tools_condition
)

builder.add_edge(
    "tools",
    "assistant"
)

graph = builder.compile()

# =====================================================
# TEST
# =====================================================

result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="What time is it right now?"
            )
        ]
    },
    config={
        "recursion_limit": 10
    }
)

print("\n========================")
print("FINAL ANSWER")
print("========================\n")

print(result["messages"][-1].content)