from langchain_core import messages
from langgraph.graph import message
import typing_extensions
from dotenv import load_dotenv
import os
import json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ===== TOOL AND MODEL SETUP =====
print("[INIT] Loading tools and model...")
tool = TavilySearch(max_results=2)
tools = [tool]
llm = init_chat_model("google_genai:gemini-2.0-flash")
print("[INIT] Tools and LLM initialized")

# ===== GRAPH STATE =====
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
llm_with_tools = llm.bind_tools(tools)

# ===== CHATBOT NODE =====
def chatbot(state: State):
    print("\n[CHATBOT NODE] Invoked")
    print("[CHATBOT NODE] Current state messages:", [m.content if hasattr(m, "content") else m for m in state["messages"]])
    result = llm_with_tools.invoke(state["messages"])
    print("[CHATBOT NODE] LLM response:", result.content)
    return {"messages": [result]}

graph_builder.add_node("chatbot", chatbot)
print("[GRAPH] ChatBot node added to graph")

# ===== TOOL NODE =====
class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: State) -> dict:
        print("\n[TOOL NODE] Invoked")
        if messages := state.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("[TOOL NODE] No messages found in input")

        print("[TOOL NODE] Last message tool_calls:", getattr(message, "tool_calls", None))
        outputs = []
        for tool_call in message.tool_calls:
            print(f"[TOOL NODE] Running tool: {tool_call['name']} with args: {tool_call['args']}")
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            print("[TOOL NODE] Tool result:", tool_result)
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        print("[TOOL NODE] Returning tool outputs to graph")
        return {"messages": outputs}  # ✅ Fixed typo from "messagse"

tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
print("[GRAPH] Tool node added to graph")

# ===== ROUTER =====
def route_tools(state: State):
    print("\n[ROUTER] Checking if tool is needed...")
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("[ROUTER] No messages found in state")

    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("[ROUTER] Tool call detected → Going to 'tools' node")
        return "tools"
    print("[ROUTER] No tool call → Ending cycle")
    return END

print("[GRAPH] Conditional edge defined")

# ===== GRAPH EDGES =====
graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "chatbot")  # After tool execution, return to chatbot
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()
print("[GRAPH] Graph compiled successfully")

# ===== STREAM FUNCTION =====
def stream_graph_updates(user_input: str):
    print("\n[STREAM] User input received:", user_input)
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        print("[STREAM] Event received from graph:", event)
        for value in event.values():
            if value and "messages" in value:
                print("Assistant:", value["messages"][-1].content)

# ===== MAIN LOOP =====
while True:
    try:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "q", "exit"]:
            print("[EXIT] Goodbye!")
            break
        stream_graph_updates(user_input)
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        user_input = "What do you know about LangGraph?"
        print("\n[FALLBACK] Using default question:", user_input)
        stream_graph_updates(user_input)
        break

