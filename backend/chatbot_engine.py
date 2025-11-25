from dotenv import load_dotenv
import os
from typing import Annotated, TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = init_chat_model("google_genai:gemini-2.0-flash")
tavily_tool = TavilySearch(max_results=3)
tools = [tavily_tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    return {"messages": [message]}
'''
def human_assistance_node(state: State):
    last_msg = state["messages"][-1]
    print("\nHuman Intervention Needed:")
    print(last_msg.content)
    correction = input("Human: ")
    return {"messages": [AIMessage(content=correction)]}
'''
def human_assistance_node(state: State):
    last_msg = state["messages"][-1]
    print("\n\n--- Human Intervention Activated ---")
    print(f"The LLM's last thought/request was:\n> {last_msg.content}")
    print("\nPlease provide the corrected, new, or specific instruction.")
    correction = input("Human Correction/Instruction: ") 
    
    # *** FIX: Wrap the human input as a HumanMessage ***
    return {"messages": [HumanMessage(content=correction)]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_node("human", human_assistance_node)

def route_condition(state: State):
    messages = state["messages"]
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    if "help" in last.content.lower() or  last.content.lower().strip() == "/human":
        return "human"
    return END

graph_builder.add_conditional_edges("chatbot", route_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def run_chatbot(user_input: str, session_id="1"):
    events = graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        {"configurable": {"thread_id": session_id}},
        stream_mode="values"
    )

    responses = []
    for event in events:
        if "messages" in event:
            msg = event["messages"][-1]
            if isinstance(msg, AIMessage):
                responses.append(msg.content)

    return responses[-1] if responses else "No response"

