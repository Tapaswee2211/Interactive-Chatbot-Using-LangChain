import sys
import traceback
import os
import json 
import typing_extensions
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_tavily import TavilySearch

from langchain_core import messages
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from langchain.chat_models import init_chat_model

from langgraph.graph import message
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langgraph.checkpoint.memory import InMemorySaver

#This is the primary Checkpoint.
#For Production Applicatino change this to SqliteSaver / PostgresSaver
memory = InMemorySaver()
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

tool = TavilySearch(max_results = 3)
tools = [tool]

llm = init_chat_model("google_genai:gemini-2.0-flash")


class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm_with_tools = llm.bind_tools(tools)

def chatbot(state : State):
    return {"messages" : [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
print("ChatBot Node binded and added to graph")

class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name : tool for tool in tools}
    def __call__(self, state:State) -> dict:
        if messages := state.get("messages", []):
            message = messages[-1]
        else :
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
            )
            outputs.append(
                    ToolMessage(
                        content = json.dumps(tool_result),
                        name = tool_call["name"],
                        tool_call_id = tool_call["id"],
                    )
            )
        return {"messages" : outputs}
tool_node = BasicToolNode(tools = [tool])
graph_builder.add_node("tools", tool_node)
print("Added tool node to the graph")

def route_tools(
        state: State,
):
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No messages found in the state")
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


graph_builder.add_conditional_edges(
        "chatbot", 
        route_tools,
        {"tools" : "tools", END : END}
    )
print("added a conditional Edge")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile(checkpointer=memory)
print("Done Compiling")

"""
def stream_graph_updates(user_input: str):
    for event in graph.stream(
            {"messages": [ {"role" : "user", "content": user_input}]}, 
            {"configurable": {"thread_id" : "2" }},
            stream_mode = "values"
    ):
        for value in event.values():
                print("assistant: ", value["messages"][-1].content)
while (True):
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "q", "exit"]:
            break
        stream_graph_updates(user_input)
    except:
        user_input = "What do you know about anything related to IT in india"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
"""

while (True):
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "q", "exit"]:
            break
        events = graph.stream(
            {"messages": [ {"role" : "user", "content": user_input}]}, 
            {"configurable": {"thread_id" : "2" }},
            stream_mode = "values"
        )
        for event in events:
            event["messages"][-1].pretty_print()
        print("\n")
        print("\n")
    except:
        user_input = "What do you know about anything related to IT in india"
        events = graph.stream(
            {"messages": [ {"role" : "user", "content": user_input}]}, 
            {"configurable": {"thread_id" : "2" }},
            stream_mode = "values"
        )
        for event in events:
            event["messages"][-1].pretty_print()
        print("\n")
        break
