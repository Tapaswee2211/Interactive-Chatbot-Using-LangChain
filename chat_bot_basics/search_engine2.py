from sys import exception
from langchain_core import messages
from langgraph.graph import message
import typing_extensions
from dotenv import load_dotenv
import os
load_dotenv()

from langchain_tavily import TavilySearch

from langchain.chat_models import init_chat_model

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import json
from langchain_core.messages import ToolMessage


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

tool = TavilySearch(max_results = 2)
tools = [tool]
# print(tool.invoke("What is a node in LangGraph"))

llm = init_chat_model("google_genai:gemini-2.0-flash")

class State(TypedDict):
    messages : Annotated[list, add_messages]

graph_builder = StateGraph(State)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages" : [llm_with_tools.invoke(state["messages"])]}
graph_builder.add_node("chatbot", chatbot)
print("ChatBot Node binded and added to graph")

class BasicToolNode:
    # A node that runs the tools requested in the last AIMessage

    def __init__ (self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__ (self, state: State) -> dict:
        if messages := state.get("messages", []):
            message = messages[-1]
        else:
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
        
        return {"messages": outputs}

tool_node = BasicToolNode(tools = [tool])
graph_builder.add_node("tools", tool_node)
print("Added tool node to the graph")

def route_tools(
        state: State,
    ):
    #use conditional Edge to route to ToolNode if The last Msg has tool calls.
    messages =state.get("messages", [])
    if not messages:
        raise ValueError("No messages found in the state")

    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# def route_tools(
#         state: State,
#     ):
#     #use conditional Edge to route to ToolNode if The last Msg has tool calls.
#     if isinstance(state, list):
#         messages =state.get("messages", [])
#         ai_message = messages[-1]
#     elif messages := state.get("message", []):
#         ai_message = messages[-1]
#     else:
#         raise ValueError(f"No messages found in the state to tool_edge: {state}")
#     
#     if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
#         return "tools"
#     return END
print("added a conditional Edge")

graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        # This dictionary lets you tell the graph to interpret the condition's output as defaults to the indentity function
        # if you want to use a node named something else apart from 'tools' you can update the value of dictonary to something else like 'my_tools;
        {"tools": "tools", END : END},
)
#Any time a tool is called we return to chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()
print("Done compiling")


# try:
#     img_data = graph.get_graph().draw_mermaid_png() 
#     with open("graph.png", "wb") as f:
#         f.write(img_data)
#     print("graph Saved")
# except Exception:
#     pass


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [ {"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant: ", value["messages"][-1].content)

while(True):
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "q", "exit"]:
            print("Goodbye")
            break
        stream_graph_updates(user_input)
    except:
        user_input = "What do you know about LangGraph?"
        print("User: "+ user_input)
        stream_graph_updates(user_input)
        break
