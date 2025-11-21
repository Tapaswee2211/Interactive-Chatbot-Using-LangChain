from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model


from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START , END
from langgraph.graph.message import add_messages

import json 
from langchain_core.messages import ToolMessage


load_dotenv()

from langchain_tavily import TavilySearch

tool = TavilySearch(max_results=2)
tools = [tool]
# result = tool.invoke("What is the answer to the equation 6x^2 + 5x + 15 = 0")
# for r in result.get("results", []):
#     print(f"- {r['title']}")
#     print(f" {r['url']}")
#     print(f" {r['content'][:5000]}...")
#     print()
tool.invoke("What's a 'node' in LangGraph?")

llm = init_chat_model("google_genai:gemini-2.0-flash")


class State(TypedDict):
  messages : Annotated[list, add_messages]

graph_builder = StateGraph(State)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
print("chatBot Node Added")

class BasicToolNode:
    # A Node that runs tools requested in the last AIMessage
    def __init__ (self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__ (self, state: State) -> dict:
        if messages:= state.get("messages",  []):
            message = messages[-1]
        else:
            raise ValueError("No Message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
            )
            outputs.append(
                    ToolMessage(
                        content= json.dumps(tool_result),
                        name = tool_call["name"],
                        tool_call_id= tool_call["id"],
                    )
            )
        return {"messages" : outputs}


tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)


def route_tools(
        state: State,
        ):
    #use in the conditional_edge to rout to the ToolNode if the last maessage has tool calls. Otherwise route to the end,

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls)>0:
        return "tools"
    return END
# The 'tools_condition' function returns tools if the chatbot asks to use a tool, and END if it is fine directly responding. 
# This conditinoal routing defines the main agent loop.


