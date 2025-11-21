from logging import INFO
from typing import Annotated
from langgraph import graph
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage, chat
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch
from langgraph.graph import START, END, StateGraph, state
from langgraph.prebuilt import ToolNode, tools_condition


from langgraph.types import Command , interrupt

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class State(TypedDict):
    messages : Annotated[list, add_messages]
    name : str
    birthday : str


@tool 
def human_assistance(
        name : str, birthday: str, tool_call_id : Annotated[str, InjectedToolCallId]) -> Command:

    """Request assistance from a human """
    human_response = interrupt(
            {
                "question" : "Is this correct?",
                "name" : name, 
                "birthday" : birthday,
            },
    )
    if human_response.get('correct', "").lower().startswith("y"):
        verified_name  = name
        verified_birthday = birthday
        response = "Correct"
    else :
        verified_name  = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction : {human_response}"

    state_update = {
            "name" : verified_name,
            "birthday": verified_birthday,
            "messages":  [ToolMessage(response, tool_call_id = tool_call_id) ],
    }
    return Command(update=state_update)

llm =init_chat_model("google_genai:gemini-2.0-flash")
tool = TavilySearch(max_results=2)
tools = [tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state:State):
    message = llm_with_tools.invoke(state["messages"])
    if isinstance(message, AIMessage) and getattr(message,"tool_calls", None):
        assert(len(message.tool_calls)<=1)
    return {"messages" : [message]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools = tools)
graph_builder.add_node("tools", chatbot)

graph_builder.add_conditional_edges(
        "chatbot", 
        tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

print("Chatbot with Human-in-the-Loop started! Type 'exit' to quit.\n")
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "q", "exit"]:
            print("Exiting chatbot...")
            break

        for event in graph.stream({
            "name" : "",
            "birthday" : "",
            "messages" : [{"role" :"user", "content":user_input}]
            }):
            for value in event.values():
                print("Assistant: ", value["messages"][-1].content)
        print("\n")

    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
        break
    except Exception as e:
        print(f"Error: {e}")
        break
