from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import os
from langchain.chat_models import init_chat_model


os.environ["GOOGLE_API_KEY"] = "AIzaSyCF6vNMihTxk9JzVMBYfzxlt42jVffuY_s"


class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = init_chat_model(
        "google_genai:gemini-2.0-flash"
)


def chatbot1(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
def chatbot2(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot1)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()
print("Done Compiling the Graph")

print("visualizaiton")
from IPython.display import Image, display

try:
    img_data = graph.get_graph().draw_mermaid_png() 
    with open("graph.png", "wb") as f:
        f.write(img_data)
    print("graph Saved")
except Exception:
    pass


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant: ", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break

