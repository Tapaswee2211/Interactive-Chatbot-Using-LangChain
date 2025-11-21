from dotenv import load_dotenv
import os
from typing import Annotated, TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from IPython.display import Image,display 
# ------------------ Load environment ------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ------------------ Define State ------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ------------------ LLM + Tools ------------------
llm = init_chat_model("google_genai:gemini-2.0-flash")
tavily_tool = TavilySearch(max_results=3)
tools = [tavily_tool]
llm_with_tools = llm.bind_tools(tools)

# ------------------ Chatbot Node ------------------
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    return {"messages": [message]}

# ------------------ Human-in-the-loop Node ------------------
def human_assistance_node(state: State):
    last_msg = state["messages"][-1]
    print("\n  HUMAN ASSISTANCE REQUIRED ")
    print("The AI requested human help for this message:")
    print(f" {last_msg.content}")
    human_reply = input("Human (enter your response): ")
    return {"messages": [AIMessage(content=human_reply)]}

# ------------------ Graph Construction ------------------
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_node("human", human_assistance_node)

# Conditional routing logic
# ------------------ Conditional routing logic ------------------
def route_condition(state: State):
    messages = state.get("messages", [])
    if not messages:
        return END
    last = messages[-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    if "help" in last.content.lower() or "human" in last.content.lower():
        return "human"
    return END

# ------------------ Graph Construction ------------------
# Use modern LangGraph conditional edge syntax
graph_builder.add_conditional_edges(
    "chatbot",
    route_condition,
    {
        "tools": "tools",
        "human": "human",
        END: END,
    },
)

# Define looping edges
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.add_edge(START, "chatbot")

# ------------------ Compile Graph ------------------
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# ------------------ Generate & Save Graph Image ------------------
def save_graph_image(graph, filename="chatbot_graph.png"):
    """
    Save a visualization of the LangGraph (compiled or not) as a PNG image
    using LangGraph's built-in Mermaid renderer.
    """
        # Get the internal graph object (works for compiled graphs too)
    g = graph.get_graph()

    # Render as PNG
    img_bytes = g.draw_mermaid_png()

    # Save image to file
    with open(filename, "wb") as f:
        f.write(img_bytes)

    print(f"✅ LangGraph image saved as {filename}")

    # Optionally display inline (works in notebooks or VS Code interactive window)
    try:
        display(Image(img_bytes))
    except Exception:
        pass

        print(f"⚠️ Could not generate graph image: {e}")
save_graph_image(graph, "chatbot_graph.png")
# ------------------ Chat Loop ------------------
print("Chatbot with Human-in-the-Loop started! Type 'exit' to quit.\n")
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "q", "exit"]:
            print("Exiting chatbot...")
            break

        events = graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            {"configurable": {"thread_id": "1"}},
            stream_mode="values"
        )

        for event in events:
            if "messages" in event and event["messages"]:
                event["messages"][-1].pretty_print()

        print("\n")

    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
        break
    except Exception as e:
        print(f"⚠️ Error: {e}")
        break

