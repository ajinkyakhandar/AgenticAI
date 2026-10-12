
import os
import warnings
from pathlib import Path
from typing import Annotated, TypedDict

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# Describes the data that flows between graph nodes. TypedDict gives the state
# a clear shape, while Annotated attaches LangGraph's message update behavior.
class State(TypedDict):
    """Chat state containing the conversation messages."""

    # Annotated tells LangGraph to use add_messages as the update rule for this
    # list: new messages are merged into the conversation rather than replacing it.
    messages: Annotated[list[AnyMessage], add_messages]


# Create a graph builder that validates and manages nodes/edges using State.
# Nodes and edges can be added to this builder before compiling the graph.
graph_builder = StateGraph(State)

# Load the Gemini key from the workspace's root .env file and create the model
# that the chatbot node will call.
workspace_root = Path(__file__).resolve().parents[1]
load_dotenv(workspace_root / ".env")
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Set GOOGLE_API_KEY in the workspace .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
)


def chatbot(state: State) -> dict[str, list[AnyMessage]]:
    """Call Gemini with the conversation and return its reply as a message."""
    # LangGraph gives us the full state.
    # state = {"messages": [ ... ]}
    print("===> Chatbot node received state:", state)
    # We pass only the message list to Gemini.
    messages = state["messages"]
    response = llm.invoke(messages)

    # Save the model reply back into the state.
    return {"messages": [response]}


def should_continue(state: State) -> str:
    """Return 'end' when the user says goodbye, otherwise continue chatting."""
    for message in reversed(state["messages"]):
        if getattr(message, "type", "").lower() == "human":
            text = str(message.content).lower()
            if any(word in text for word in ("bye", "goodbye", "exit", "quit")):
                return "end"
            break
    return "continue"


# Register the chatbot node, then add a conditional edge.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    should_continue,
    {
        "continue": "chatbot",
        "end": END,
    },
)

# Compile the configured builder into a runnable LangGraph graph.
graph = graph_builder.compile()


if __name__ == "__main__":
    # Example input: a user message that ends the conversation immediately.
    # If you want a longer chat, use a normal greeting instead of "bye".
    state = {"messages": [HumanMessage(content="Hi")]}

    # Run the graph and get the output from the chatbot node.
    result = graph.invoke(state)

    # Print the AI response.
    print("=== Result ===",result)
