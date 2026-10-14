
import os
import warnings
from pathlib import Path
from typing import Annotated, TypedDict

import pymongo
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


graph_builder = StateGraph(State)

workspace_root = Path(__file__).resolve().parents[1]
load_dotenv(workspace_root / ".env")
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Set GOOGLE_API_KEY in the workspace .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    retries=0,
)


def chatbot(state: State) -> dict[str, list[AnyMessage]]:
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


if __name__ == "__main__":
    mongodb_uri = os.getenv(
        "MONGODB_URI",
        "mongodb://admin:admin123@localhost:27017/?authSource=admin",
    )
    thread_id = "name-memory-demo"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        with MongoDBSaver.from_conn_string(
            mongodb_uri,
            db_name="langgraph",
        ) as checkpointer:
            graph = graph_builder.compile(checkpointer=checkpointer)

            msg = graph.invoke(
                {"messages": [HumanMessage(content=f"whats my name?")]},
                config=config,
            )

            print("============>Assistant:", msg["messages"][-1].content)
    except pymongo.errors.ServerSelectionTimeoutError as exc:
        print("MongoDB is not running on localhost:27017.")
        print("Start it with: docker compose up -d")
        print("Then run this script again.")
        raise SystemExit(1) from exc
