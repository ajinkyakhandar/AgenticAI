import os
import sys
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from mem0 import Memory
from neo4j import GraphDatabase

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
google_api_key = os.getenv("GOOGLE_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

if not google_api_key:
    raise RuntimeError(
        "GOOGLE_API_KEY is missing. Add it to a .env file in the project root "
        "or set it in your terminal environment."
    )

if not all((neo4j_uri, neo4j_username, neo4j_password)):
    raise RuntimeError(
        "Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in the project-root .env file."
    )


def check_python_version():
    if sys.version_info < (3, 10):
        raise RuntimeError("Use Python 3.10 or newer (this project has .venv312).")


check_python_version()

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "gemini",
        "config": {
            "api_key": google_api_key,
            "model": "models/gemini-embedding-001",
            "embedding_dims": 1536,
        },
    },
    "llm": {
        "provider": "gemini",
        "config": {"api_key": google_api_key, "model": "gemini-3.5-flash-lite"},
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": "localhost", "port": 6333},
    },
}

mem_client = Memory.from_config(config)
neo4j_driver = GraphDatabase.driver(
    neo4j_uri,
    auth=(neo4j_username, neo4j_password),
)
neo4j_driver.verify_connectivity()

chat_model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=google_api_key,
)
user_id = "ajinkya"

while True:
    user_query = input("> ").strip()
    if user_query.lower() in {"exit", "quit"}:
        break
    if not user_query:
        continue

    # Retrieve relevant memories and make them available as prompt context.
    search_response = mem_client.search(
        query=user_query, filters={"user_id": user_id}
    )
    memories = search_response.get("results", [])

    user_message = (
        "Use this saved context when it is relevant:\n"
        f"{memories}\n\n"
        f"User query: {user_query}"
    )

    response = chat_model.invoke([HumanMessage(content=user_message)])
    ai_response = str(response.content)
    print(ai_response)

    with neo4j_driver.session(database=neo4j_database) as session:
        session.run(
            """
            MERGE (user:User {user_id: $user_id})
            CREATE (conversation:Conversation {
                id: $conversation_id,
                query: $user_query,
                answer: $answer,
                created_at: datetime()
            })
            MERGE (user)-[:HAD_CONVERSATION]->(conversation)
            """,
            user_id=user_id,
            conversation_id=str(uuid.uuid4()),
            user_query=user_query,
            answer=ai_response,
        ).consume()
    print("Conversation saved to Neo4j.")

    mem_client.add(
        user_id=user_id,
        messages=[{"role": "user", "content": user_query}],
        infer=False,
    )
    print("Memory saved to Qdrant.")

neo4j_driver.close()
