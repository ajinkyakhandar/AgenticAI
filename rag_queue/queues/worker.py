"""RQ worker and query-processing job for the JavaScript PDF index."""

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from client.rq_client import connection, queue
from rq import SimpleWorker

warnings.filterwarnings("ignore", category=Warning)

# Load configuration and initialize shared service clients when the worker starts.
workspace_root = Path(__file__).resolve().parents[2]
load_dotenv(workspace_root / ".env")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Please set GOOGLE_API_KEY in the workspace .env file")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key,
)
vector_store = QdrantVectorStore.from_existing_collection(
    collection_name="javascript_book",
    embedding=embeddings,
    url="http://localhost:6333",
)
llm = ChatGoogleGenerativeAI(
    model="models/gemini-3.8-flash",
    google_api_key=api_key,
)


def process_queue(query: str) -> dict[str, object]:
    """Process one queued query and return relevant passages and the answer."""
    retrieved_docs = vector_store.similarity_search(query, k=5)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer using only the provided context.\n\n"
                f"Context:\n{context}"
            ),
        },
        {"role": "user", "content": f"Question: {query}"},
    ]
    response = llm.invoke(messages)

    return {
        "query": query,
        "retrieved_results": [doc.page_content[:500] for doc in retrieved_docs],
        "answer": response.content,
    }


if __name__ == "__main__":
    # SimpleWorker runs jobs in this process instead of forking a work-horse.
    # This avoids macOS native-library crashes caused by forking initialized
    # Google/gRPC clients. Keep it running or API-submitted jobs stay queued.
    worker = SimpleWorker([queue], connection=connection)
    worker.work()
