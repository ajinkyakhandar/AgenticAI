import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=Warning)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# Read Google API key only from .env
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Please set GOOGLE_API_KEY in the .env file")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key,
)

vector_store = QdrantVectorStore.from_existing_collection(
    collection_name="javascript_book",
    embedding=embeddings,
    url="http://localhost:6333",
)

# Read user query from command line or terminal input.
query = " ".join(sys.argv[1:])
if not query:
    query = input("Enter your query: ").strip() or "What is this document about?"

retrieved_docs = vector_store.similarity_search(query, k=5)
context = "\n\n".join(doc.page_content for doc in retrieved_docs)

llm = ChatGoogleGenerativeAI(
    model="models/gemini-3.8-flash",
    google_api_key=api_key,
)

# ChatML-style message format: system gets the context, user asks the question.
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

print(f"Query: {query}\n")
print("Top retrieval results:")
for i, doc in enumerate(retrieved_docs, start=1):
    print(f"Result {i}:\n{doc.page_content[:500]}\n{'-' * 80}\n")

print("\n ====> Final Gemini answer:\n")
print(response.content)

# ------------------------------------------------------------
# Optional OpenAI alternative (commented out)
# ------------------------------------------------------------
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     raise ValueError("Please set OPENAI_API_KEY in the .env file")
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     openai_api_key=openai_api_key,
# )
# vector_store = QdrantVectorStore.from_existing_collection(
#     collection_name="javascript_book",
#     embedding=embeddings,
#     url="http://localhost:6333",
# )
# retrieved_docs = vector_store.similarity_search(query, k=5)
# context = "\n\n".join(doc.page_content for doc in retrieved_docs)
# llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
# messages = [
#     SystemMessage(content="You are a helpful assistant. Answer using only the provided context."),
#     HumanMessage(content=f"Question: {query}\n\nContext:\n{context}"),
# ]
# response = llm.invoke(messages)
# print(response.content)
