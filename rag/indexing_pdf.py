import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Read Google API key only from .env
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Please set GOOGLE_API_KEY in the .env file")

pdf_path = Path(__file__).resolve().parent / "javascript.pdf"
loader = PyPDFLoader(str(pdf_path))
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)
chunks = splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key,
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="javascript_book",
)

print(f"Inserted {len(chunks)} chunks into Qdrant collection 'javascript_book'")
print(f"Qdrant store created: {vector_store}")
