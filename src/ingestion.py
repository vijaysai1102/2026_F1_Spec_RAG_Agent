import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Configuration
PDF_PATH = os.path.join("data", "2026_f1_technical_regulations.pdf")
VECTOR_DB_DIR = "vector_db"

def ingest_data():
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at {PDF_PATH}")
        return

    print("Loading PDF...")
    loader = PyMuPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Initializing embeddings and ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    print(f"Vector store created and saved to {VECTOR_DB_DIR}")

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment. Please set it in .env file.")
    else:
        ingest_data()
