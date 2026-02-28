import os
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions


def extract_text_from_pdf(pdf_path):
    """Step 1: Extract (Read the PDF)"""
    print(f"Reading {pdf_path}...")
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"
    return raw_text


def chunk_text(text, chunk_size=300, overlap=50): # <--- chunk size should be updated as per the document provided
    """Step 2: Transform (Break text into overlapping chunks)"""
    print(f"Chunking text into {chunk_size}-character blocks...")
    chunks = []
    # We use an overlap so we don't accidentally cut a sentence in half
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


def build_vector_db(chunks):
    """Step 3: Load (Embed and save to ChromaDB)"""
    print("Initializing ChromaDB and embedding model...")

    # PersistentClient saves the database to hard drive in a folder called "rag_db"
    client = chromadb.PersistentClient(path="./rag_db")

    # We use a free, local, open-source embedding model from HuggingFace
    # It converts text into a vector of 384 floating point numbers
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Create or load a "table" (called a collection in vector DBs)
    collection = client.get_or_create_collection(
        name="my_local_knowledge",
        embedding_function=sentence_transformer_ef
    )

    # IDs for the database ("id_0", "id_1")
    ids = [f"id_{i}" for i in range(len(chunks))]

    print(f"Embedding and storing {len(chunks)} chunks... (This might take a moment on the first run)")
    collection.add(
        documents=chunks,
        ids=ids
    )
    print("Done! Database saved to ./rag_db")


if __name__ == "__main__":

    if not os.path.exists("document.pdf"):
        print("Error: Please put a 'document.pdf' file in this directory.")
    else:
        full_text = extract_text_from_pdf("document.pdf")
        text_chunks = chunk_text(full_text)
        build_vector_db(text_chunks)

