import streamlit as st
import os
import re
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from openai import OpenAI

# 1. SETUP & CONFIGURATION

st.set_page_config(page_title="Dynamic RAG Assistant", page_icon="🧠", layout="wide")

# Initialize the LLM Client
llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

# Initialize Session State variables to hold our databases and chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_processed" not in st.session_state:
    st.session_state.is_processed = False
if "chroma_collection" not in st.session_state:
    st.session_state.chroma_collection = None
if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []


# ==========================================
# 2. PIPELINE FUNCTIONS
# ==========================================
def process_pdf(uploaded_file):
    """Reads the PDF and dynamically calculates the optimal chunk size."""
    reader = PdfReader(uploaded_file)
    raw_text = ""
    total_chars = 0
    num_pages = len(reader.pages)

    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"
            total_chars += len(text)

    # The Dynamic Chunking Heuristic
    avg_chars_per_page = total_chars / num_pages if num_pages > 0 else 0

    if avg_chars_per_page < 800:
        chunk_size, overlap = 300, 50
        doc_type = "Sparse (Slide Deck / Resume)"
    else:
        chunk_size, overlap = 1000, 200
        doc_type = "Dense (Textbook / Article)"

    st.sidebar.success(f"Detected: {doc_type}\nUsing Chunk Size: {chunk_size}")

    # Perform the chunking
    chunks = []
    for i in range(0, len(raw_text), chunk_size - overlap):
        chunks.append(raw_text[i:i + chunk_size])

    return chunks


def tokenize(text):
    """NLP tokenizer for BM25 Keyword Search."""
    words = re.findall(r'\w+', text.lower())
    stop_words = {'what', 'are', 'is', 'the', 'of', 'in', 'and', 'to', 'a', 'for'}
    return [w for w in words if w not in stop_words]


def build_indexes(chunks):
    """Builds the Vector DB and Keyword Index in RAM."""
    # 1. Ephemeral ChromaDB (Lives only in RAM, deletes when app closes)
    db_client = chromadb.EphemeralClient()
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Recreate collection cleanly
    try:
        db_client.delete_collection("dynamic_knowledge")
    except:
        pass

    collection = db_client.create_collection(
        name="dynamic_knowledge",
        embedding_function=sentence_transformer_ef
    )

    ids = [f"id_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)

    # 2. Build BM25
    tokenized_docs = [tokenize(doc) for doc in chunks]
    bm25 = BM25Okapi(tokenized_docs)

    # 3. Save to Session State so Streamlit remembers them between chat messages
    st.session_state.chroma_collection = collection
    st.session_state.bm25_index = bm25
    st.session_state.all_chunks = chunks
    st.session_state.is_processed = True


def hybrid_search(query, top_k=5):
    """Executes Vector + Keyword search with Reciprocal Rank Fusion."""
    collection = st.session_state.chroma_collection
    bm25 = st.session_state.bm25_index
    all_docs = st.session_state.all_chunks

    # A. Vector Search
    vector_results = collection.query(query_texts=[query], n_results=10)
    vector_docs = vector_results['documents'][0]

    # B. Keyword Search
    tokenized_query = tokenize(query)
    keyword_docs = bm25.get_top_n(tokenized_query, all_docs, n=10)

    # C. Reciprocal Rank Fusion
    fused_scores = {}

    def apply_rrf(doc_list):
        for rank, doc in enumerate(doc_list):
            if doc not in fused_scores:
                fused_scores[doc] = 0.0
            fused_scores[doc] += 1.0 / (rank + 60)

    apply_rrf(vector_docs)
    apply_rrf(keyword_docs)

    sorted_docs = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc[0] for doc in sorted_docs[:top_k]]


def ask_pdf(user_query):
    """Generates the final answer using Groq and the retrieved chunks."""
    relevant_chunks = hybrid_search(user_query, top_k=5)
    context_string = "\n\n---\n\n".join(relevant_chunks)

    system_prompt = f"""You are a precise data extraction assistant. 
    Answer the user's question based ONLY on this context.
    If the answer is not in the context, reply: "I do not know based on the provided document."

    CONTEXT:
    {context_string}
    """
    try:
        completion = llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"API Error: {e}"


# ==========================================
# 3. USER INTERFACE
# ==========================================
st.title("📄 Dynamic Document AI")
st.markdown("Upload any PDF, and the pipeline will automatically optimize its processing strategy.")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and st.button("Process PDF"):
        with st.spinner("Extracting text and calculating density..."):
            chunks = process_pdf(uploaded_file)
        with st.spinner("Building Vector & Keyword Indexes in RAM..."):
            build_indexes(chunks)
        st.success("Ready to chat!")

# Main Chat Interface
if st.session_state.is_processed:
    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_query := st.chat_input("Ask a question about your PDF..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Searching dynamic index..."):
                bot_answer = ask_pdf(user_query)
                st.markdown(bot_answer)
        st.session_state.messages.append({"role": "assistant", "content": bot_answer})
else:
    st.info("👈 Please upload and process a PDF in the sidebar to begin.")