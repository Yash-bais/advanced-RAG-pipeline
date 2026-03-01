# 🧠 Dynamic Hybrid RAG Pipeline

An advanced Retrieval-Augmented Generation (RAG) system built to dynamically process, index, and query unstructured PDF documents using Hybrid Search and a Streamlit UI.

## 🚀 Key Engineering Features

* **Hybrid Search Architecture (RRF):** Implemented Reciprocal Rank Fusion to mathematically merge dense vector retrieval (Semantic/ChromaDB) with sparse keyword retrieval (BM25). This eliminates the "hallucinations" and "lost context" common in basic RAG systems.
* **Dynamic Chunking Heuristic:** Engineered the ETL pipeline to automatically calculate document character density upon upload. It dynamically shifts between small chunk sizes for sparse documents (Slide Decks/Resumes) and larger chunks for dense documents (Textbooks/Articles).
* **Ephemeral In-Memory Indexing:** Transitioned from persistent disk storage to `chromadb.EphemeralClient()`, allowing multi-user scalability without bloating server storage.
* **Automated Evaluation (LLM-as-a-Judge):** Built a custom testing suite to automatically grade the pipeline against a Golden Dataset, mathematically verifying **Context Precision** and **Faithfulness**.

## 🛠️ Tech Stack
* **Backend:** Python, Groq API (Llama-3-8b-instant)
* **Vector Database:** ChromaDB (all-MiniLM-L6-v2 embeddings)
* **Keyword Index:** BM25Okapi (with custom regex NLP tokenization)
* **Frontend:** Streamlit
* **Data Processing:** PyPDF

## 💻 How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Groq API key to a `.env` file (`GROQ_API_KEY=your_key`).
4. Run the app: `streamlit run app.py`

## 🧪 Testing
Run `python evaluate.py` to trigger the automated LLM grading suite, which tests the retrieval accuracy and generation faithfulness of the hybrid search engine against ground-truth data.