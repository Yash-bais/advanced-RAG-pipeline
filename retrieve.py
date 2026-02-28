import re
import os
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# 1. Setup the LLM Client (Using Groq for speed/cost, just like Phase 1)
llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

# 2. Connect to the Local Vector Database
print("Connecting to database...")
db_client = chromadb.PersistentClient(path="./rag_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = db_client.get_collection(name="my_local_knowledge", embedding_function=sentence_transformer_ef)

print("Building Keyword Search Index (BM25)...")
all_docs_dict = collection.get()
all_docs = all_docs_dict['documents']

def tokenize(text):
    """Extracts words, forces lowercase, and removes useless stop-words."""
    words = re.findall(r'\w+', text.lower())
    # A basic set of stop words to ignore
    stop_words = {'what', 'are', 'is', 'the', 'of', 'in', 'and', 'to', 'a', 'for'}
    return [w for w in words if w not in stop_words]

tokenized_docs = [tokenize(doc) for doc in all_docs]
bm25 = BM25Okapi(tokenized_docs)

def hybrid_search(query, top_k=3):
    """Executes both Vector and Keyword search, merging with RRF"""

    # A. Vector Search
    vector_results = collection.query(query_texts=[query], n_results=10)
    vector_docs = vector_results['documents'][0]

    # B. Keyword Search
    tokenized_query = tokenize(query)
    keyword_docs = bm25.get_top_n(tokenized_query, all_docs, n=10)

    """   # --- DIAGNOSTIC PROBE ---
    print("\n[PROBE] Top Vector Match:", repr(vector_docs[0][:100]))
    if keyword_docs:
        print("[PROBE] Top Keyword Match:", repr(keyword_docs[0][:100]))
    """

    # C. Reciprocal Rank Fusion (The Merger)
    fused_scores = {}

    def apply_rrf(doc_list):
        for rank, doc in enumerate(doc_list):
            if doc not in fused_scores:
                fused_scores[doc] = 0.0
            fused_scores[doc] += 1.0 / (rank + 60)

    apply_rrf(vector_docs)
    apply_rrf(keyword_docs)

    # Sort by highest score and grab the top_k
    sorted_docs = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc[0] for doc in sorted_docs[:top_k]]


def ask_pdf(user_query):
    """Combines the retrieved hybrid context with the LLM prompt."""

    # 1. Get the absolute best chunks using Hybrid Search
    relevant_chunks = hybrid_search(user_query, top_k=5)
    context_string = "\n\n---\n\n".join(relevant_chunks)

    #print(f"\n[DEBUG] Number of chunks retrieved: {len(relevant_chunks)}")
    #print(f"[DEBUG] The exact text sent to the LLM:\n{context_string}\n-------------------")

    # 2. Build the strict prompt
    system_prompt = f"""You are a precise data extraction assistant. 
    Read the provided context below and answer the user's question based ONLY on this context.
    If the answer is not contained in the context, you must reply: "I do not know based on the provided document."

    CONTEXT:
    {context_string}
    """

    # 3. Generate the answer
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


if __name__ == "__main__":
    print("--- Advanced RAG Bot Ready (Hybrid Search Enabled) ---")
    while True:
        question = input("\nAsk a question (or type 'quit'): ")
        if question.lower() in ['quit', 'exit']:
            break

        print("Searching and Thinking...")
        answer = ask_pdf(question)
        print(f"\nBot: {answer}")