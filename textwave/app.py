import os
import glob
from flask import Flask, request, jsonify
import faiss
import numpy as np

# Use absolute imports (no leading dots)
from modules.extraction.preprocessing import DocumentProcessing
from modules.extraction.embedding import Embedding
from modules.retrieval.indexing import FaissIndexer
from modules.retrieval.search import FaissSearcher
from modules.retrieval.reranker import Reranker
from modules.generator.Youtubeing import QuestionAnswering

app = Flask(__name__)

# --- System Configuration ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIRECTORY = os.path.join(APP_ROOT, 'storage')

CHUNKING_STRATEGY = 'sentence'
CHUNKING_PARAMETERS = {"num_sentences": 4, "overlap_size": 1}
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
INDEX_STRATEGY = "bruteforce"
RERANKING_STRATEGY = "cross_encoder"
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 3

# --- Global Variables ---
faiss_index = None
text_chunks = []
preprocessor = DocumentProcessing()
embedder = Embedding(model_name=EMBEDDING_MODEL)

def initialize_index():
    global faiss_index, text_chunks
    doc_paths = glob.glob(os.path.join(STORAGE_DIRECTORY, "*.txt.clean"))
    if not doc_paths:
        print(f"Warning: No documents found in {STORAGE_DIRECTORY}. The index will be empty.")
        return None, []
    
    all_chunks = []
    for path in doc_paths:
        all_chunks.extend(preprocessor.run(path, strategy=CHUNKING_STRATEGY, **CHUNKING_PARAMETERS))
    if not all_chunks:
        print("Warning: No chunks were created from the documents.")
        return None, []
    text_chunks.clear()
    text_chunks.extend(all_chunks)
    chunk_embeddings = embedder.embed_documents(text_chunks)
    indexer = FaissIndexer(dim=embedder.get_embedding_dim(), strategy=INDEX_STRATEGY)
    faiss_index = indexer.create_index()
    indexer.add_to_index(chunk_embeddings)
    print(f"Index initialized with {faiss_index.ntotal} vectors from {len(text_chunks)} chunks.")
    return faiss_index, text_chunks

@app.route("/generate", methods=["POST"])
def generate_answer():
    global faiss_index, text_chunks
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    query = data.get("query")
    if not query or not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Query is missing or empty"}), 422
    if faiss_index is None or faiss_index.ntotal == 0:
        return jsonify({"error": "Index is not initialized or is empty."}), 503
    searcher = FaissSearcher(faiss_index)
    reranker = Reranker(type=RERANKING_STRATEGY)
    generator = QuestionAnswering(api_key=os.getenv("MISTRAL_API_KEY"))
    query_embedding = embedder.embed_query(query)
    _, retrieved_indices = searcher.search(np.array([query_embedding]), k=TOP_K_RETRIEVAL)
    retrieved_docs = [text_chunks[i] for i in retrieved_indices[0]]
    ranked_docs, _, _ = reranker.rerank(query, retrieved_docs)
    top_docs = ranked_docs[:TOP_K_RERANK]
    context_str = "\n\n".join(top_docs)
    final_answer = generator.generate(context=context_str, question=query)
    return jsonify({"query": query, "answer": final_answer})

# Initialize the index when the app module is first loaded.
initialize_index()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)