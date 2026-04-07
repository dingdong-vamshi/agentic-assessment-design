from sentence_transformers import SentenceTransformer
import numpy as np

# Load once — small model, ~80MB, free, runs locally
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_knowledge_base(path="src/rag/knowledge_base.txt"):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines

def retrieve_relevant_principles(problems: list[str], top_k: int = 3) -> list[str]:
    """
    Given a list of problems (strings), find the top_k most
    relevant principles from the knowledge base.
    """
    docs = load_knowledge_base()
    
    # Combine problems into one query
    query = " ".join(problems)
    
    # Encode everything into vectors
    doc_embeddings = model.encode(docs)
    query_embedding = model.encode([query])
    
    # Cosine similarity: dot product of normalized vectors
    scores = np.dot(doc_embeddings, query_embedding.T).flatten()
    
    # Get top_k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    return [docs[i] for i in top_indices]
