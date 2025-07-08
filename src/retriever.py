import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

class Retriever:
    def __init__(self, vector_store_path, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.vector_store_path = Path(vector_store_path)
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(str(self.vector_store_path / 'faiss_index.bin'))
        with open(self.vector_store_path / 'metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
    
    def retrieve(self, query, k=5):
        """Retrieve top-k relevant chunks for a query."""
        # Embed the query
        query_embedding = self.model.encode([query], show_progress_bar=False)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Retrieve corresponding chunks and metadata
        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

if __name__ == "__main__":
    # Test retriever
    retriever = Retriever('vector_store/')
    query = "Why are people unhappy with BNPL?"
    results = retriever.retrieve(query, k=5)
    for res in results:
        print(f"Chunk ID: {res['chunk_id']}, Product: {res['product']}")
        print(f"Text: {res['text'][:100]}...")
        print("-" * 50)