from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path
from src.text_chunking import chunk_narratives

def create_vector_store(input_path, vector_store_path):
    """Generate embeddings and index in FAISS."""
    # Load chunked data
    chunks_df = chunk_narratives(input_path)
    
    # Initialize embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings
    texts = chunks_df['text'].tolist()
    embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    
    # Save vector store
    faiss.write_index(index, str(vector_store_path / 'faiss_index.bin'))
    
    # Save metadata
    metadata = chunks_df[['chunk_id', 'complaint_id', 'product', 'text']].to_dict('records')
    with open(vector_store_path / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save embedding model name for reference
    with open(vector_store_path / 'model_info.txt', 'w') as f:
        f.write('sentence-transformers/all-MiniLM-L6-v2')
    
    print(f"Vector store saved to {vector_store_path}")
    return index, metadata, embedding_model

if __name__ == "__main__":
    DATA_DIR = Path('data/')
    VECTOR_STORE_DIR = Path('../vector_store')
    INPUT_PATH = DATA_DIR / 'filtered_complaints.csv'
    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    create_vector_store(INPUT_PATH, VECTOR_STORE_DIR)