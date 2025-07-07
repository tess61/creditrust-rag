from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from pathlib import Path

def chunk_narratives(input_path):
    """Split complaint narratives into chunks."""
    # Load cleaned dataset
    df = pd.read_csv(input_path)
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    # Prepare output data
    chunks = []
    for idx, row in df.iterrows():
        narrative = row['Consumer complaint narrative']
        complaint_id = row.get('Complaint ID', idx)  # Use index if Complaint ID is missing
        product = row['Product']
        
        # Split narrative into chunks
        split_texts = text_splitter.split_text(narrative)
        
        # Store each chunk with metadata
        for i, chunk in enumerate(split_texts):
            chunks.append({
                'chunk_id': f"{complaint_id}_{i}",
                'complaint_id': complaint_id,
                'product': product,
                'text': chunk
            })
    
    # Convert to DataFrame
    chunks_df = pd.DataFrame(chunks)
    return chunks_df

if __name__ == "__main__":
    DATA_DIR = Path('data/')
    INPUT_PATH = DATA_DIR / 'filtered_complaints.csv'
    chunks_df = chunk_narratives(INPUT_PATH)
    print(f"Created {len(chunks_df)} chunks.")
    # Save chunks for debugging (optional)
    chunks_df.to_csv(DATA_DIR / 'chunks.csv', index=False)