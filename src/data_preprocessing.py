import pandas as pd
import re
from pathlib import Path

def clean_text(text):
    """Clean a single text narrative."""
    if pd.isna(text):
        return text
    # Lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Remove boilerplate
    boilerplate = [
        r'i am writing to file a complaint',
        r'please assist',
        r'thank you for your attention'
    ]
    for phrase in boilerplate:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def preprocess_data(input_path, output_path):
    """Filter and clean the CFPB dataset."""
    # Load data
    df = pd.read_csv(input_path)
    
    # Define target products
    target_products = [
        'Credit card',
        'Consumer Loan',  # Often used for Personal Loan
        'Payday loan, title loan, personal loan, or advance loan',  # Includes BNPL
        'Checking or savings account',
        'Money transfer, virtual currency, or money service'
    ]
    
    # Filter by products
    df_filtered = df[df['Product'].isin(target_products)]
    
    # Remove empty narratives
    df_filtered = df_filtered[df_filtered['Consumer complaint narrative'].notnull()]
    
    # Clean narratives
    df_filtered['Consumer complaint narrative'] = df_filtered['Consumer complaint narrative'].apply(clean_text)
    
    # Save cleaned dataset
    df_filtered.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")
    return df_filtered

if __name__ == "__main__":
    DATA_DIR = Path('data/')
    RAW_DATA_PATH = DATA_DIR / 'raw/complaints.csv'
    OUTPUT_PATH = DATA_DIR / 'filtered_complaints.csv'
    preprocess_data(RAW_DATA_PATH, OUTPUT_PATH)