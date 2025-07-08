from src.rag_pipeline import RAGPipeline
import pandas as pd

def evaluate_rag(vector_store_path):
    """Evaluate the RAG pipeline with test questions."""
    rag = RAGPipeline(vector_store_path)
    questions = [
        "Why are people unhappy with BNPL?",
        "What issues are reported with Credit Cards?",
        "Are there complaints about delays in Money Transfers?",
        "What are the common problems with Savings Accounts?",
        "How do Personal Loan complaints differ from BNPL complaints?"
    ]
    
    evaluation = []
    for question in questions:
        result = rag.answer(question)
        answer = result['answer']
        sources = result['sources'][:2]  # Top 2 sources
        # Placeholder quality score and comments (to be manually assigned after inspection)
        quality_score = 0  # Update after reviewing output
        comments = "Pending manual review"
        evaluation.append({
            "Question": question,
            "Generated Answer": answer,
            "Retrieved Sources": "\n".join([f"Chunk ID: {src['chunk_id']}, Text: {src['text'][:100]}..." for src in sources]),
            "Quality Score": quality_score,
            "Comments": comments
        })
    
    # Save evaluation table
    eval_df = pd.DataFrame(evaluation)
    eval_df.to_csv('../reports/evaluation_table.csv', index=False)
    return eval_df

if __name__ == "__main__":
    eval_df = evaluate_rag('../vector_store')
    print(eval_df)