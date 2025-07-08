from src.retriever import Retriever
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

class RAGPipeline:
    def __init__(self, vector_store_path, llm_model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        load_dotenv()
        self.retriever = Retriever(vector_store_path)
        self.llm = HuggingFaceEndpoint(
            repo_id=llm_model,
            max_new_tokens=500,
            temperature=0.7,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints based solely on the provided context. Summarize the key issues concisely and accurately. If the context does not contain enough information to answer the question, state: "I don't have enough information to answer this question."

Context:
{context}

Question:
{question}

Answer:
"""
        )
    
    def answer(self, question):
        """Generate an answer for the given question."""
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(question, k=5)
        context = "\n".join([chunk['text'] for chunk in chunks])
        
        # Format prompt
        prompt = self.prompt_template.format(context=context, question=question)
        
        # Generate answer
        answer = self.llm.invoke(prompt)
        
        return {
            "answer": answer,
            "sources": chunks
        }

if __name__ == "__main__":
    # Test RAG pipeline
    rag = RAGPipeline('../vector_store')
    question = "Why are people unhappy with BNPL?"
    result = rag.answer(question)
    print("Answer:", result['answer'])
    print("\nSources:")
    for src in result['sources'][:2]:  # Show top 2 for brevity
        print(f"Chunk ID: {src['chunk_id']}, Product: {src['product']}")
        print(f"Text: {src['text'][:100]}...")
        print("-" * 50)