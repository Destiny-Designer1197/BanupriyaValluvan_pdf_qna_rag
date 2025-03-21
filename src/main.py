import os
from dotenv import load_dotenv
from openai import OpenAI
import argparse
from utils import get_text_chunks, get_vectorstore, retrieve_relevant_chunks , generate_answer
from dataset import extract_text_from_pdfs
from config import *

# Load environment variables
load_dotenv()


GPT_MODEL = MODEL#"gpt-4o-mini-2024-07-18"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir",type=str,required=True,help="Path to Directory that contains PDF")
    args = parser.parse_args()
    pdf_dir  = args.pdf_dir
    print("Extracting text from PDFs...")
    pdf_text = extract_text_from_pdfs(pdf_dir)
    print("Chunking text...")
    text_chunks = get_text_chunks(pdf_text,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    print("Building vector store...")
    vector_index, chunks = get_vectorstore(text_chunks)
    
    
    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            break
        print("Retrieving relevant chunks...")
        retrieved = retrieve_relevant_chunks(query, vector_index, chunks)
        print("Generating answer...")
        answer = generate_answer(query=query, retrieved_chunks=retrieved,client=client,GPT_MODEL=MODEL,temperature=temperature,max_tokens=max_tokens,frequency_penalty=frequency_penalty)
        print("Answer:", answer)
