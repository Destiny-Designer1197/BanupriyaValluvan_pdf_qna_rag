from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import faiss
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings


def get_text_chunks(text,chunk_size=1000, chunk_overlap=200):
    """
    This function splits a given text into smaller chunks using a RecursiveCharacterTextSplitter.
    The splitter divides the text into chunks of size 1000 characters with an overlap of 200 characters.

    Parameters:
        text (str): The input text to be split into chunks.

    Returns:
        List[str]: A list of strings, where each string represents a chunk of the input text.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    """
    This function generates a vector store from a list of text chunks using OpenAI embeddings.
    The function embeds each chunk using the OpenAI embeddings model, creates a FAISS index,
    and adds the embedded vectors to the index.
    Args:
        text_chunks (List[str]): A list of strings, where each string represents a chunk of text.

    Returns:
        Tuple[faiss.Index, List[str]]: A tuple containing the FAISS index and the list of text chunks.
        The FAISS index can be used for efficient vector similarity search.
    """    
    embeddings = OpenAIEmbeddings()
    vectors = [embeddings.embed_query(chunk) for chunk in text_chunks]
    # vectors = embeddings.embed_documents(text_chunks) 
    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors, dtype=np.float32))
    return index, text_chunks


def retrieve_relevant_chunks(query, index, text_chunks, top_k=3):
    """
    This function retrieves the top-k most relevant chunks from a given vector store based on a query.
    It uses the FAISS library for efficient vector similarity search.

    Parameters:
        - query (str) : The input query string for which relevant chunks need to be retrieved.
        - index (faiss.Index) : The FAISS index containing the embedded vectors of the text chunks.
        - text_chunks (List[str]) : A list of strings representing the text chunks.
        - top_k (int, optional) : The number of top-k relevant chunks to retrieve. Default is 3.

    Returns:
        - List[str]: A list of strings representing the top-k most relevant chunks based on the input query.
    """
    embedding = OpenAIEmbeddings().embed_query(query)
    _, indices = index.search(np.array([embedding], dtype=np.float32), top_k)
    return [text_chunks[i] for i in indices[0] if i < len(text_chunks)]

def generate_answer(query, retrieved_chunks,client,GPT_MODEL,temperature=0.5,max_tokens=170,frequency_penalty=1.5):
    """
    This function generates an answer to a given query using a GPT-4 model.
    The function takes a query and a list of retrieved chunks as input.
    It constructs a prompt using the retrieved chunks and sends it to the GPT-4 model.
    The model generates an answer based on the provided context and question.

    Parameters:
        - query (str): The input query string for which an answer needs to be generated.
        - retrieved_chunks (List[str]): A list of strings representing the retrieved chunks of text.

    Returns:
        - str: The generated answer to the input query.
    """
    context = "\n".join(retrieved_chunks)
    prompt = prompt = f"""
    You are an AI assistant to help in retrieving and analyzing information from documents.

    ### Instructions:
    - Analyze the context carefully and answer the user's question based on the provided information.
    - If the context does not contain enough information to answer the question, **do not make assumptions**. Instead, clearly state that there is insufficient information to answer the query.
    - Provide a **clear, accurate, concise**, and **well-structured answer** based solely on the context provided.

    ### Context:
    {context}

    ### User Query:
    {query}

    ### AI Response:
    """

    
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
    )
    
    return response.choices[0].message.content.strip()

