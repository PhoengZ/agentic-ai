from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

@tool
def search_knowledge(query: str, k: int = 5)->str:
    """
    Search the knowledge base for relevant information.
    
    Args:
        query: The specific keyword or phrase to search for. (Extract this from user's question).
        k: The number of top relevant chunks to return. 
           - Use k=3 for simple, specific factual questions.
           - Use k=5 (default) for general questions.
           - Use k=10 if the user asks for a comprehensive summary, list, or detailed explanation.
    """
    try:
        with open("knowledge_base.txt", "r", encoding="utf-8") as f:
            contents = f.read()
    except Exception as e:
        return f"Failed to read the knowledge file: {e}" 
    text_split = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len
    )

    chunks = text_split.split_text(contents)
    # Knowledge are in english format
    chunk_tokenized = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(chunk_tokenized)

    query_tokenized  = query.lower().split()

    scores = bm25.get_scores(query_tokenized)

    relevant_chunk = []
    for idx, chunk in enumerate(chunks):
        if scores[idx] > 0:
            relevant_chunk.append({
                "chunk": chunk,
                "score": scores[idx] 
            }) 

    relevant_chunk.sort(key=lambda x: x["score"], reverse=True)
    
    if relevant_chunk:
        top_k_chunks = relevant_chunk[:k]
        top_k_chunks_str = [f"Score: {item['score']}\nChunk: {item['chunk']}" for item in top_k_chunks]
        return "\n\n".join(top_k_chunks_str)
    else:
        return "Not found in knowledge base"