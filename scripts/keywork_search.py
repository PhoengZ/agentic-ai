from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

@tool
def search_knowledge(query)->str:
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
    
    relevant_chunk = []
    for idx, chunk in enumerate(chunks):
        if query.lower() in chunk.lower():
            relevant_chunk.append(f"chunk: {idx+1}\n\n {chunk}") 

    if relevant_chunk:
        return "\n".join(relevant_chunk)
    else:
        return "Not found in knowledge base"