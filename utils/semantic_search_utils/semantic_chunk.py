import re

def semantic_chunk(text: str, sentence_chunk_size: int = 4, overlap: int = 1):
    splited_text = re.split(r'(?<=[.!?])\s+', text)
    overlap = overlap
            
    chunks = []
            
    for i in range(0, len(splited_text), sentence_chunk_size - overlap):
        chunks.append(' '.join(splited_text[i:i+sentence_chunk_size]))
    
    return chunks