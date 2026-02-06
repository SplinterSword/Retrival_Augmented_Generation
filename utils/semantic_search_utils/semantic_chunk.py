import re

def semantic_chunk(text: str, sentence_chunk_size: int = 4, overlap: int = 1):
    # Strip leading and trailing whitespace from input text
    stripped_text = text.strip()
    
    # If there's nothing left after stripping, return an empty list
    if not stripped_text:
        return []
    
    # Split sentences using regex
    splited_text = re.split(r'(?<=[.!?])\s+', stripped_text)
    
    # After splitting sentences, if there's only one sentence and it doesn't end with punctuation
    if len(splited_text) == 1 and not splited_text[0].endswith(('.', '!', '?')):
        # Treat the whole text as one sentence
        splited_text = [stripped_text]
    
    # Strip leading and trailing whitespace from each sentence and filter empty ones
    cleaned_sentences = []
    for sentence in splited_text:
        stripped_sentence = sentence.strip()
        if stripped_sentence:  # Only use chunks that still have content after stripping
            cleaned_sentences.append(stripped_sentence)
    
    # If no sentences left after cleaning, return empty list
    if not cleaned_sentences:
        return []
    
    overlap = overlap
            
    chunks = []
            
    for i in range(0, len(cleaned_sentences), sentence_chunk_size - overlap):
        chunk = ' '.join(cleaned_sentences[i:i+sentence_chunk_size])
        if chunk.strip():  # Ensure chunk has content
            chunks.append(chunk)
    
    return chunks