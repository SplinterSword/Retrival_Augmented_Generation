from sentence_transformers import SentenceTransformer

class SemanticSearch:
    modal = None
    
    def __init__(self):
        self.modal = SentenceTransformer('all-MiniLM-L6-v2')
    
    def encode_text(self, text: str):
        return self.modal.encode(text)


def verify_modal():
    try:
        sematic_search = SemanticSearch()
        print(f"Model loaded: {sematic_search.modal}")
        print(f"Max sequence length: {sematic_search.modal.max_seq_length}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False