from classes.inverted_index import InvertedIndex
from utils.constants import BM25_K1

def get_bm25_idf(term: str) -> float:
    index = InvertedIndex()
    index.load()
    
    bm25idf = index.get_bm25_idf(term)
    return bm25idf

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1) -> float:
    index = InvertedIndex()
    index.load()
    
    bm25tf = index.get_bm25_tf(doc_id, term, k1)
    return bm25tf