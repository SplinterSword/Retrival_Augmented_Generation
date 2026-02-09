def hybrid_score(bm25_score, semantic_score, alpha):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(bm25_rank, semantic_rank, k):
    bm25_recip_rank = 1 / (bm25_rank + k) if bm25_rank > 0 else 0
    semantic_recip_rank = 1 / (semantic_rank + k) if semantic_rank > 0 else 0
    return bm25_recip_rank + semantic_recip_rank