"""Hybrid retrieval: BM25 candidate generation, cascaded with chunked-embedding reranking.

Full-corpus dense embedding is expensive (every candidate file, times however many chunks
each has) and, per embedding_retriever.py's whole-file ceiling test, didn't even beat BM25
when tried standalone on the full corpus. Rather than embedding the entire corpus in
parallel with BM25 and fusing two independent full rankings, this narrows to BM25's own
top candidate_pool_size files first (cheap), then chunk-embeds only those (see
rank_files_embedding_chunked -- AST-based chunks; the literature documents chunked
embedding scoring dramatically higher than whole-file), and fuses the two rankings via
Reciprocal Rank Fusion. This can only rerank within BM25's own candidate pool; it can never
recover a file BM25's first pass excluded entirely -- Hit@k/recall@k for k beyond
candidate_pool_size are therefore not meaningful for this ranker.
"""

import time

from dataset.utils import get_logger
from method.bm25_retriever import rank_files_bm25_with_symbols
from method.embedding_retriever import rank_files_embedding_chunked

logger = get_logger(__name__)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[str]:
    """Combine multiple ranked file-path lists into one fused ranking via (optionally
    weighted) Reciprocal Rank Fusion: each file's fused score is the sum of
    weight_i / (k + rank) across every input ranking it appears in (files absent from a
    ranking contribute 0 for that one, not a penalty). k=60 is the standard RRF constant,
    also used by the BLAZE paper's BM25+dense fusion. `weights` defaults to equal weight
    (1.0) per ranking, i.e. standard unweighted RRF -- pass e.g. [1.0, 2.0] to favor the
    second ranking (see method/embedding_retriever.py's chunked embedding beating
    unweighted RRF at n=30 in results/README.md §4, motivating this parameter).
    """
    weights = weights if weights is not None else [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for weight, ranking in zip(weights, rankings):
        for rank, path in enumerate(ranking, start=1):
            scores[path] = scores.get(path, 0.0) + weight / (k + rank)
    return sorted(scores, key=lambda p: scores[p], reverse=True)


def rank_files_hybrid(
    bug,
    top_k: int | None = 100,
    candidate_pool_size: int = 200,
    embedding_model: str = "microsoft/unixcoder-base",
    rrf_k: int = 60,
    weights: list[float] | None = None,
    bm25_rank_fn=None,
) -> tuple[list[str], dict]:
    """Rank bug.code_files via BM25 (symbols representation) -> chunked-embedding rerank of
    that candidate pool -> Reciprocal Rank Fusion of the two. Returns (ranked_file_paths,
    timing_info). Pass top_k=None to get the full fused ranking of the candidate pool
    (size candidate_pool_size, NOT the full original corpus). `weights` is
    [bm25_weight, embedding_weight] passed through to reciprocal_rank_fusion (default
    unweighted 1:1) -- e.g. [1.0, 5.0] to favor the embedding ranking, matching whichever
    ratio scored best for a given dataset/model (see docs/hybrid_rrf_qwen3_result.md-style
    per-dataset sweep results before picking a value blind).
    """
    bm25_rank_fn = bm25_rank_fn or (lambda b: rank_files_bm25_with_symbols(b, top_k=candidate_pool_size))

    t0 = time.time()
    bm25_candidates = bm25_rank_fn(bug)
    t_bm25 = time.time() - t0

    if not bm25_candidates:
        return bm25_candidates, {"bm25_s": t_bm25}

    # Rerank only the BM25 candidate pool with chunked embeddings, not the full corpus.
    candidate_bug = bug.model_copy(update={"code_files": bm25_candidates})
    embedding_ranking, embed_timing = rank_files_embedding_chunked(
        candidate_bug, top_k=None, model_name=embedding_model
    )

    fused = reciprocal_rank_fusion([bm25_candidates, embedding_ranking], k=rrf_k, weights=weights)
    ranked = fused[:top_k] if top_k is not None else fused

    timing = {"bm25_s": t_bm25, **embed_timing}
    return ranked, timing
