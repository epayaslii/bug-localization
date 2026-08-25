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
from method.fusion_signals import rank_files_ast_similarity, rank_files_commit_recency, rank_files_dependency_graph
from method.commit_history_retriever import rank_files_commit_history_scored

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


def rank_files_hybrid_with_history_rerank(
    bug,
    top_k: int | None = 100,
    candidate_pool_size: int = 200,
    embedding_model: str = "microsoft/unixcoder-base",
    rrf_k: int = 60,
    weights: list[float] | None = None,
    bm25_rank_fn=None,
) -> tuple[list[str], dict]:
    """Like rank_files_hybrid, but fuses commit-history as a genuine THIRD RRF ranking signal
    instead of unioning history-matched files into the candidate pool
    (rank_files_bm25_with_history_union). The union approach buries every history-only file
    at the tail of the pool regardless of match strength (confirmed real recall gain,
    Recall@200 +18.4%, but net MRR -4% once wired into the full pipeline, 2026-08-20) --
    RRF fusion instead lets a strong commit-history match actually rank near the top, the
    same way a strong BM25 or embedding match would.

    `weights` order is [bm25, embedding, commit_history]; defaults to equal weight -- an
    unweighted starting point to ablate from, not a tuned config (this project's own history
    with rank_files_hybrid shows equal weighting is rarely the best fusion ratio).
    """
    bm25_rank_fn = bm25_rank_fn or (lambda b: rank_files_bm25_with_symbols(b, top_k=candidate_pool_size))

    t0 = time.time()
    bm25_candidates = bm25_rank_fn(bug)
    t_bm25 = time.time() - t0

    if not bm25_candidates:
        return bm25_candidates, {"bm25_s": t_bm25}

    candidate_bug = bug.model_copy(update={"code_files": bm25_candidates})
    embedding_ranking, embed_timing = rank_files_embedding_chunked(
        candidate_bug, top_k=None, model_name=embedding_model
    )

    t1 = time.time()
    history_ranking = rank_files_commit_history_scored(bug)
    t_history = time.time() - t1

    fused = reciprocal_rank_fusion([bm25_candidates, embedding_ranking, history_ranking], k=rrf_k, weights=weights)
    ranked = fused[:top_k] if top_k is not None else fused

    timing = {"bm25_s": t_bm25, "history_s": t_history, **embed_timing}
    return ranked, timing


def rank_files_hybrid_extended(
    bug,
    top_k: int | None = 100,
    candidate_pool_size: int = 200,
    embedding_model: str = "microsoft/unixcoder-base",
    rrf_k: int = 60,
    weights: list[float] | None = None,
) -> tuple[list[str], dict]:
    """Like rank_files_hybrid, but also fuses the three Phase 4.1 signals (AST-similarity,
    dependency-graph, commit-recency -- see method/fusion_signals.py) alongside BM25 and
    chunked embedding. All five signals are computed over the SAME BM25 candidate pool, not
    the full corpus, for cost and comparability with rank_files_hybrid.

    `weights` order is [bm25, embedding, ast_similarity, dependency_graph, commit_recency];
    defaults to equal weight. This is deliberately the naive unweighted baseline to ablate
    from -- rank_files_hybrid's own history (results/README.md §4) shows equal weighting is
    often NOT the best fusion weight, so an equal-weight result here should be read the same
    way: a starting point, not a tuned config. rank_files_hybrid itself is left untouched
    (this is a new function, not a modification) so the already-confirmed 2-signal result
    (weighted RRF 1:10, MRR 0.281) stays comparable and isn't put at risk.
    """
    bm25_candidates = rank_files_bm25_with_symbols(bug, top_k=candidate_pool_size)
    if not bm25_candidates:
        return bm25_candidates, {}

    candidate_bug = bug.model_copy(update={"code_files": bm25_candidates})

    t0 = time.time()
    embedding_ranking, embed_timing = rank_files_embedding_chunked(
        candidate_bug, top_k=None, model_name=embedding_model
    )
    t_embed = time.time() - t0

    t0 = time.time()
    ast_ranking = rank_files_ast_similarity(candidate_bug, top_k=None)
    t_ast = time.time() - t0

    t0 = time.time()
    dependency_ranking = rank_files_dependency_graph(candidate_bug, bm25_candidates, top_k=None)
    t_dependency = time.time() - t0

    t0 = time.time()
    recency_ranking = rank_files_commit_recency(candidate_bug, top_k=None)
    t_recency = time.time() - t0

    rankings = [bm25_candidates, embedding_ranking, ast_ranking, dependency_ranking, recency_ranking]
    fused = reciprocal_rank_fusion(rankings, k=rrf_k, weights=weights)
    ranked = fused[:top_k] if top_k is not None else fused

    timing = {
        "embed_s": t_embed, "ast_similarity_s": t_ast,
        "dependency_graph_s": t_dependency, "commit_recency_s": t_recency,
        **{k: v for k, v in embed_timing.items() if k != "embed_s"},
    }
    return ranked, timing
