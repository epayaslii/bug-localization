"""Path-only BM25 screening over an evaluation manifest.

For each bug instance, ranks the full before-fix corpus by BM25 relevance to the bug
report and reports the best rank among its *localizable* ground-truth files (see
dataset/localizability.py) -- files added by the fix itself are excluded, since no
retrieval method could ever find them, and counting them as misses would understate
real recall.
"""

from collections import Counter

from method.bm25_retriever import rank_files_bm25
from dataset.localizability import classify_bug_instance, LOCALIZABLE_CLASSES
from dataset.utils import get_logger

logger = get_logger(__name__)

RECALL_KS = (100, 200)
HIT_KS = (1, 5, 10, 100, 200)


def _difficulty_band(best_rank):
    """Difficulty bands over the best localizable-GT rank. Thresholds follow the Hit@10/
    Hit@100/Hit@200 cutoffs already tracked elsewhere in this pipeline; there's no
    universal standard for the easy/medium/hard split, so these are a judgment call."""
    if best_rank is None:
        return "no_localizable_gt"
    if best_rank <= 10:
        return "easy"
    if best_rank <= 100:
        return "medium"
    if best_rank <= 200:
        return "hard"
    return "outside_top200"


def _default_rank_fn(bug):
    return rank_files_bm25(bug.bug_report, bug.code_files, top_k=None)


def _average_precision(gt_ranks: dict, num_localizable_gts: int) -> float:
    """Average Precision for one instance: for each localizable GT found in the ranking
    (in increasing rank order), precision-at-that-rank = (how many GTs found so far,
    including this one) / rank. AP is the mean of those precisions over ALL localizable
    GTs for the instance -- GTs never found in the ranking contribute 0 to the sum but
    still count in the denominator, so missing ground truth is penalized even though it
    never appears in gt_ranks.
    """
    if num_localizable_gts == 0:
        return 0.0
    sorted_ranks = sorted(gt_ranks.values())
    precision_sum = sum((i + 1) / rank for i, rank in enumerate(sorted_ranks))
    return precision_sum / num_localizable_gts


def screen_bug_instance(bug, token=None, cache=None, rank_fn=None):
    """Run BM25 over the full corpus for one bug instance and rank it.

    `rank_fn(bug) -> list[str]` controls how files are scored/ranked -- defaults to
    path-only BM25 (rank_files_bm25). Pass e.g. `lambda b: rank_files_bm25_with_symbols(b,
    top_k=None)` to screen a different document representation on the same instances,
    for direct comparison (see scripts/compare_bm25_representations.py).
    """
    rank_fn = rank_fn or _default_rank_fn
    classifications = classify_bug_instance(bug, token=token, cache=cache)
    localizable_gts = [p for p, c in classifications.items() if c in LOCALIZABLE_CLASSES]

    if not localizable_gts:
        return {
            "instance_id": bug.instance_id,
            "repo": bug.repo,
            "classifications": classifications,
            "gt_ranks": {},
            "best_rank": None,
            "hit_at": {k: 0 for k in HIT_KS},
            "recall_at": {k: 0.0 for k in RECALL_KS},
            "average_precision": 0.0,
            "difficulty": "no_localizable_gt",
        }

    ranked = rank_fn(bug)
    rank_of = {path: i + 1 for i, path in enumerate(ranked)}

    gt_rank_map = {p: rank_of[p] for p in localizable_gts if p in rank_of}
    gt_ranks = list(gt_rank_map.values())
    best_rank = min(gt_ranks) if gt_ranks else None

    return {
        "instance_id": bug.instance_id,
        "repo": bug.repo,
        "classifications": classifications,
        "gt_ranks": gt_rank_map,
        "best_rank": best_rank,
        "hit_at": {k: int(any(r <= k for r in gt_ranks)) for k in HIT_KS},
        "recall_at": {k: (sum(1 for r in gt_ranks if r <= k) / len(localizable_gts)) for k in RECALL_KS},
        "average_precision": _average_precision(gt_rank_map, len(localizable_gts)),
        "difficulty": _difficulty_band(best_rank),
    }


def screen_manifest(bugs, token=None, cache=None, rank_fn=None):
    """Screen every bug instance in `bugs` and aggregate the difficulty distribution."""
    results = []
    for i, bug in enumerate(bugs):
        results.append(screen_bug_instance(bug, token=token, cache=cache, rank_fn=rank_fn))
        if (i + 1) % 10 == 0:
            logger.info(f"Screened {i + 1}/{len(bugs)} instances")

    difficulty_counts = Counter(r["difficulty"] for r in results)
    return {
        "per_instance": results,
        "difficulty_distribution": dict(difficulty_counts),
        "total": len(results),
    }


def summarize_screening(screening_report):
    """Macro-averaged Hit@k, MRR, and MAP over a screen_manifest() report, for comparing
    BM25 document representations against each other (e.g. path-only vs symbols vs
    symbols+imports) on the same manifest. Instances with no localizable ground truth
    contribute 0 to every metric, consistent with their all-zero hit_at/recall_at.
    """
    results = screening_report["per_instance"]
    n = len(results) or 1

    macro_hit_at = {k: sum(r["hit_at"].get(k, 0) for r in results) / n for k in HIT_KS}
    macro_recall_at = {k: sum(r["recall_at"].get(k, 0.0) for r in results) / n for k in RECALL_KS}
    mrr = sum((1.0 / r["best_rank"]) if r["best_rank"] else 0.0 for r in results) / n
    map_score = sum(r.get("average_precision", 0.0) for r in results) / n

    return {
        "n": len(results),
        "macro_hit_at": macro_hit_at,
        "macro_recall_at": macro_recall_at,
        "mrr": mrr,
        "map": map_score,
    }
