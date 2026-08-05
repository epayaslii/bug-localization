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


def screen_bug_instance(bug, token=None, cache=None):
    """Run path-only BM25 over the full corpus for one bug instance."""
    classifications = classify_bug_instance(bug, token=token, cache=cache)
    localizable_gts = [p for p, c in classifications.items() if c in LOCALIZABLE_CLASSES]

    if not localizable_gts:
        return {
            "instance_id": bug.instance_id,
            "repo": bug.repo,
            "classifications": classifications,
            "best_rank": None,
            "hit_at": {k: 0 for k in HIT_KS},
            "recall_at": {k: 0.0 for k in RECALL_KS},
            "difficulty": "no_localizable_gt",
        }

    ranked = rank_files_bm25(bug.bug_report, bug.code_files, top_k=None)
    rank_of = {path: i + 1 for i, path in enumerate(ranked)}

    gt_ranks = [rank_of[p] for p in localizable_gts if p in rank_of]
    best_rank = min(gt_ranks) if gt_ranks else None

    return {
        "instance_id": bug.instance_id,
        "repo": bug.repo,
        "classifications": classifications,
        "best_rank": best_rank,
        "hit_at": {k: int(any(r <= k for r in gt_ranks)) for k in HIT_KS},
        "recall_at": {k: (sum(1 for r in gt_ranks if r <= k) / len(localizable_gts)) for k in RECALL_KS},
        "difficulty": _difficulty_band(best_rank),
    }


def screen_manifest(bugs, token=None, cache=None):
    """Screen every bug instance in `bugs` and aggregate the difficulty distribution."""
    results = []
    for i, bug in enumerate(bugs):
        results.append(screen_bug_instance(bug, token=token, cache=cache))
        if (i + 1) % 10 == 0:
            logger.info(f"Screened {i + 1}/{len(bugs)} instances")

    difficulty_counts = Counter(r["difficulty"] for r in results)
    return {
        "per_instance": results,
        "difficulty_distribution": dict(difficulty_counts),
        "total": len(results),
    }
