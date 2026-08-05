"""Retrieval-vs-reranking failure attribution.

A low final score for a bug instance can come from two different places, and they
call for opposite fixes:

  FAILURE MODE 1 (retrieval failure) -- the correct file never enters the candidate
  set BM25 hands to the LLM. No amount of reranking can fix this; the retrieval
  stage itself needs to improve (better tokenization, symbol-level matching, a
  larger top_k, etc).

  FAILURE MODE 2 (reranking failure) -- the correct file IS in the candidate set,
  but the LLM doesn't surface it in the final ranked output. This calls for
  prompt/model changes, not retrieval changes.

`classify_retrieval_reach` splits these two using only the BM25 ranking already
computed by evaluation/screening.py -- no LLM call, no cost. `run_oracle_diagnostic`
isolates pure reranking ability by force-injecting every localizable ground-truth
file into the candidate set (so retrieval recall is artificially 100%) and asking
whether the LLM places them well; this DOES call the LLM and therefore costs API
calls, so callers should treat it as opt-in.
"""

from dataset.localizability import classify_bug_instance, LOCALIZABLE_CLASSES
from dataset.utils import get_logger

logger = get_logger(__name__)

REACHED_CANDIDATE_SET = "reached_candidate_set"
RETRIEVAL_FAILURE = "retrieval_failure"


def classify_retrieval_reach(screen_result, candidate_size=100):
    """Given one screen_bug_instance() result, classify each of its localizable
    ground-truth files as REACHED_CANDIDATE_SET (rank <= candidate_size, so an LLM
    reranker over that candidate set could in principle surface it) or
    RETRIEVAL_FAILURE (rank > candidate_size, or never ranked at all -- the LLM
    literally never sees it). Purely offline; reuses screening's precomputed ranks.
    """
    gt_ranks = screen_result.get("gt_ranks", {})
    return {
        path: (REACHED_CANDIDATE_SET if rank <= candidate_size else RETRIEVAL_FAILURE)
        for path, rank in gt_ranks.items()
    }


def summarize_retrieval_reach(screening_report, candidate_size=100):
    """Aggregate classify_retrieval_reach over every instance in a screen_manifest()
    report. Returns per-instance-file counts of reached-candidate-set vs
    retrieval-failure, plus how many instances have zero reachable localizable GTs
    (every localizable GT is a retrieval failure -- reranking can't possibly help).
    """
    reach_counts = {REACHED_CANDIDATE_SET: 0, RETRIEVAL_FAILURE: 0}
    instances_fully_unreachable = 0
    instances_with_localizable_gt = 0

    for result in screening_report["per_instance"]:
        gt_ranks = result.get("gt_ranks", {})
        if not gt_ranks:
            continue
        instances_with_localizable_gt += 1
        reach = classify_retrieval_reach(result, candidate_size=candidate_size)
        for status in reach.values():
            reach_counts[status] += 1
        if all(status == RETRIEVAL_FAILURE for status in reach.values()):
            instances_fully_unreachable += 1

    return {
        "candidate_size": candidate_size,
        "file_level_counts": reach_counts,
        "instances_with_localizable_gt": instances_with_localizable_gt,
        "instances_fully_unreachable": instances_fully_unreachable,
    }


def prepare_oracle_candidate_set(bug, candidate_size, cache=None, token=None):
    """Build the candidate set for the oracle diagnostic: the BM25 top candidate_size
    files, with every localizable ground-truth file force-injected if BM25 didn't
    already place it there. This guarantees retrieval recall = 100% for this bug,
    isolating reranking ability from retrieval quality. No LLM call.
    """
    from method.bm25_retriever import rank_files_bm25

    classifications = classify_bug_instance(bug, token=token, cache=cache)
    localizable_gts = [p for p, c in classifications.items() if c in LOCALIZABLE_CLASSES]

    top_candidates = rank_files_bm25(bug.bug_report, bug.code_files, top_k=candidate_size)
    candidate_set = list(top_candidates)
    injected = [p for p in localizable_gts if p not in candidate_set]
    candidate_set.extend(injected)

    return candidate_set, injected


def run_oracle_diagnostic(bug, localizer, candidate_size=100, token=None, cache=None):
    """Run the LLM reranker on an oracle candidate set (retrieval recall forced to
    100% via prepare_oracle_candidate_set) and report whether it places the
    localizable ground-truth files well. THIS CALLS THE LLM -- costs one API call
    per bug instance. Restores bug.code_files afterward regardless of outcome.

    Returns None (and logs a warning) if the bug has no localizable ground truth,
    since there's nothing for the oracle test to measure.
    """
    classifications = classify_bug_instance(bug, token=token, cache=cache)
    localizable_gts = [p for p, c in classifications.items() if c in LOCALIZABLE_CLASSES]
    if not localizable_gts:
        logger.warning(f"{bug.instance_id}: no localizable ground truth, skipping oracle diagnostic")
        return None

    candidate_set, injected = prepare_oracle_candidate_set(
        bug, candidate_size, cache=cache, token=token
    )

    original_code_files = bug.code_files
    try:
        bug.code_files = candidate_set
        response = localizer.localize(bug)
    finally:
        bug.code_files = original_code_files

    predicted = response.candidate_files if response else []
    top1_hit = int(bool(predicted) and predicted[0] in localizable_gts)
    top10_hit_count = sum(1 for p in predicted[:10] if p in localizable_gts)

    return {
        "instance_id": bug.instance_id,
        "repo": bug.repo,
        "candidate_set_size": len(candidate_set),
        "injected_count": len(injected),
        "localizable_gt_count": len(localizable_gts),
        "predicted_candidate_files": predicted,
        "top1_hit": top1_hit,
        "top10_hit_count": top10_hit_count,
        "top10_hit_fraction": top10_hit_count / len(localizable_gts),
    }
