"""Build labeled (bug report, code chunk, relevant/not) training pairs for fine-tuning a
CodeBERT-style relevance classifier (Phase B of the fine-tuning/prompt-optimization plan --
see docs/relevance_feedback_scoping.md's 2026-08-21 update for why: the zero-shot LLM
relevance-judgment path was confirmed to be a genuine mechanism-quality problem, not a
prompt-wording or output-format issue, which raises the case for a trained classifier
instead of further LLM-prompt iteration).

Deliberately diverges from IQLoc's own negative-sampling strategy (random methods from
*other* systems) in favor of hard negatives: non-ground-truth chunks from the SAME bug's own
BM25 candidate pool. Defended, not accidental -- at inference time the classifier's actual
job is discriminating among already-retrieved, lexically-similar candidates for one specific
bug, so training against easy random-system negatives would teach an easier, less relevant
task. Keeps IQLoc's own 4:1 negative:positive ratio.

Same file-level-ground-truth-implies-relevant labeling as
scripts/build_relevance_dev_set.py (Bench4BL has no chunk/line-level ground truth) --
same noisy-proxy caveat applies here, worth remembering when interpreting eval numbers
downstream. Holdout-safe against every eval manifest, same mechanism as the dev-set builder.

Local, CPU-only -- BM25 candidate retrieval needs no GPU/embedding model, deliberately not
an MN5 job (MN5's 20-CPU-per-GPU allocation ratio would be wasted on pure-CPU work).
"""

import os
import sys
import argparse
import json
import hashlib
import logging
import random
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.bench4bl import Bench4BL
from dataset.repo_cache import get_file_contents_batch
from dataset.utils import setup_logging, get_logger
from evaluation.manifest import select_diverse_manifest, DEFAULT_MANIFEST_DIR
from method.embedding_retriever import _chunk_file_content

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_relevance_feedback_test import _initial_ranking  # noqa: E402

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

DEFAULT_HOLDOUT_PATH = os.path.join(DEFAULT_MANIFEST_DIR, "bench4bl_eval_holdout_ids.json")
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "classifier_training_data"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--size', type=int, default=1200, help='Number of training bug instances (not chunks). Plan targets 1000-1500.')
    parser.add_argument('--seed', type=int, default=44, help='Distinct from every eval manifest seed (42) and the dev-set seed (43).')
    parser.add_argument('--max-per-repo', type=int, default=50, help='Higher than eval manifests\' (2-40) since 1200 across ~30 usable repos needs a looser per-repo cap to be reachable at all.')
    parser.add_argument('--min-distinct-repos', type=int, default=None)
    parser.add_argument('--candidate-pool-size', type=int, default=50, help='Matches the IQLoc-literal-mechanism track\'s config (the primary Phase-B4 evaluation target) -- see scripts/mn5/bench4bl_iqloc_approximation_array.sbatch.')
    parser.add_argument('--max-chunks-per-file', type=int, default=5)
    parser.add_argument('--max-positives-per-bug', type=int, default=10, help='Caps one large ground-truth file from dominating a single bug\'s training pairs.')
    parser.add_argument('--negative-ratio', type=int, default=4, help='Negatives per positive, matching IQLoc\'s own 4:1 ratio.')
    parser.add_argument('--holdout-ids', default=DEFAULT_HOLDOUT_PATH)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    holdout = json.load(open(args.holdout_ids))
    holdout_ids = set(holdout["instance_ids"])
    logger.info(f"Excluding {len(holdout_ids)} eval-manifest instance IDs")

    ds = Bench4BL()
    pool = ds.get_bug_instances(exclude_instance_ids=holdout_ids)
    logger.info(f"Holdout-safe pool: {len(pool)} instances")

    train_bugs = select_diverse_manifest(
        pool, args.size, seed=args.seed, max_per_repo=args.max_per_repo, min_distinct_repos=args.min_distinct_repos
    )
    train_ids = sorted(b.instance_id for b in train_bugs)
    overlap = set(train_ids) & holdout_ids
    assert not overlap, f"Training pool leaked {len(overlap)} holdout IDs -- aborting: {sorted(overlap)[:5]}"
    logger.info(f"Selected {len(train_bugs)} training bugs across {len(set(b.repo for b in train_bugs))} repos")

    records = []
    bugs_with_no_positives = 0
    for i, bug in enumerate(train_bugs):
        candidates = _initial_ranking(
            bug, retriever="bm25", candidate_pool_size=args.candidate_pool_size,
            embedding_model=None, rrf_weights=None, bm25_repr="skeleton",
        )
        contents = get_file_contents_batch(bug.repo, bug.base_commit, candidates)
        gt_set = set(bug.ground_truths)

        positive_chunks = []
        negative_chunks = []
        for path in candidates:
            file_chunks = _chunk_file_content(contents.get(path), path=path)
            is_gt_file = path in gt_set
            for idx, chunk_text in enumerate(file_chunks[: args.max_chunks_per_file]):
                entry = {"file": path, "chunk_index": idx, "chunk_text": chunk_text}
                (positive_chunks if is_gt_file else negative_chunks).append(entry)

        if not positive_chunks:
            bugs_with_no_positives += 1
            continue

        rng.shuffle(positive_chunks)
        positive_chunks = positive_chunks[: args.max_positives_per_bug]

        n_negatives_wanted = len(positive_chunks) * args.negative_ratio
        rng.shuffle(negative_chunks)
        negative_chunks = negative_chunks[:n_negatives_wanted]

        for entry in positive_chunks:
            records.append({
                "bug_instance_id": bug.instance_id, "repo": bug.repo,
                "file": entry["file"], "chunk_index": entry["chunk_index"], "chunk_text": entry["chunk_text"],
                "bug_report": bug.bug_report, "hints_text": bug.hints_text, "label": True,
            })
        for entry in negative_chunks:
            records.append({
                "bug_instance_id": bug.instance_id, "repo": bug.repo,
                "file": entry["file"], "chunk_index": entry["chunk_index"], "chunk_text": entry["chunk_text"],
                "bug_report": bug.bug_report, "hints_text": bug.hints_text, "label": False,
            })

        if (i + 1) % 50 == 0:
            logger.info(f"[{i + 1}/{len(train_bugs)}] {len(records)} records so far")

    n_positive = sum(1 for r in records if r["label"])
    n_total = len(records)
    logger.info(
        f"{n_total} chunks total, {n_positive} positive ({n_positive / n_total:.1%}) "
        f"across {len(set(r['bug_instance_id'] for r in records))} bugs "
        f"({bugs_with_no_positives} bugs skipped -- no ground-truth file found in their candidate pool)"
        if n_total else "0 chunks produced"
    )

    digest = hashlib.sha256(",".join(train_ids).encode()).hexdigest()[:12]
    dataset_id = f"bench4bl-classifier-train-n{len(train_bugs)}-s{args.seed}-{digest}"

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, f"{dataset_id}.jsonl")
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    provenance = {
        "dataset_id": dataset_id,
        "seed": args.seed,
        "size": len(train_bugs),
        "bugs_with_no_positives": bugs_with_no_positives,
        "distinct_repos": len(set(b.repo for b in train_bugs)),
        "candidate_pool_size": args.candidate_pool_size,
        "max_chunks_per_file": args.max_chunks_per_file,
        "max_positives_per_bug": args.max_positives_per_bug,
        "negative_ratio": args.negative_ratio,
        "num_chunks": n_total,
        "num_positive_chunks": n_positive,
        "holdout_ids_source": args.holdout_ids,
        "instance_ids": train_ids,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = os.path.join(DEFAULT_MANIFEST_DIR, f"{dataset_id}.json")
    with open(manifest_path, "w") as f:
        json.dump(provenance, f, indent=2)

    logger.info(f"Saved {n_total} chunk records to {jsonl_path}")
    logger.info(f"Saved provenance manifest to {manifest_path}")


if __name__ == "__main__":
    main()
