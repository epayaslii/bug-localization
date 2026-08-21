"""Build a small labeled chunk-relevance dev set for scoring LLM relevance-judgment prompt
variants (scripts/optimize_relevance_prompt.py) -- cheap classification-accuracy scoring
against derived labels, not a full retrieval+rerank cycle per variant.

Bench4BL has no chunk/line-level ground truth (BugInstance.line_mappings is never populated
for this dataset -- see dataset/models.py), so labels are derived the same way the pipeline's
own relevance-filtering functions treat file-level ground truth: any chunk belonging to a
ground-truth file is labeled relevant, every other candidate-pool chunk is labeled irrelevant.
This is a noisy proxy (a large ground-truth file's unrelated boilerplate chunks are labeled
relevant too), not a fix for the underlying data gap -- worth stating plainly wherever this
dev set's scores get reported, not just here.

Uses --seed 43 (different from every eval manifest's seed=42) and, more importantly, excludes
every instance ID in the Phase-0 holdout-IDs file (scripts/build_eval_holdout_ids.py's output)
so this dev set can never overlap with any manifest a real number has been reported against.
"""

import os
import sys
import argparse
import json
import hashlib
import logging
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
DEFAULT_DEV_SET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "relevance_dev_sets"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=30, help='Number of dev bug instances (not chunks).')
    parser.add_argument('--seed', type=int, default=43, help='Different from every eval manifest seed (42).')
    parser.add_argument('--max-per-repo', type=int, default=3)
    parser.add_argument('--min-distinct-repos', type=int, default=None)
    parser.add_argument('--candidate-pool-size', type=int, default=50, help='Matches the production IQLoc-branch LLM-relevance-judgment config (scripts/mn5/bench4bl_iqloc_approximation_array.sbatch), not the 100 used by the embedding-cosine track -- a larger pool here just produces a dev set that doesn\'t reflect the actual prompt size the LLM path runs with.')
    parser.add_argument('--max-chunks-per-file', type=int, default=5)
    parser.add_argument('--holdout-ids', default=DEFAULT_HOLDOUT_PATH)
    parser.add_argument('--output-dir', default=DEFAULT_DEV_SET_DIR)
    args = parser.parse_args()

    holdout = json.load(open(args.holdout_ids))
    holdout_ids = set(holdout["instance_ids"])
    logger.info(f"Excluding {len(holdout_ids)} eval-manifest instance IDs")

    ds = Bench4BL()
    pool = ds.get_bug_instances(exclude_instance_ids=holdout_ids)
    logger.info(f"Holdout-safe pool: {len(pool)} instances")

    dev_bugs = select_diverse_manifest(
        pool, args.size, seed=args.seed, max_per_repo=args.max_per_repo, min_distinct_repos=args.min_distinct_repos
    )
    dev_ids = sorted(b.instance_id for b in dev_bugs)
    overlap = set(dev_ids) & holdout_ids
    assert not overlap, f"Dev set leaked {len(overlap)} holdout IDs -- aborting: {sorted(overlap)[:5]}"
    logger.info(f"Selected {len(dev_bugs)} dev bugs across {len(set(b.repo for b in dev_bugs))} repos")

    records = []
    for bug in dev_bugs:
        candidates = _initial_ranking(
            bug, retriever="bm25", candidate_pool_size=args.candidate_pool_size,
            embedding_model=None, rrf_weights=None, bm25_repr="skeleton",
        )
        contents = get_file_contents_batch(bug.repo, bug.base_commit, candidates)
        gt_set = set(bug.ground_truths)
        for path in candidates:
            file_chunks = _chunk_file_content(contents.get(path), path=path)
            for idx, chunk_text in enumerate(file_chunks[: args.max_chunks_per_file]):
                records.append({
                    "bug_instance_id": bug.instance_id,
                    "repo": bug.repo,
                    "file": path,
                    "chunk_index": idx,
                    "chunk_text": chunk_text,
                    "bug_report": bug.bug_report,
                    "hints_text": bug.hints_text,
                    "label": path in gt_set,
                })

    n_positive = sum(1 for r in records if r["label"])
    n_total = len(records)
    logger.info(f"{n_total} chunks total, {n_positive} positive ({n_positive / n_total:.1%})" if n_total else "0 chunks produced")

    digest = hashlib.sha256(",".join(dev_ids).encode()).hexdigest()[:12]
    dev_set_id = f"bench4bl-relevance-dev-n{len(dev_bugs)}-s{args.seed}-{digest}"

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, f"{dev_set_id}.jsonl")
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    provenance = {
        "dev_set_id": dev_set_id,
        "seed": args.seed,
        "size": len(dev_bugs),
        "distinct_repos": len(set(b.repo for b in dev_bugs)),
        "candidate_pool_size": args.candidate_pool_size,
        "max_chunks_per_file": args.max_chunks_per_file,
        "num_chunks": n_total,
        "num_positive_chunks": n_positive,
        "holdout_ids_source": args.holdout_ids,
        "instance_ids": dev_ids,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = os.path.join(DEFAULT_MANIFEST_DIR, f"{dev_set_id}.json")
    with open(manifest_path, "w") as f:
        json.dump(provenance, f, indent=2)

    logger.info(f"Saved {n_total} chunk records to {jsonl_path}")
    logger.info(f"Saved provenance manifest to {manifest_path}")


if __name__ == "__main__":
    main()
