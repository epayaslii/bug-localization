"""Merges the per-shard JSON files written by run_dense_retrieval_shard.py.

Usage:
    python scripts/aggregate_dense_retrieval_shards.py --shard-dir <dir> --num-shards 8 --output <path>
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.screening import summarize_screening


def _fix_int_keys(per_instance):
    for r in per_instance:
        r["hit_at"] = {int(k): v for k, v in r["hit_at"].items()}
        r["recall_at"] = {int(k): v for k, v in r["recall_at"].items()}
    return per_instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard-dir', required=True)
    parser.add_argument('--num-shards', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-missing', action='store_true')
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(os.path.join(args.shard_dir, "shard_*.json")))
    found_indices = set()
    manifest_id = None
    model = None
    merged_per_instance = []
    all_instance_ids = []

    for path in shard_paths:
        with open(path) as f:
            shard = json.load(f)
        idx = shard["shard_index"]
        found_indices.add(idx)
        if not shard.get("configs"):
            continue
        manifest_id = manifest_id or shard["manifest_id"]
        model = model or shard.get("model")
        all_instance_ids.extend(shard["instance_ids"])
        per_instance = _fix_int_keys(shard["configs"]["dense_embedding"]["screening_report"]["per_instance"])
        merged_per_instance.extend(per_instance)

    missing = sorted(set(range(args.num_shards)) - found_indices)
    if missing and not args.allow_missing:
        raise SystemExit(
            f"Missing {len(missing)}/{args.num_shards} shards: {missing}. Re-run those array "
            f"indices, or pass --allow-missing to aggregate the partial result anyway."
        )
    if missing:
        print(f"WARNING: aggregating with {len(missing)}/{args.num_shards} shards missing: {missing}")

    difficulty_counts = Counter(r["difficulty"] for r in merged_per_instance)
    screening_report = {
        "per_instance": merged_per_instance,
        "difficulty_distribution": dict(difficulty_counts),
        "total": len(merged_per_instance),
    }
    summary = summarize_screening(screening_report)

    result = {
        "manifest_id": manifest_id,
        "model": model,
        "mode": "full_corpus_dense_only",
        "num_shards": args.num_shards,
        "shards_found": len(found_indices),
        "total_instances": len(set(all_instance_ids)),
        "configs": {"dense_embedding": {"screening_report": screening_report, "summary": summary}},
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Aggregated {len(found_indices)}/{args.num_shards} shards, "
          f"{len(set(all_instance_ids))} unique instances -> {args.output}")
    print(f"Hit@1={summary['macro_hit_at'][1]:.3f} Hit@5={summary['macro_hit_at'][5]:.3f} "
          f"Hit@10={summary['macro_hit_at'][10]:.3f} MRR={summary['mrr']:.4f} MAP={summary['map']:.4f}")


if __name__ == "__main__":
    main()
