"""Merges the per-shard JSON files written by run_hybrid_rrf_weighting_shard.py (one Slurm
array job) into a single result file with the same shape run_hybrid_rrf_weighting_test.py
would have produced in one long serial run -- so downstream analysis doesn't need to know
the run was sharded.

Usage:
    python scripts/aggregate_rrf_shards.py --shard-dir <dir> --num-shards 50 --output <path>
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.screening import summarize_screening, HIT_KS, RECALL_KS


def _fix_int_keys(per_instance):
    """JSON round-trips int dict keys (hit_at/recall_at) as strings -- convert back so
    summarize_screening()'s int-keyed .get(k, ...) lookups actually match."""
    for r in per_instance:
        r["hit_at"] = {int(k): v for k, v in r["hit_at"].items()}
        r["recall_at"] = {int(k): v for k, v in r["recall_at"].items()}
    return per_instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard-dir', required=True)
    parser.add_argument('--num-shards', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-missing', action='store_true',
                         help='Aggregate whatever shards are present instead of failing on missing ones (for checking partial progress)')
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(os.path.join(args.shard_dir, "shard_*.json")))
    found_indices = set()
    manifest_id = None
    candidate_pool_size = None
    weight_configs = None
    merged_per_instance = {}  # config_name -> list of per_instance dicts
    all_instance_ids = []

    for path in shard_paths:
        with open(path) as f:
            shard = json.load(f)
        idx = shard["shard_index"]
        found_indices.add(idx)
        if not shard.get("configs"):
            continue  # empty shard placeholder
        manifest_id = manifest_id or shard["manifest_id"]
        candidate_pool_size = candidate_pool_size or shard.get("candidate_pool_size")
        weight_configs = weight_configs or shard.get("weight_configs")
        all_instance_ids.extend(shard["instance_ids"])
        for name, cfg in shard["configs"].items():
            per_instance = _fix_int_keys(cfg["screening_report"]["per_instance"])
            merged_per_instance.setdefault(name, []).extend(per_instance)

    missing = sorted(set(range(args.num_shards)) - found_indices)
    if missing and not args.allow_missing:
        raise SystemExit(
            f"Missing {len(missing)}/{args.num_shards} shards: {missing[:20]}"
            f"{'...' if len(missing) > 20 else ''}. Re-run those array indices, or pass "
            f"--allow-missing to aggregate the partial result anyway."
        )
    if missing:
        print(f"WARNING: aggregating with {len(missing)}/{args.num_shards} shards missing: {missing}")

    configs = {}
    for name, per_instance in merged_per_instance.items():
        difficulty_counts = Counter(r["difficulty"] for r in per_instance)
        screening_report = {
            "per_instance": per_instance,
            "difficulty_distribution": dict(difficulty_counts),
            "total": len(per_instance),
        }
        configs[name] = {
            "screening_report": screening_report,
            "summary": summarize_screening(screening_report),
        }

    result = {
        "manifest_id": manifest_id,
        "candidate_pool_size": candidate_pool_size,
        "weight_configs": weight_configs,
        "num_shards": args.num_shards,
        "shards_found": len(found_indices),
        "total_instances": len(set(all_instance_ids)),
        "configs": configs,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Aggregated {len(found_indices)}/{args.num_shards} shards, "
          f"{len(set(all_instance_ids))} unique instances -> {args.output}")
    print(f"{'config':<20} {'Hit@1':>7} {'Hit@5':>7} {'Hit@10':>7} {'Hit@100':>8} {'MRR':>8} {'MAP':>8}")
    for name, cfg in sorted(configs.items()):
        s = cfg["summary"]
        print(f"{name:<20} {s['macro_hit_at'][1]:>7.3f} {s['macro_hit_at'][5]:>7.3f} "
              f"{s['macro_hit_at'][10]:>7.3f} {s['macro_hit_at'][100]:>8.3f} {s['mrr']:>8.4f} {s['map']:>8.4f}")


if __name__ == "__main__":
    main()
