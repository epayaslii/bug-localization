"""Array-job shard of run_hybrid_rrf_weighting_test.py, for running the full n=500
SWE-bench Verified manifest on MN5 as a Slurm array job instead of one long serial job.

Each shard processes only a slice of the manifest's instances (selected by --num-shards /
--shard-index) and writes its own JSON to --output-dir, atomically (write to a .tmp path,
then os.replace) so a task killed mid-write never leaves a corrupt/partial file behind.
A separate aggregation script (aggregate_rrf_shards.py) merges all shard JSONs afterward.

Same weight-sweep logic and config set as run_hybrid_rrf_weighting_test.py -- this file
intentionally duplicates that logic rather than importing it, since the two differ only in
instance selection and output shape (dir-of-shards vs. one file), and array jobs benefit
from having no runtime dependency on argument-parsing changes in the non-sharded script.
"""

import os
import sys
import time
import argparse
import json
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from dataset.swebench import SWEBench
from dataset.beetlebox import BeetleBox
from dataset.bench4bl import Bench4BL
from dataset.iqloc import IQLocExtended
from dataset.localizability import load_cache, save_cache
from dataset.utils import setup_logging, get_logger
from evaluation.manifest import load_manifest
from evaluation.screening import screen_manifest, summarize_screening
from method.bm25_retriever import (
    rank_files_bm25, rank_files_bm25_with_symbols, rank_files_bm25_with_skeleton, rank_files_bm25_refined,
)
from method.embedding_retriever import rank_files_embedding_chunked
from method.hybrid_retriever import reciprocal_rank_fusion

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

_BM25_REPR_FNS = {
    "path_only": lambda bug, top_k: rank_files_bm25(bug.bug_report, bug.code_files, top_k=top_k),
    "symbols_with_imports": lambda bug, top_k: rank_files_bm25_with_symbols(bug, top_k=top_k, include_imports=True),
    "symbols_no_imports": lambda bug, top_k: rank_files_bm25_with_symbols(bug, top_k=top_k, include_imports=False),
    "skeleton": lambda bug, top_k: rank_files_bm25_with_skeleton(bug, top_k=top_k),
    "refined": lambda bug, top_k: rank_files_bm25_refined(bug, top_k=top_k, include_imports=True),
}

WEIGHT_CONFIGS = [
    ("rrf_1_1", [1.0, 1.0]),
    ("rrf_1_2", [1.0, 2.0]),
    ("rrf_1_3", [1.0, 3.0]),
    ("rrf_1_5", [1.0, 5.0]),
    ("rrf_1_10", [1.0, 10.0]),
    ("rrf_1_15", [1.0, 15.0]),
    ("rrf_1_20", [1.0, 20.0]),
    ("rrf_1_30", [1.0, 30.0]),
    ("rrf_1_50", [1.0, 50.0]),
]


def _shard_slice(items, num_shards, shard_index):
    """Contiguous, near-equal-size slice -- e.g. 500 instances / 50 shards = 10 each,
    with any remainder distributed to the first shards one at a time."""
    n = len(items)
    base, extra = divmod(n, num_shards)
    start = shard_index * base + min(shard_index, extra)
    size = base + (1 if shard_index < extra else 0)
    return items[start:start + size]


def _compute_base_rankings(bugs, candidate_pool_size, model_name, bm25_repr="symbols_with_imports"):
    bm25_rank_fn = _BM25_REPR_FNS[bm25_repr]
    rankings = {}
    for i, bug in enumerate(bugs):
        t0 = time.time()
        bm25_candidates = bm25_rank_fn(bug, candidate_pool_size)
        t_bm25 = time.time() - t0

        if not bm25_candidates:
            rankings[bug.instance_id] = {"bm25": [], "chunked_embedding": []}
            continue

        candidate_bug = bug.model_copy(update={"code_files": bm25_candidates})
        t1 = time.time()
        embedding_ranking, timing = rank_files_embedding_chunked(candidate_bug, top_k=None, model_name=model_name)
        t_embed = time.time() - t1

        rankings[bug.instance_id] = {"bm25": bm25_candidates, "chunked_embedding": embedding_ranking}
        logger.info(
            f"[{i + 1}/{len(bugs)}] {bug.instance_id}: bm25={t_bm25:.2f}s "
            f"embed={t_embed:.2f}s ({timing.get('num_chunks', 0)} chunks over {timing.get('num_files', 0)} files)"
        )
    return rankings


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox', 'bench4bl', 'iqloc'], default=None)
    parser.add_argument('--iqloc-strict', action='store_true',
                       help="With --dataset iqloc: must match whatever the manifest was built "
                            "with (--iqloc-strict on generate_evaluation_manifest.py), or the "
                            "re-derived pool won't line up with the manifest's instance IDs.")
    parser.add_argument('--pool-size', type=int, default=None,
                       help='Override the manifest\'s stored pool_size when re-deriving the pool -- '
                            'needed if this environment\'s dataset mirror has a different total instance '
                            'count than the one the manifest was generated against (e.g. Bench4BL: more '
                            'projects mirrored locally than on MN5). Pass a value >= this environment\'s '
                            'total instance count to disable sampling and guarantee every manifest '
                            'instance is found.')
    parser.add_argument('--candidate-pool-size', type=int, default=200)
    parser.add_argument('--rrf-k-values', default='60',
                       help='Comma-separated RRF k constants to sweep, fused from the same '
                            'shared base rankings (see run_hybrid_rrf_weighting_test.py).')
    parser.add_argument('--bm25-repr', choices=list(_BM25_REPR_FNS), default='symbols_with_imports')
    parser.add_argument('--model', default='microsoft/unixcoder-base')
    parser.add_argument('--num-shards', type=int, required=True)
    parser.add_argument('--shard-index', type=int, required=True, help='0-based')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"shard-index {args.shard_index} out of range for num-shards {args.num_shards}")

    manifest = load_manifest(args.manifest)
    dataset_name = args.dataset or manifest['dataset']
    if dataset_name == 'swebench':
        instance = SWEBench()
    elif dataset_name == 'bench4bl':
        instance = Bench4BL()
    elif dataset_name == 'iqloc':
        instance = IQLocExtended(include_partial=not args.iqloc_strict)
    else:
        instance = BeetleBox()

    pool_size = args.pool_size or manifest.get('pool_size') or manifest['size']
    pool = instance.get_bug_instances(sample_size=pool_size, random_sample=True, random_seed=manifest['seed'])
    wanted = {inst['instance_id'] for inst in manifest['instances']}
    all_bugs = [b for b in pool if b.instance_id in wanted]
    missing = wanted - {b.instance_id for b in all_bugs}
    if missing:
        logger.warning(f"{len(missing)} manifest instance(s) not found when re-deriving the pool: {sorted(missing)[:5]}")

    # Sort by instance_id before sharding so the slice is deterministic regardless of the
    # pool's own (seeded-random) ordering -- every shard index always gets the same instances.
    all_bugs.sort(key=lambda b: b.instance_id)
    bugs = _shard_slice(all_bugs, args.num_shards, args.shard_index)
    logger.info(
        f"Shard {args.shard_index}/{args.num_shards}: {len(bugs)} instances "
        f"({bugs[0].instance_id if bugs else 'none'}..{bugs[-1].instance_id if bugs else 'none'}), "
        f"manifest {manifest['manifest_id']}, candidate_pool_size={args.candidate_pool_size}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"shard_{args.shard_index:04d}.json")

    if not bugs:
        _atomic_write_json(output_path, {
            "manifest_id": manifest["manifest_id"], "shard_index": args.shard_index,
            "num_shards": args.num_shards, "instance_ids": [], "configs": {},
        })
        logger.info(f"Shard {args.shard_index} empty, wrote placeholder to {output_path}")
        return

    k_values = [int(k) for k in args.rrf_k_values.split(',')]

    logger.info(f"--- Computing base rankings (bm25[{args.bm25_repr}] + chunked_embedding) ---")
    base_rankings = _compute_base_rankings(bugs, args.candidate_pool_size, args.model, bm25_repr=args.bm25_repr)

    fused_rankings = {}
    for bug in bugs:
        base = base_rankings[bug.instance_id]
        fused_rankings[bug.instance_id] = {
            f"{name}_k{k}": reciprocal_rank_fusion([base["bm25"], base["chunked_embedding"]], k=k, weights=weights)
            if base["bm25"] else []
            for name, weights in WEIGHT_CONFIGS
            for k in k_values
        }

    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    fused_config_names = [f"{name}_k{k}" for name, _ in WEIGHT_CONFIGS for k in k_values]
    config_names = ["bm25", "chunked_embedding"] + fused_config_names
    results = {}
    for name in config_names:
        if name == "bm25":
            rank_fn = lambda bug: base_rankings[bug.instance_id]["bm25"]
        elif name == "chunked_embedding":
            rank_fn = lambda bug: base_rankings[bug.instance_id]["chunked_embedding"]
        else:
            rank_fn = lambda bug, _name=name: fused_rankings[bug.instance_id][_name]
        report = screen_manifest(bugs, token=token, cache=cache, rank_fn=rank_fn)
        summary = summarize_screening(report)
        results[name] = {"screening_report": report, "summary": summary}

    save_cache(cache)

    logger.info(f"=== Shard {args.shard_index} summary (n={len(bugs)}) ===")
    for name in config_names:
        s = results[name]["summary"]
        logger.info(f"{name:<20} MRR={s['mrr']:.4f} MAP={s['map']:.4f} Hit@1={s['macro_hit_at'][1]:.3f}")

    _atomic_write_json(output_path, {
        "manifest_id": manifest["manifest_id"],
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "candidate_pool_size": args.candidate_pool_size,
        "weight_configs": dict(WEIGHT_CONFIGS),
        "instance_ids": [b.instance_id for b in bugs],
        "configs": results,
    })
    logger.info(f"Wrote shard {args.shard_index} report to {output_path}")


if __name__ == "__main__":
    main()
