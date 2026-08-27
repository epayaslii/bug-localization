"""Array-job shard of compare_bm25_representations.py, for running the full Bench4BL
population (4,418 instances, no diversity sampling) on MN5 as a Slurm array job instead of
one multi-hour serial job -- same pattern as run_hybrid_rrf_weighting_shard.py.

Each shard processes only a slice of the manifest's instances (selected by --num-shards /
--shard-index) and writes its own JSON to --output-dir, atomically (write to a .tmp path,
then os.replace) so a task killed mid-write never leaves a corrupt/partial file behind.
aggregate_bm25_shards.py merges all shard JSONs afterward.
"""

import os
import sys
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
from method.bm25_retriever import rank_files_bm25, rank_files_bm25_with_skeleton, rank_files_bm25_with_symbols, rank_files_bm25_refined

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

REPRESENTATIONS = {
    "path_only": lambda b: rank_files_bm25(b.bug_report, b.code_files, top_k=None),
    "skeleton": lambda b: rank_files_bm25_with_skeleton(b, top_k=None),
    "symbols_with_imports": lambda b: rank_files_bm25_with_symbols(b, top_k=None, include_imports=True),
    "symbols_no_imports": lambda b: rank_files_bm25_with_symbols(b, top_k=None, include_imports=False),
    "refined": lambda b: rank_files_bm25_refined(b, top_k=None, include_imports=True),
}


def _shard_slice(items, num_shards, shard_index):
    n = len(items)
    base, extra = divmod(n, num_shards)
    start = shard_index * base + min(shard_index, extra)
    size = base + (1 if shard_index < extra else 0)
    return items[start:start + size]


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
    parser.add_argument('--pool-size', type=int, default=None)
    parser.add_argument('--representations', nargs='+', choices=list(REPRESENTATIONS), default=None)
    parser.add_argument('--num-shards', type=int, required=True)
    parser.add_argument('--shard-index', type=int, required=True, help='0-based')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"shard-index {args.shard_index} out of range for num-shards {args.num_shards}")

    manifest = load_manifest(args.manifest)
    dataset_name = args.dataset or manifest['dataset']
    instance = {'swebench': SWEBench, 'beetlebox': BeetleBox, 'bench4bl': Bench4BL, 'iqloc': IQLocExtended}[dataset_name]()

    pool_size = args.pool_size or manifest.get('pool_size') or manifest['size']
    pool = instance.get_bug_instances(sample_size=pool_size, random_sample=True, random_seed=manifest['seed'])
    wanted = {inst['instance_id'] for inst in manifest['instances']}
    all_bugs = [b for b in pool if b.instance_id in wanted]
    missing = wanted - {b.instance_id for b in all_bugs}
    if missing:
        logger.warning(f"{len(missing)} manifest instance(s) not found when re-deriving the pool: {sorted(missing)[:5]}")

    all_bugs.sort(key=lambda b: b.instance_id)
    bugs = _shard_slice(all_bugs, args.num_shards, args.shard_index)
    logger.info(
        f"Shard {args.shard_index}/{args.num_shards}: {len(bugs)} instances "
        f"({bugs[0].instance_id if bugs else 'none'}..{bugs[-1].instance_id if bugs else 'none'}), "
        f"manifest {manifest['manifest_id']}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"shard_{args.shard_index:04d}.json")

    if not bugs:
        _atomic_write_json(output_path, {
            "manifest_id": manifest["manifest_id"], "shard_index": args.shard_index,
            "num_shards": args.num_shards, "instance_ids": [], "representations": {},
        })
        logger.info(f"Shard {args.shard_index} empty, wrote placeholder to {output_path}")
        return

    names = args.representations or list(REPRESENTATIONS)
    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    results = {}
    for name in names:
        logger.info(f"--- Shard {args.shard_index}, representation: {name} ---")
        report = screen_manifest(bugs, token=token, cache=cache, rank_fn=REPRESENTATIONS[name])
        summary = summarize_screening(report)
        results[name] = {"screening_report": report, "summary": summary}
        logger.info(f"  Hit@1={summary['macro_hit_at'][1]:.3f} MRR={summary['mrr']:.4f} MAP={summary['map']:.4f}")

    save_cache(cache)

    _atomic_write_json(output_path, {
        "manifest_id": manifest["manifest_id"],
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "instance_ids": [b.instance_id for b in bugs],
        "representations": results,
    })
    logger.info(f"Wrote shard {args.shard_index} report to {output_path}")


if __name__ == "__main__":
    main()
