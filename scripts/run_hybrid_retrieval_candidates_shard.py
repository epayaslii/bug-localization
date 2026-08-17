"""Array-job shard: computes hybrid-RRF retrieval candidates (BM25 -> chunked embedding ->
weighted RRF, via method/hybrid_retriever.rank_files_hybrid) for a manifest's instances --
no LLM call. Phase 1 of a two-phase end-to-end eval, split out because the final LLM call
needs live internet (can't run on MN5) while real per-instance Java-aware chunking is too
slow for one serial job anywhere (~900s/instance observed on Bench4BL). Writes
{instance_id: [ranked candidate file paths]} per shard, atomically, AFTER EVERY INSTANCE
(not just once at the end -- this project has lost partial progress to that exact pattern
before). A separate local script (main.py --candidates-file) loads the aggregated result and
does the fast LLM calls where internet is available.
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
from dataset.utils import setup_logging, get_logger
from evaluation.manifest import load_manifest
from method.hybrid_retriever import rank_files_hybrid

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


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
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox', 'bench4bl'], default=None)
    parser.add_argument('--pool-size', type=int, default=None)
    parser.add_argument('--candidate-pool-size', type=int, default=200)
    parser.add_argument('--retrieval-top-k', type=int, default=100)
    parser.add_argument('--rrf-weights', default='1,5')
    parser.add_argument('--embedding-model', default='Qwen/Qwen3-Embedding-0.6B')
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
    else:
        instance = BeetleBox()

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
        f"manifest {manifest['manifest_id']}, retrieval_top_k={args.retrieval_top_k}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"shard_{args.shard_index:04d}.json")

    weights = [float(w) for w in args.rrf_weights.split(',')]
    candidates = {}
    _atomic_write_json(output_path, candidates)  # placeholder so a killed-immediately task still leaves a file

    for i, bug in enumerate(bugs):
        t0 = time.time()
        ranked, timing = rank_files_hybrid(
            bug, top_k=args.retrieval_top_k, candidate_pool_size=args.candidate_pool_size,
            embedding_model=args.embedding_model, weights=weights,
        )
        candidates[bug.instance_id] = ranked
        logger.info(
            f"[{i + 1}/{len(bugs)}] {bug.instance_id}: {len(ranked)} candidates in {time.time() - t0:.1f}s "
            f"(bm25={timing.get('bm25_s', 0):.1f}s, embed={timing.get('embed_s', 0):.1f}s)"
        )
        _atomic_write_json(output_path, candidates)  # after every instance, not just at the end

    logger.info(f"Wrote {len(candidates)} candidate lists to {output_path}")


if __name__ == "__main__":
    main()
