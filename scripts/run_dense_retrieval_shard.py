"""Array-job shard for a TRUE dense-retrieval-only test: chunked embedding ranking over the
FULL repository corpus (bug.code_files, no BM25 pre-filter at all), not this project's usual
BM25-first-then-embed-rerank pattern. Matches what "dense retrieval" means in the co-intern's
own deck (Jina candidate, "buggy snapshot files .java only", no lexical stage) -- run here
with Qwen3-Embedding-0.6B (this project's own confirmed-best embedding model; no Jina model is
staged in this project's MN5 environment).

Real prior context: this project already ran a whole-file (not chunked) dense-only test with
UniXCoder and found it negative (`experiment/embedding-ceiling`, kept off main as a documented
dead end). This is a different, stronger version -- chunked (not whole-file) and Qwen3 (not
UniXCoder, which the project's own bake-off found much weaker) -- worth testing for real
before assuming the same negative result holds.

Full-corpus embedding is real GPU compute per instance (every file in the repo, not a ~100-
file BM25-narrowed pool) -- scoped as a small 8-shard bounded probe on the n=30 diverse
manifest first, matching the co-intern's own probe scale, not a full run.
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
from method.embedding_retriever import rank_files_embedding_chunked

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
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox', 'bench4bl', 'iqloc'], default=None)
    parser.add_argument('--pool-size', type=int, default=None)
    parser.add_argument('--model', default='Qwen/Qwen3-Embedding-0.6B')
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
        f"manifest {manifest['manifest_id']}, full-corpus dense retrieval, model={args.model}"
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

    per_bug_ranking = {}
    for i, bug in enumerate(bugs):
        t0 = time.time()
        ranked, timing = rank_files_embedding_chunked(bug, top_k=None, model_name=args.model)
        per_bug_ranking[bug.instance_id] = ranked
        logger.info(
            f"[{i + 1}/{len(bugs)}] {bug.instance_id}: {len(bug.code_files)} files in full corpus, "
            f"{timing.get('num_chunks', '?')} chunks, {time.time() - t0:.2f}s"
        )

    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    rank_fn = lambda bug: per_bug_ranking[bug.instance_id]
    report = screen_manifest(bugs, token=token, cache=cache, rank_fn=rank_fn)
    summary = summarize_screening(report)
    logger.info(f"=== Shard {args.shard_index} summary (n={len(bugs)}) === MRR={summary['mrr']:.4f} MAP={summary['map']:.4f} Hit@1={summary['macro_hit_at'][1]:.3f}")

    save_cache(cache)

    _atomic_write_json(output_path, {
        "manifest_id": manifest["manifest_id"],
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "model": args.model,
        "mode": "full_corpus_dense_only",
        "instance_ids": [b.instance_id for b in bugs],
        "configs": {"dense_embedding": {"screening_report": report, "summary": summary}},
    })
    logger.info(f"Wrote shard {args.shard_index} report to {output_path}")


if __name__ == "__main__":
    main()
