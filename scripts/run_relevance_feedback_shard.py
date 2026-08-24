"""Array-job shard of run_relevance_feedback_test.py, for running the recall-oriented
relevance-feedback pipeline (hybrid-RRF retriever -> embedding-cosine relevance filtering,
no LLM call -> semantic EmbedRank/MMR reformulation) at real scale on MN5 GPU.

Unlike the IQLoc-approximation shard, this needs no Ollama server lifecycle at all --
--relevance-mode embedding-cosine makes zero LLM calls, so localizer is always None. Uses a
fully-local embedding model (default Qwen3-Embedding-0.6B, not OpenAI) for both the retriever
and the relevance/reformulation stages, since MN5 has no outbound internet and an
OpenAI-dependent pipeline cannot execute there at all -- this is a real config swap from the
confirmed-best local (OpenAI+skeleton) result, not the identical config re-run on GPU.
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
from method.prompt import PromptGenerator
from scripts.run_relevance_feedback_test import _run_one

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
    parser.add_argument('--candidate-pool-size', type=int, default=100)
    parser.add_argument('--retriever', choices=['bm25', 'hybrid-rrf'], default='hybrid-rrf')
    parser.add_argument('--embedding-model', default='Qwen/Qwen3-Embedding-0.6B')
    parser.add_argument('--rrf-weights', default='1,5')
    parser.add_argument('--bm25-repr', choices=['symbols_with_imports', 'symbols_no_imports', 'skeleton'], default='skeleton')
    parser.add_argument('--granularity', choices=['file', 'chunk'], default='chunk')
    parser.add_argument('--max-chunks-per-file', type=int, default=5)
    parser.add_argument('--reformulation-mode', choices=['literal', 'semantic'], default='semantic')
    parser.add_argument('--keyword-model', default='Qwen/Qwen3-Embedding-0.6B')
    parser.add_argument('--top-n-keywords', type=int, default=15)
    parser.add_argument('--relevance-mode', choices=['llm', 'embedding-cosine'], default='embedding-cosine')
    parser.add_argument('--relevance-embedding-model', default='Qwen/Qwen3-Embedding-0.6B')
    parser.add_argument('--keep-fraction', type=float, default=0.5)
    parser.add_argument('--num-shards', type=int, required=True)
    parser.add_argument('--shard-index', type=int, required=True, help='0-based')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"shard-index {args.shard_index} out of range for num-shards {args.num_shards}")

    manifest = load_manifest(args.manifest)
    dataset_name = args.dataset or manifest['dataset']
    instance = {'swebench': SWEBench, 'beetlebox': BeetleBox, 'bench4bl': Bench4BL, 'iqloc': IQLocExtended}[dataset_name]()

    rrf_weights = [float(w) for w in args.rrf_weights.split(',')] if args.retriever == 'hybrid-rrf' else None

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
            "num_shards": args.num_shards, "instance_ids": [], "configs": {},
        })
        logger.info(f"Shard {args.shard_index} empty, wrote placeholder to {output_path}")
        return

    prompt_gen = PromptGenerator()

    per_bug = {}
    for i, bug in enumerate(bugs):
        t0 = time.time()
        result = _run_one(
            None, prompt_gen, bug, args.candidate_pool_size, args.retriever,
            args.embedding_model, rrf_weights, args.granularity, args.max_chunks_per_file,
            args.bm25_repr, args.reformulation_mode, args.keyword_model, args.top_n_keywords,
            args.relevance_mode, args.relevance_embedding_model, args.keep_fraction,
        )
        per_bug[bug.instance_id] = result
        logger.info(
            f"[{i + 1}/{len(bugs)}] {bug.instance_id}: {result['relevant_count']}/{len(result['retriever'])} "
            f"judged relevant, {len(result['reformulation_terms'])} reformulation terms, {time.time() - t0:.2f}s"
        )

    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    config_names = ["retriever", "relevance_filtered", "reformulated"]
    results = {}
    for name in config_names:
        rank_fn = lambda bug, _name=name: per_bug[bug.instance_id][_name]
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
        "embedding_model": args.embedding_model,
        "reformulation_mode": args.reformulation_mode,
        "relevance_mode": args.relevance_mode,
        "keep_fraction": args.keep_fraction,
        "instance_ids": [b.instance_id for b in bugs],
        "configs": results,
    })
    logger.info(f"Wrote shard {args.shard_index} report to {output_path}")


if __name__ == "__main__":
    main()
