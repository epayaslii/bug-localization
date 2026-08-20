"""Array-job shard of run_iqloc_approximation_test.py, for running the IQLoc-approximation
pipeline (BM25 skeleton -> Ollama chunk relevance judge -> EmbedRank/MMR keyword extraction
-> cosine reformulation -> BM25 rerank) at real scale on MN5, GPU-accelerated -- same
shard/aggregate pattern as run_bm25_comparison_shard.py and run_hybrid_rrf_weighting_shard.py.

Unlike those, this pipeline needs a *running local Ollama server* on the same node -- the
sbatch script that launches this shard is responsible for starting `ollama serve` (with
OLLAMA_MODELS pointed at the transferred model dir) before invoking this script, and killing
it after. Imports _run_one from run_iqloc_approximation_test.py directly rather than
duplicating the pipeline logic, since keeping the approximation mechanism single-sourced
matters more here than shard-script isolation from CLI changes.
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
from dataset.localizability import load_cache, save_cache
from dataset.utils import setup_logging, get_logger
from evaluation.manifest import load_manifest
from evaluation.screening import screen_manifest, summarize_screening
from method.ollama_localizer import OllamaLocalizer
from method.prompt import PromptGenerator
from scripts.run_iqloc_approximation_test import _run_one

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
    parser.add_argument('--candidate-pool-size', type=int, default=50)
    parser.add_argument('--max-chunks-per-file', type=int, default=5)
    parser.add_argument('--top-n-keywords', type=int, default=15)
    parser.add_argument('--keyword-model', default='microsoft/unixcoder-base')
    parser.add_argument('--model', default='qwen2.5-coder-7b')
    parser.add_argument('--ollama-host', default=None)
    parser.add_argument('--num-ctx', type=int, default=16384)
    parser.add_argument('--max-tokens', type=int, default=8192,
                       help='Raised from OllamaLocalizer\'s own 4096 default -- fixes a real JSON-'
                            'truncation failure mode confirmed on the n=200 run (3/200 instances, 1.5%%).')
    parser.add_argument('--num-shards', type=int, required=True)
    parser.add_argument('--shard-index', type=int, required=True, help='0-based')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"shard-index {args.shard_index} out of range for num-shards {args.num_shards}")

    manifest = load_manifest(args.manifest)
    dataset_name = args.dataset or manifest['dataset']
    instance = {'swebench': SWEBench, 'beetlebox': BeetleBox, 'bench4bl': Bench4BL}[dataset_name]()

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

    localizer = OllamaLocalizer(model=args.model, host=args.ollama_host, num_ctx=args.num_ctx, max_tokens=args.max_tokens)
    prompt_gen = PromptGenerator()

    per_bug = {}
    for i, bug in enumerate(bugs):
        t0 = time.time()
        result = _run_one(
            localizer, prompt_gen, bug, args.candidate_pool_size, args.max_chunks_per_file,
            args.keyword_model, args.top_n_keywords,
        )
        per_bug[bug.instance_id] = result
        logger.info(
            f"[{i + 1}/{len(bugs)}] {bug.instance_id}: {result['relevant_count']}/{len(result['retriever'])} "
            f"judged relevant, {len(result['reformulation_terms'])} reformulation terms, {time.time() - t0:.2f}s"
        )

    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    config_names = ["retriever", "relevance_filtered", "iqloc_reformulated"]
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

    stats = localizer.total_stats()
    logger.info(f"LLM call stats: {stats}")

    _atomic_write_json(output_path, {
        "manifest_id": manifest["manifest_id"],
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "candidate_pool_size": args.candidate_pool_size,
        "keyword_model": args.keyword_model,
        "top_n_keywords": args.top_n_keywords,
        "llm_call_stats": stats,
        "instance_ids": [b.instance_id for b in bugs],
        "configs": results,
    })
    logger.info(f"Wrote shard {args.shard_index} report to {output_path}")


if __name__ == "__main__":
    main()
