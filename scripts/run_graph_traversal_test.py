"""Compares plain BM25 (symbols) against BM25-seeded graph traversal (method/graph_retriever.py)
on the same manifest -- retrieval-only, free/offline (no LLM calls, no embeddings), so this
is cheap to run at any n. Reuses the same manifest as prior BM25/hybrid-retrieval work for
direct comparability against those published numbers.
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
from dataset.localizability import load_cache, save_cache
from dataset.utils import setup_logging, get_logger
from evaluation.manifest import load_manifest
from evaluation.screening import screen_manifest, summarize_screening
from method.bm25_retriever import rank_files_bm25_with_symbols
from method.graph_retriever import rank_files_graph_traversal

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Compare BM25 alone vs. BM25-seeded graph traversal")
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox'], default=None)
    parser.add_argument('--seed-size', type=int, default=10,
                       help='How many of BM25\'s top ranks seed the graph traversal')
    parser.add_argument('--hops', type=int, default=2)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    dataset_name = args.dataset or manifest['dataset']
    instance = SWEBench() if dataset_name == 'swebench' else BeetleBox()

    pool_size = manifest.get('pool_size') or manifest['size']
    pool = instance.get_bug_instances(sample_size=pool_size, random_sample=True, random_seed=manifest['seed'])

    wanted = {inst['instance_id']: inst for inst in manifest['instances']}
    bugs = [b for b in pool if b.instance_id in wanted]
    missing = set(wanted) - {b.instance_id for b in bugs}
    if missing:
        logger.warning(f"{len(missing)} manifest instance(s) not found when re-deriving the pool")

    logger.info(f"Evaluating {len(bugs)}/{manifest['size']} manifest instances (manifest {manifest['manifest_id']})")

    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    configs = {
        "bm25_symbols": lambda bug: rank_files_bm25_with_symbols(bug, top_k=None),
        "graph_traversal": lambda bug: rank_files_graph_traversal(
            bug,
            seed_ranking=rank_files_bm25_with_symbols(bug, top_k=None),
            seed_size=args.seed_size,
            hops=args.hops,
            top_k=None,
        ),
    }

    results = {}
    for name, rank_fn in configs.items():
        logger.info(f"--- screening: {name} ---")
        report = screen_manifest(bugs, token=token, cache=cache, rank_fn=rank_fn)
        summary = summarize_screening(report)
        results[name] = {"screening_report": report, "summary": summary}
        logger.info(f"{name}: {summary}")

    save_cache(cache)

    print("\n=== Summary ===")
    for name, data in results.items():
        s = data["summary"]
        print(f"{name}: MRR={s['mrr']:.3f} MAP={s['map']:.3f} "
              f"Hit@1={s['macro_hit_at'][1]:.3f} Hit@10={s['macro_hit_at'][10]:.3f} "
              f"Hit@100={s['macro_hit_at'][100]:.3f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "manifest_id": manifest["manifest_id"],
                "seed_size": args.seed_size,
                "hops": args.hops,
                "configs": results,
            }, f, indent=2)
        logger.info(f"Wrote full report to {args.output}")


if __name__ == "__main__":
    main()
