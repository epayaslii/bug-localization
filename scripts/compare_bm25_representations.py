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
from dataset.localizability import load_cache, save_cache
from dataset.utils import setup_logging, get_logger
from evaluation.manifest import load_manifest
from evaluation.screening import screen_manifest, summarize_screening
from method.bm25_retriever import rank_files_bm25, rank_files_bm25_with_skeleton, rank_files_bm25_with_symbols

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

REPRESENTATIONS = {
    "path_only": lambda b: rank_files_bm25(b.bug_report, b.code_files, top_k=None),
    "skeleton": lambda b: rank_files_bm25_with_skeleton(b, top_k=None),
    "symbols_with_imports": lambda b: rank_files_bm25_with_symbols(b, top_k=None, include_imports=True),
    "symbols_no_imports": lambda b: rank_files_bm25_with_symbols(b, top_k=None, include_imports=False),
}


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Compare BM25 document representations (path-only, skeleton, "
                    "symbols+imports, symbols-no-imports) on the same evaluation manifest -- "
                    "free and offline, no LLM calls."
    )
    parser.add_argument('--manifest', required=True, help='Path to a manifest JSON')
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox', 'bench4bl'], default=None,
                       help='Overrides the dataset recorded in the manifest, if needed')
    parser.add_argument('--representations', nargs='+', choices=list(REPRESENTATIONS), default=None,
                       help=f'Subset to compare (default: all of {list(REPRESENTATIONS)})')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    dataset_name = args.dataset or manifest['dataset']
    instance = {'swebench': SWEBench, 'beetlebox': BeetleBox, 'bench4bl': Bench4BL}[dataset_name]()

    pool_size = manifest.get('pool_size') or manifest['size']
    pool = instance.get_bug_instances(sample_size=pool_size, random_sample=True, random_seed=manifest['seed'])
    wanted = {inst['instance_id'] for inst in manifest['instances']}
    bugs = [b for b in pool if b.instance_id in wanted]
    missing = wanted - {b.instance_id for b in bugs}
    if missing:
        logger.warning(f"{len(missing)} manifest instance(s) not found when re-deriving the pool: {sorted(missing)[:5]}")
    logger.info(f"Comparing representations over {len(bugs)}/{manifest['size']} manifest instances (manifest {manifest['manifest_id']})")

    names = args.representations or list(REPRESENTATIONS)
    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    results = {}
    for name in names:
        logger.info(f"--- Representation: {name} ---")
        report = screen_manifest(bugs, token=token, cache=cache, rank_fn=REPRESENTATIONS[name])
        summary = summarize_screening(report)
        results[name] = {"screening_report": report, "summary": summary}
        logger.info(
            f"  Hit@1={summary['macro_hit_at'][1]:.3f} Hit@5={summary['macro_hit_at'][5]:.3f} "
            f"Hit@10={summary['macro_hit_at'][10]:.3f} MRR={summary['mrr']:.4f} MAP={summary['map']:.4f}"
        )
        logger.info(f"  Difficulty: {report['difficulty_distribution']}")

    save_cache(cache)

    logger.info("=== Summary (macro, across all representations) ===")
    logger.info(f"{'representation':<22} {'Hit@1':>7} {'Hit@5':>7} {'Hit@10':>7} {'MRR':>8} {'MAP':>8}")
    for name in names:
        s = results[name]["summary"]
        logger.info(
            f"{name:<22} {s['macro_hit_at'][1]:>7.3f} {s['macro_hit_at'][5]:>7.3f} "
            f"{s['macro_hit_at'][10]:>7.3f} {s['mrr']:>8.4f} {s['map']:>8.4f}"
        )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"manifest_id": manifest["manifest_id"], "representations": results}, f, indent=2)
        logger.info(f"Wrote full report to {args.output}")


if __name__ == "__main__":
    main()
