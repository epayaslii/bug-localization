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
from dataset.utils import setup_logging, get_logger, get_token_count
from evaluation.manifest import load_manifest
from evaluation.screening import screen_manifest
from evaluation.failure_attribution import summarize_retrieval_reach, prepare_oracle_candidate_set, run_oracle_diagnostic
from method.prompt import PromptGenerator

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def _load_manifest_bugs(manifest, dataset_name):
    instance = SWEBench() if dataset_name == 'swebench' else BeetleBox()
    pool_size = manifest.get('pool_size') or manifest['size']
    pool = instance.get_bug_instances(sample_size=pool_size, random_sample=True, random_seed=manifest['seed'])
    wanted = {inst['instance_id'] for inst in manifest['instances']}
    bugs = [b for b in pool if b.instance_id in wanted]
    missing = wanted - {b.instance_id for b in bugs}
    if missing:
        logger.warning(f"{len(missing)} manifest instance(s) not found when re-deriving the pool: {sorted(missing)[:5]}")
    return instance, bugs


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Split misses into retrieval failures (GT never reached the candidate "
                    "set) vs reranking failures (GT was reachable), and optionally run the "
                    "oracle diagnostic (force-inject GT, measure pure reranking ability)."
    )
    parser.add_argument('--manifest', required=True, help='Path to a manifest JSON')
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox'], default=None,
                       help='Overrides the dataset recorded in the manifest, if needed')
    parser.add_argument('--candidate-size', type=int, default=100,
                       help='BM25 top-k treated as "the candidate set the LLM sees" for retrieval-reach classification')
    parser.add_argument('--run-oracle', action='store_true',
                       help='Actually call the LLM reranker on oracle candidate sets (COSTS API CALLS). '
                            'Without this flag, only the free offline retrieval-reach summary runs.')
    parser.add_argument('--model', default='gpt-oss-20b', help='OpenRouter model for --run-oracle (default stays free-tier)')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    dataset_name = args.dataset or manifest['dataset']
    instance, bugs = _load_manifest_bugs(manifest, dataset_name)
    logger.info(f"Loaded {len(bugs)}/{manifest['size']} manifest instances (manifest {manifest['manifest_id']})")

    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    logger.info("Running BM25 screening (free, offline)...")
    screening_report = screen_manifest(bugs, token=token, cache=cache)

    reach_summary = summarize_retrieval_reach(screening_report, candidate_size=args.candidate_size)
    logger.info(f"=== Retrieval-vs-reranking split (candidate_size={args.candidate_size}) ===")
    logger.info(f"  Localizable-GT files reaching the candidate set: {reach_summary['file_level_counts']['reached_candidate_set']}")
    logger.info(f"  Localizable-GT files that are retrieval failures: {reach_summary['file_level_counts']['retrieval_failure']}")
    logger.info(f"  Instances with >=1 localizable GT: {reach_summary['instances_with_localizable_gt']}")
    logger.info(f"  Instances where EVERY localizable GT is a retrieval failure (reranking can't help): {reach_summary['instances_fully_unreachable']}")

    oracle_results = None
    if args.run_oracle:
        # Cost dry-run info before spending anything.
        prompt_gen = PromptGenerator()
        estimated_tokens = 0
        eligible = 0
        for bug in bugs:
            candidate_set, _ = prepare_oracle_candidate_set(bug, args.candidate_size, cache=cache, token=token)
            if not candidate_set:
                continue
            eligible += 1
            prompt = prompt_gen.generate_openai_prompt(bug, candidate_set)
            estimated_tokens += get_token_count(prompt, model="gpt-4o")
        logger.info(f"Oracle diagnostic: {eligible} instance(s) eligible, ~{estimated_tokens:,} estimated prompt tokens, model={args.model}")

        from method.openrouter_localizer import OpenRouterLocalizer
        localizer = OpenRouterLocalizer(model=args.model)

        oracle_results = []
        for i, bug in enumerate(bugs):
            result = run_oracle_diagnostic(bug, localizer, candidate_size=args.candidate_size, token=token, cache=cache)
            if result is not None:
                oracle_results.append(result)
            logger.info(f"Oracle {i + 1}/{len(bugs)} done")

        if oracle_results:
            top1_rate = sum(r["top1_hit"] for r in oracle_results) / len(oracle_results)
            top10_rate = sum(r["top10_hit_fraction"] for r in oracle_results) / len(oracle_results)
            logger.info(f"=== Oracle diagnostic (retrieval recall forced to 100%) ===")
            logger.info(f"  Top-1 hit rate: {top1_rate:.3f}")
            logger.info(f"  Mean Top-10 GT coverage: {top10_rate:.3f}")
            logger.info(f"  API stats: {localizer.total_stats()}")

    save_cache(cache)

    if args.output:
        report = {
            "manifest_id": manifest["manifest_id"],
            "candidate_size": args.candidate_size,
            "retrieval_reach_summary": reach_summary,
            "oracle_results": oracle_results,
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Wrote full report to {args.output}")


if __name__ == "__main__":
    main()
