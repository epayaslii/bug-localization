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
from dataset.localizability import load_cache, save_cache
from dataset.utils import setup_logging, get_logger
from evaluation.manifest import load_manifest
from evaluation.screening import screen_manifest, summarize_screening
from method.bm25_retriever import rank_files_bm25
from method.embedding_retriever import rank_files_embedding

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def _embedding_rank_fn(model_name):
    def rank_fn(bug):
        ranked, timing = rank_files_embedding(bug, top_k=None, model_name=model_name)
        if timing:
            logger.info(
                f"  {bug.instance_id}: {timing['num_files']} files, "
                f"fetch={timing['fetch_s']:.2f}s embed={timing['embed_s']:.2f}s"
            )
        return ranked
    return rank_fn


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Measure the retrieval recall CEILING of UniXCoder embeddings -- how "
                    "often the correct file lands in the top-K by cosine similarity alone, "
                    "before any LLM reranking -- and compare directly against BM25 path-only "
                    "on the same manifest. No API calls, no cost; runs the embedding model locally."
    )
    parser.add_argument('--manifest', required=True, help='Path to a manifest JSON')
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox'], default=None,
                       help='Overrides the dataset recorded in the manifest, if needed')
    parser.add_argument('--model', default='microsoft/unixcoder-base')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    dataset_name = args.dataset or manifest['dataset']
    instance = SWEBench() if dataset_name == 'swebench' else BeetleBox()

    pool_size = manifest.get('pool_size') or manifest['size']
    pool = instance.get_bug_instances(sample_size=pool_size, random_sample=True, random_seed=manifest['seed'])
    wanted = {inst['instance_id'] for inst in manifest['instances']}
    bugs = [b for b in pool if b.instance_id in wanted]
    missing = wanted - {b.instance_id for b in bugs}
    if missing:
        logger.warning(f"{len(missing)} manifest instance(s) not found when re-deriving the pool: {sorted(missing)[:5]}")
    logger.info(f"Ceiling test over {len(bugs)}/{manifest['size']} manifest instances (manifest {manifest['manifest_id']})")

    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    logger.info(f"--- Embedding ({args.model}) ---")
    t0 = time.time()
    embedding_report = screen_manifest(bugs, token=token, cache=cache, rank_fn=_embedding_rank_fn(args.model))
    embedding_elapsed = time.time() - t0
    embedding_summary = summarize_screening(embedding_report)
    logger.info(f"  Elapsed: {embedding_elapsed:.1f}s total")

    logger.info("--- BM25 (path-only baseline) ---")
    bm25_report = screen_manifest(
        bugs, token=token, cache=cache,
        rank_fn=lambda b: rank_files_bm25(b.bug_report, b.code_files, top_k=None),
    )
    bm25_summary = summarize_screening(bm25_report)

    save_cache(cache)

    logger.info("=== Ceiling comparison (macro, across manifest) ===")
    logger.info(f"{'method':<20} {'Hit@1':>7} {'Hit@5':>7} {'Hit@10':>7} {'Hit@100':>8} {'MRR':>8}")
    for name, s in [("bm25_path_only", bm25_summary), (f"embedding_{args.model}", embedding_summary)]:
        logger.info(
            f"{name:<20} {s['macro_hit_at'][1]:>7.3f} {s['macro_hit_at'][5]:>7.3f} "
            f"{s['macro_hit_at'][10]:>7.3f} {s['macro_hit_at'][100]:>8.3f} {s['mrr']:>8.4f}"
        )

    mrr_delta = embedding_summary["mrr"] - bm25_summary["mrr"]
    hit10_delta = embedding_summary["macro_hit_at"][10] - bm25_summary["macro_hit_at"][10]
    logger.info(f"Delta (embedding - BM25): MRR={mrr_delta:+.4f}, Hit@10={hit10_delta:+.3f}")
    if mrr_delta <= 0 and hit10_delta <= 0:
        logger.info("VERDICT: embeddings do NOT beat BM25 on this manifest -- consistent with "
                     "both the original team's and the co-intern's prior findings on RAG/embeddings here.")
    else:
        logger.info("VERDICT: embeddings beat BM25 on this manifest -- worth a larger manifest run before deciding further.")

    if args.output:
        report = {
            "manifest_id": manifest["manifest_id"],
            "embedding_model": args.model,
            "embedding_elapsed_s": embedding_elapsed,
            "bm25_path_only": {"screening_report": bm25_report, "summary": bm25_summary},
            "embedding": {"screening_report": embedding_report, "summary": embedding_summary},
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Wrote full report to {args.output}")


if __name__ == "__main__":
    main()
