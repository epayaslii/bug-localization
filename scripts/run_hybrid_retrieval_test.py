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
from method.bm25_retriever import rank_files_bm25_with_symbols
from method.embedding_retriever import rank_files_embedding_chunked
from method.hybrid_retriever import reciprocal_rank_fusion

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def _compute_rankings_once(bugs, candidate_pool_size, model_name, rrf_k):
    """Compute BM25, chunked-embedding, and hybrid-RRF rankings for every bug in a single
    pass, so the (expensive) chunked-embedding step runs exactly once per instance instead
    of once per config that needs it. Returns {instance_id: {"bm25": [...], "chunked_embedding": [...], "hybrid_rrf": [...]}}.
    """
    rankings = {}
    for i, bug in enumerate(bugs):
        t0 = time.time()
        bm25_candidates = rank_files_bm25_with_symbols(bug, top_k=candidate_pool_size)
        t_bm25 = time.time() - t0

        if not bm25_candidates:
            rankings[bug.instance_id] = {"bm25": [], "chunked_embedding": [], "hybrid_rrf": []}
            continue

        candidate_bug = bug.model_copy(update={"code_files": bm25_candidates})
        t1 = time.time()
        embedding_ranking, timing = rank_files_embedding_chunked(candidate_bug, top_k=None, model_name=model_name)
        t_embed = time.time() - t1

        hybrid_ranking = reciprocal_rank_fusion([bm25_candidates, embedding_ranking], k=rrf_k)

        rankings[bug.instance_id] = {
            "bm25": bm25_candidates,
            "chunked_embedding": embedding_ranking,
            "hybrid_rrf": hybrid_ranking,
        }
        logger.info(
            f"[{i + 1}/{len(bugs)}] {bug.instance_id}: bm25={t_bm25:.2f}s "
            f"embed={t_embed:.2f}s ({timing.get('num_chunks', 0)} chunks over {timing.get('num_files', 0)} files)"
        )
    return rankings


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Compare BM25 (symbols) alone vs chunked-embedding-reranked vs "
                    "BM25+chunked-embedding hybrid (RRF fusion) on the same manifest. "
                    "Computes the expensive chunked-embedding step exactly once per "
                    "instance and reuses it for both the chunked_embedding and hybrid_rrf "
                    "configs. Local model, no API cost."
    )
    parser.add_argument('--manifest', required=True, help='Path to a manifest JSON')
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox'], default=None)
    parser.add_argument('--candidate-pool-size', type=int, default=200,
                       help='BM25 top-k narrowed to before chunked-embedding rerank/fusion. '
                            'Hit@k/recall@k beyond this size are not meaningful for the '
                            'embedding-only and hybrid rankers, since they never see files outside the pool.')
    parser.add_argument('--rrf-k', type=int, default=60)
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
    logger.info(f"Comparing over {len(bugs)}/{manifest['size']} manifest instances (manifest {manifest['manifest_id']}), candidate_pool_size={args.candidate_pool_size}")

    logger.info("--- Computing rankings (one pass, shared across configs) ---")
    rankings_by_instance = _compute_rankings_once(bugs, args.candidate_pool_size, args.model, args.rrf_k)

    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    config_names = ["bm25", "chunked_embedding", "hybrid_rrf"]
    results = {}
    for name in config_names:
        rank_fn = lambda bug, _name=name: rankings_by_instance[bug.instance_id][_name]
        report = screen_manifest(bugs, token=token, cache=cache, rank_fn=rank_fn)
        summary = summarize_screening(report)
        results[name] = {"screening_report": report, "summary": summary}

    save_cache(cache)

    logger.info(f"=== Summary (macro, candidate_pool_size={args.candidate_pool_size}) ===")
    logger.info(f"{'config':<20} {'Hit@1':>7} {'Hit@5':>7} {'Hit@10':>7} {'Hit@100':>8} {'MRR':>8} {'MAP':>8}")
    for name in config_names:
        s = results[name]["summary"]
        logger.info(
            f"{name:<20} {s['macro_hit_at'][1]:>7.3f} {s['macro_hit_at'][5]:>7.3f} "
            f"{s['macro_hit_at'][10]:>7.3f} {s['macro_hit_at'][100]:>8.3f} {s['mrr']:>8.4f} {s['map']:>8.4f}"
        )

    hybrid_mrr = results["hybrid_rrf"]["summary"]["mrr"]
    bm25_mrr = results["bm25"]["summary"]["mrr"]
    delta = hybrid_mrr - bm25_mrr
    logger.info(f"Delta MRR (hybrid - bm25): {delta:+.4f}")
    if delta <= 0:
        logger.info("VERDICT: hybrid RRF fusion did NOT beat BM25-symbols alone on this manifest.")
    else:
        logger.info("VERDICT: hybrid RRF fusion beat BM25-symbols alone -- worth a larger manifest run.")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"manifest_id": manifest["manifest_id"], "candidate_pool_size": args.candidate_pool_size, "configs": results}, f, indent=2)
        logger.info(f"Wrote full report to {args.output}")


if __name__ == "__main__":
    main()
