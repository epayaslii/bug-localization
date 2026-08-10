"""Follow-up to run_hybrid_retrieval_test.py: at n=30, unweighted RRF fusion (bm25 weight=1,
embedding weight=1) underperformed chunked_embedding alone on Hit@1/MRR (see
results/README.md §4) -- several instances where embedding alone landed rank 1 got dragged
down by fusion with BM25's much weaker ranking. This sweeps embedding-favored RRF weights
to check whether up-weighting the embedding ranking recovers (or beats) embedding-alone
performance, rather than discarding hybrid fusion outright.

Computes BM25 + chunked-embedding rankings exactly once per instance (the expensive step)
and reuses them across every weight variant, so the sweep costs the same as a single run.
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
from dataset.localizability import load_cache, save_cache
from dataset.utils import setup_logging, get_logger
from evaluation.manifest import load_manifest
from evaluation.screening import screen_manifest, summarize_screening
from method.bm25_retriever import rank_files_bm25_with_symbols
from method.embedding_retriever import rank_files_embedding_chunked
from method.hybrid_retriever import reciprocal_rank_fusion

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

# (config_name, [bm25_weight, embedding_weight]) -- 1:1 reproduces the original unweighted
# hybrid_rrf result as a sanity check; higher embedding weight tests whether that recovers
# ground lost to BM25 dragging down otherwise-good embedding ranks.
WEIGHT_CONFIGS = [
    ("rrf_1_1", [1.0, 1.0]),
    ("rrf_1_2", [1.0, 2.0]),
    ("rrf_1_3", [1.0, 3.0]),
    ("rrf_1_5", [1.0, 5.0]),
    ("rrf_1_10", [1.0, 10.0]),
    ("rrf_1_15", [1.0, 15.0]),
    ("rrf_1_20", [1.0, 20.0]),
    ("rrf_1_30", [1.0, 30.0]),
    ("rrf_1_50", [1.0, 50.0]),
]


def _compute_base_rankings(bugs, candidate_pool_size, model_name):
    """BM25 + chunked-embedding rankings only -- the expensive shared inputs every RRF
    weight variant fuses from. Returns {instance_id: {"bm25": [...], "chunked_embedding": [...]}}.
    """
    rankings = {}
    for i, bug in enumerate(bugs):
        t0 = time.time()
        bm25_candidates = rank_files_bm25_with_symbols(bug, top_k=candidate_pool_size)
        t_bm25 = time.time() - t0

        if not bm25_candidates:
            rankings[bug.instance_id] = {"bm25": [], "chunked_embedding": []}
            continue

        candidate_bug = bug.model_copy(update={"code_files": bm25_candidates})
        t1 = time.time()
        embedding_ranking, timing = rank_files_embedding_chunked(candidate_bug, top_k=None, model_name=model_name)
        t_embed = time.time() - t1

        rankings[bug.instance_id] = {"bm25": bm25_candidates, "chunked_embedding": embedding_ranking}
        logger.info(
            f"[{i + 1}/{len(bugs)}] {bug.instance_id}: bm25={t_bm25:.2f}s "
            f"embed={t_embed:.2f}s ({timing.get('num_chunks', 0)} chunks over {timing.get('num_files', 0)} files)"
        )
    return rankings


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Sweep RRF weight ratios (bm25 vs. embedding) to test whether "
                    "up-weighting embedding recovers/beats chunked_embedding-alone "
                    "performance, following up on unweighted RRF losing to embedding-alone "
                    "at n=30. Computes BM25 + chunked-embedding once per instance, reuses "
                    "across every weight config. Local model, no API cost."
    )
    parser.add_argument('--manifest', required=True, help='Path to a manifest JSON')
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox'], default=None)
    parser.add_argument('--candidate-pool-size', type=int, default=200)
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
    logger.info(f"Sweeping RRF weights over {len(bugs)}/{manifest['size']} manifest instances (manifest {manifest['manifest_id']}), candidate_pool_size={args.candidate_pool_size}")

    logger.info("--- Computing base rankings (bm25 + chunked_embedding, shared across all weight configs) ---")
    base_rankings = _compute_base_rankings(bugs, args.candidate_pool_size, args.model)

    fused_rankings = {}
    for bug in bugs:
        base = base_rankings[bug.instance_id]
        fused_rankings[bug.instance_id] = {
            name: reciprocal_rank_fusion([base["bm25"], base["chunked_embedding"]], k=args.rrf_k, weights=weights)
            if base["bm25"] else []
            for name, weights in WEIGHT_CONFIGS
        }

    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    # Include the two reference points (bm25 alone, chunked_embedding alone) alongside the
    # weight sweep so this run's numbers are directly comparable to results/README.md §4
    # without cross-referencing a second file.
    config_names = ["bm25", "chunked_embedding"] + [name for name, _ in WEIGHT_CONFIGS]
    results = {}
    for name in config_names:
        if name == "bm25":
            rank_fn = lambda bug: base_rankings[bug.instance_id]["bm25"]
        elif name == "chunked_embedding":
            rank_fn = lambda bug: base_rankings[bug.instance_id]["chunked_embedding"]
        else:
            rank_fn = lambda bug, _name=name: fused_rankings[bug.instance_id][_name]
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

    embedding_mrr = results["chunked_embedding"]["summary"]["mrr"]
    best_name = max(config_names, key=lambda n: results[n]["summary"]["mrr"])
    best_mrr = results[best_name]["summary"]["mrr"]
    logger.info(f"Best MRR: {best_name} ({best_mrr:.4f}); chunked_embedding alone: {embedding_mrr:.4f}")
    if best_mrr > embedding_mrr:
        logger.info(f"VERDICT: weighted RRF ({best_name}) beats chunked_embedding alone -- fusion recovers an edge with the right weighting.")
    else:
        logger.info("VERDICT: no RRF weighting tested beats chunked_embedding alone -- embedding-alone remains the stronger reranker at this scale.")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"manifest_id": manifest["manifest_id"], "candidate_pool_size": args.candidate_pool_size, "weight_configs": dict(WEIGHT_CONFIGS), "configs": results}, f, indent=2)
        logger.info(f"Wrote full report to {args.output}")


if __name__ == "__main__":
    main()
