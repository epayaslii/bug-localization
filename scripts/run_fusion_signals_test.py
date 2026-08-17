"""Phase 4.1 evaluation: screens each individual signal (bm25, chunked_embedding,
ast_similarity, dependency_graph, commit_recency) plus the 5-way fused combination
(rank_files_hybrid_extended) and the existing 2-signal hybrid (rank_files_hybrid, for a
side-by-side reference point) on the same manifest -- so the per-signal contribution is
visible, not just the fused end result. Mirrors the structure of
scripts/run_hybrid_rrf_weighting_test.py: compute each expensive shared input once, reuse
across every config that needs it.
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
from method.fusion_signals import rank_files_ast_similarity, rank_files_commit_recency, rank_files_dependency_graph
from method.hybrid_retriever import reciprocal_rank_fusion

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def _compute_base_rankings(bugs, candidate_pool_size, embedding_model):
    """All 5 signals, computed once per instance over the SAME BM25 candidate pool, reused
    across every config below (individual signals + both fused combos)."""
    rankings = {}
    for i, bug in enumerate(bugs):
        t0 = time.time()
        bm25_candidates = rank_files_bm25_with_symbols(bug, top_k=candidate_pool_size)
        if not bm25_candidates:
            rankings[bug.instance_id] = None
            continue

        candidate_bug = bug.model_copy(update={"code_files": bm25_candidates})
        embedding_ranking, _ = rank_files_embedding_chunked(candidate_bug, top_k=None, model_name=embedding_model)
        ast_ranking = rank_files_ast_similarity(candidate_bug, top_k=None)
        dependency_ranking = rank_files_dependency_graph(candidate_bug, bm25_candidates, top_k=None)
        recency_ranking = rank_files_commit_recency(candidate_bug, top_k=None)

        rankings[bug.instance_id] = {
            "bm25": bm25_candidates, "chunked_embedding": embedding_ranking,
            "ast_similarity": ast_ranking, "dependency_graph": dependency_ranking,
            "commit_recency": recency_ranking,
        }
        logger.info(f"[{i + 1}/{len(bugs)}] {bug.instance_id}: {time.time() - t0:.1f}s")
    return rankings


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox'], default=None)
    parser.add_argument('--candidate-pool-size', type=int, default=200)
    parser.add_argument('--embedding-model', default='microsoft/unixcoder-base')
    parser.add_argument('--rrf-k', type=int, default=60)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    dataset_name = args.dataset or manifest['dataset']
    instance = SWEBench() if dataset_name == 'swebench' else BeetleBox()

    pool_size = manifest.get('pool_size') or manifest['size']
    pool = instance.get_bug_instances(sample_size=pool_size, random_sample=True, random_seed=manifest['seed'])
    wanted = {inst['instance_id'] for inst in manifest['instances']}
    bugs = [b for b in pool if b.instance_id in wanted]

    logger.info(f"Screening {len(bugs)}/{manifest['size']} instances (manifest {manifest['manifest_id']}), candidate_pool_size={args.candidate_pool_size}")

    base = _compute_base_rankings(bugs, args.candidate_pool_size, args.embedding_model)

    signal_names = ["bm25", "chunked_embedding", "ast_similarity", "dependency_graph", "commit_recency"]
    fused_2way = {}   # bm25 + embedding only, matches rank_files_hybrid
    fused_5way = {}   # all 5 signals, matches rank_files_hybrid_extended
    for bug in bugs:
        b = base.get(bug.instance_id)
        if b is None:
            fused_2way[bug.instance_id] = []
            fused_5way[bug.instance_id] = []
            continue
        fused_2way[bug.instance_id] = reciprocal_rank_fusion([b["bm25"], b["chunked_embedding"]], k=args.rrf_k)
        fused_5way[bug.instance_id] = reciprocal_rank_fusion(
            [b[name] for name in signal_names], k=args.rrf_k
        )

    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    config_names = signal_names + ["hybrid_2way", "hybrid_5way"]
    results = {}
    for name in config_names:
        if name == "hybrid_2way":
            rank_fn = lambda bug: fused_2way[bug.instance_id]
        elif name == "hybrid_5way":
            rank_fn = lambda bug: fused_5way[bug.instance_id]
        else:
            rank_fn = lambda bug, _name=name: (base[bug.instance_id] or {}).get(_name, [])
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

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"manifest_id": manifest["manifest_id"], "candidate_pool_size": args.candidate_pool_size, "configs": results}, f, indent=2)
        logger.info(f"Wrote full report to {args.output}")


if __name__ == "__main__":
    main()
