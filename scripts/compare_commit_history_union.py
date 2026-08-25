"""Measures the real impact of unioning commit-message-based candidates with content-based
BM25 retrieval (method/commit_history_retriever.py) -- cheap, offline, no LLM/API cost.
Compares content-only vs. content+history-union on the same manifest and candidate_pool_size,
so Recall@k/Hit@k/MRR/MAP are directly comparable.
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
from dataset.bench4bl import Bench4BL
from dataset.iqloc import IQLocExtended
from dataset.localizability import load_cache, save_cache
from dataset.utils import setup_logging, get_logger
from evaluation.manifest import load_manifest
from evaluation.screening import screen_manifest, summarize_screening
from method.bm25_retriever import rank_files_bm25_with_skeleton
from method.commit_history_retriever import rank_files_commit_history, rank_files_commit_history_scored, union_candidates
from method.hybrid_retriever import reciprocal_rank_fusion

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox', 'bench4bl', 'iqloc'], default=None)
    parser.add_argument('--pool-size', type=int, default=None)
    parser.add_argument('--candidate-pool-size', type=int, default=100,
                       help='Content-based BM25 top-K; commit-history candidates are unioned on top of this, '
                            'so the union config can exceed this size (recall-oriented, not a fixed budget).')
    parser.add_argument('--max-commits', type=int, default=3000)
    parser.add_argument('--top-k-commits', type=int, default=20)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    dataset_name = args.dataset or manifest['dataset']
    instance = {'swebench': SWEBench, 'beetlebox': BeetleBox, 'bench4bl': Bench4BL, 'iqloc': IQLocExtended}[dataset_name]()

    pool_size = args.pool_size or manifest.get('pool_size') or manifest['size']
    pool = instance.get_bug_instances(sample_size=pool_size, random_sample=True, random_seed=manifest['seed'])
    wanted = {inst['instance_id'] for inst in manifest['instances']}
    bugs = [b for b in pool if b.instance_id in wanted]
    missing = wanted - {b.instance_id for b in bugs}
    if missing:
        logger.warning(f"{len(missing)} manifest instance(s) not found when re-deriving the pool: {sorted(missing)[:5]}")

    logger.info(f"Comparing content-only vs. content+commit-history-union vs. RRF-reranked over {len(bugs)}/{manifest['size']} instances")

    per_bug = {}
    for i, bug in enumerate(bugs):
        content = rank_files_bm25_with_skeleton(bug, top_k=args.candidate_pool_size)
        history = rank_files_commit_history(bug, max_commits=args.max_commits, top_k_commits=args.top_k_commits)
        history_scored = rank_files_commit_history_scored(bug, max_commits=args.max_commits, top_k_commits=args.top_k_commits)
        union = union_candidates(content, history)
        rerank = reciprocal_rank_fusion([content, history_scored])
        new_files = [f for f in history if f not in content]
        gt_only_via_history = [g for g in bug.ground_truths if g in history and g not in content]
        per_bug[bug.instance_id] = {"content": content, "union": union, "rerank": rerank}
        logger.info(
            f"[{i + 1}/{len(bugs)}] {bug.instance_id}: content={len(content)} history={len(history)} "
            f"union={len(union)} new_from_history={len(new_files)} gt_only_via_history={len(gt_only_via_history)}"
        )

    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    results = {}
    for name in ["content", "union", "rerank"]:
        rank_fn = lambda bug, _name=name: per_bug[bug.instance_id][_name]
        report = screen_manifest(bugs, token=token, cache=cache, rank_fn=rank_fn)
        summary = summarize_screening(report)
        results[name] = {"screening_report": report, "summary": summary}

    save_cache(cache)

    logger.info("=== Summary (macro) ===")
    logger.info(f"{'config':<10} {'Hit@1':>7} {'Hit@5':>7} {'Hit@10':>7} {'MRR':>8} {'MAP':>8} {'Recall@100':>11}")
    for name in ["content", "union", "rerank"]:
        s = results[name]["summary"]
        rec = s.get("macro_recall_at", {})
        logger.info(
            f"{name:<10} {s['macro_hit_at'][1]:>7.3f} {s['macro_hit_at'][5]:>7.3f} "
            f"{s['macro_hit_at'][10]:>7.3f} {s['mrr']:>8.4f} {s['map']:>8.4f} "
            f"{rec.get(100, rec.get('100', float('nan'))):>11.3f}"
        )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"manifest_id": manifest["manifest_id"], "configs": results}, f, indent=2)
        logger.info(f"Wrote full report to {args.output}")


if __name__ == "__main__":
    main()
