"""Sweeps --keep-fraction for the embedding-cosine relevance filter + semantic reformulation
pipeline (scripts/run_relevance_feedback_test.py), to check whether the fixed 0.5 used so far
is actually the peak, the same way run_hybrid_rrf_weighting_test.py swept RRF weight ratios.

Computes the (expensive, paid-API) retriever candidates once per bug and reuses them across
every keep_fraction value -- only the relevance-cosine + reformulation stages (local/free
embedding calls) are recomputed per value, since keep_fraction inherently changes which
chunks get selected and re-embedded for reformulation.
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
from dataset.repo_cache import get_file_contents_batch, is_repo_cached
from dataset.localizability import load_cache, save_cache
from dataset.utils import setup_logging, get_logger
from evaluation.manifest import load_manifest
from evaluation.screening import screen_manifest, summarize_screening
from scripts.run_relevance_feedback_test import (
    _initial_ranking, _rerank_with_reformulated_query,
    _relevance_feedback_embedding_cosine, _semantic_reformulation_terms,
)

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

KEEP_FRACTIONS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def _compute_base(bug, retriever, candidate_pool_size, embedding_model, rrf_weights, bm25_repr):
    candidates = _initial_ranking(bug, retriever, candidate_pool_size, embedding_model, rrf_weights, bm25_repr)
    contents = get_file_contents_batch(bug.repo, bug.base_commit, candidates) if candidates and is_repo_cached(bug.repo) else {}
    return candidates, contents


def _run_one_fraction(bug, candidates, contents, keep_fraction, relevance_embedding_model, max_chunks_per_file,
                       keyword_model, top_n_keywords, retriever, embedding_model, rrf_weights, bm25_repr):
    if not candidates:
        return {"retriever": [], "relevance_filtered": [], "reformulated": []}

    relevant, _judged, _terms, relevant_chunk_texts = _relevance_feedback_embedding_cosine(
        bug, candidates, contents, relevance_embedding_model, max_chunks_per_file, keep_fraction
    )
    not_relevant = [c for c in candidates if c not in relevant]
    relevance_filtered = relevant + not_relevant

    terms = _semantic_reformulation_terms(bug, relevant_chunk_texts, keyword_model, top_n_keywords)
    if terms:
        reformulated_query = bug.bug_report + "\n" + " ".join(terms)
        reformulated = _rerank_with_reformulated_query(bug, candidates, reformulated_query, retriever, embedding_model, rrf_weights, bm25_repr)
    else:
        reformulated = candidates

    return {"retriever": candidates, "relevance_filtered": relevance_filtered, "reformulated": reformulated}


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox', 'bench4bl'], default=None)
    parser.add_argument('--pool-size', type=int, default=None)
    parser.add_argument('--candidate-pool-size', type=int, default=100)
    parser.add_argument('--retriever', choices=['bm25', 'hybrid-rrf'], default='hybrid-rrf')
    parser.add_argument('--embedding-model', default='text-embedding-3-small')
    parser.add_argument('--rrf-weights', default='1,2')
    parser.add_argument('--bm25-repr', choices=['symbols_with_imports', 'symbols_no_imports', 'skeleton'], default='skeleton')
    parser.add_argument('--max-chunks-per-file', type=int, default=5)
    parser.add_argument('--keyword-model', default='microsoft/unixcoder-base')
    parser.add_argument('--top-n-keywords', type=int, default=15)
    parser.add_argument('--relevance-embedding-model', default='microsoft/unixcoder-base')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    dataset_name = args.dataset or manifest['dataset']
    instance = {'swebench': SWEBench, 'beetlebox': BeetleBox, 'bench4bl': Bench4BL}[dataset_name]()

    rrf_weights = [float(w) for w in args.rrf_weights.split(',')] if args.retriever == 'hybrid-rrf' else None

    pool_size = args.pool_size or manifest.get('pool_size') or manifest['size']
    pool = instance.get_bug_instances(sample_size=pool_size, random_sample=True, random_seed=manifest['seed'])
    wanted = {inst['instance_id'] for inst in manifest['instances']}
    bugs = [b for b in pool if b.instance_id in wanted]
    missing = wanted - {b.instance_id for b in bugs}
    if missing:
        logger.warning(f"{len(missing)} manifest instance(s) not found when re-deriving the pool: {sorted(missing)[:5]}")

    logger.info(f"Sweeping keep_fraction over {len(bugs)}/{manifest['size']} instances (manifest {manifest['manifest_id']})")

    logger.info("--- Computing base retriever candidates (paid API, shared across all keep_fraction values) ---")
    base = {}
    for i, bug in enumerate(bugs):
        t0 = time.time()
        candidates, contents = _compute_base(bug, args.retriever, args.candidate_pool_size, args.embedding_model, rrf_weights, args.bm25_repr)
        base[bug.instance_id] = (candidates, contents)
        logger.info(f"[{i + 1}/{len(bugs)}] {bug.instance_id}: {len(candidates)} candidates, {time.time() - t0:.2f}s")

    per_bug_per_fraction = {}
    for kf in KEEP_FRACTIONS:
        logger.info(f"--- keep_fraction={kf} ---")
        for i, bug in enumerate(bugs):
            candidates, contents = base[bug.instance_id]
            t0 = time.time()
            result = _run_one_fraction(
                bug, candidates, contents, kf, args.relevance_embedding_model, args.max_chunks_per_file,
                args.keyword_model, args.top_n_keywords, args.retriever, args.embedding_model, rrf_weights, args.bm25_repr,
            )
            per_bug_per_fraction[(bug.instance_id, kf)] = result
            logger.info(f"  [{i + 1}/{len(bugs)}] {bug.instance_id}: {time.time() - t0:.2f}s")

    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    results = {}
    for kf in KEEP_FRACTIONS:
        for config_name in ["relevance_filtered", "reformulated"]:
            key = f"{config_name}_kf{kf}"
            rank_fn = lambda bug, _kf=kf, _cn=config_name: per_bug_per_fraction[(bug.instance_id, _kf)][_cn]
            report = screen_manifest(bugs, token=token, cache=cache, rank_fn=rank_fn)
            summary = summarize_screening(report)
            results[key] = {"screening_report": report, "summary": summary}

    # retriever baseline is keep_fraction-independent -- report once
    rank_fn = lambda bug: per_bug_per_fraction[(bug.instance_id, KEEP_FRACTIONS[0])]["retriever"]
    report = screen_manifest(bugs, token=token, cache=cache, rank_fn=rank_fn)
    results["retriever"] = {"screening_report": report, "summary": summarize_screening(report)}

    save_cache(cache)

    logger.info("=== Summary (macro) ===")
    logger.info(f"{'config':<30} {'Hit@1':>7} {'MRR':>8} {'MAP':>8} {'Recall@100':>11}")
    s = results["retriever"]["summary"]
    rec = s.get("macro_recall_at", {})
    logger.info(f"{'retriever':<30} {s['macro_hit_at'][1]:>7.3f} {s['mrr']:>8.4f} {s['map']:>8.4f} {rec.get(100, rec.get('100', float('nan'))):>11.3f}")
    for kf in KEEP_FRACTIONS:
        for config_name in ["relevance_filtered", "reformulated"]:
            key = f"{config_name}_kf{kf}"
            s = results[key]["summary"]
            rec = s.get("macro_recall_at", {})
            logger.info(f"{key:<30} {s['macro_hit_at'][1]:>7.3f} {s['mrr']:>8.4f} {s['map']:>8.4f} {rec.get(100, rec.get('100', float('nan'))):>11.3f}")

    best_key = max((f"reformulated_kf{kf}" for kf in KEEP_FRACTIONS), key=lambda k: results[k]["summary"]["mrr"])
    logger.info(f"Best reformulated config: {best_key} (MRR={results[best_key]['summary']['mrr']:.4f}); retriever baseline MRR={results['retriever']['summary']['mrr']:.4f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "manifest_id": manifest["manifest_id"],
                "candidate_pool_size": args.candidate_pool_size,
                "keep_fractions": KEEP_FRACTIONS,
                "configs": results,
            }, f, indent=2)
        logger.info(f"Wrote full report to {args.output}")


if __name__ == "__main__":
    main()
