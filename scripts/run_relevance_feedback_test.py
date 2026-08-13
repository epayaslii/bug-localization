"""Prototype: LLM relevance feedback + algorithmic query reformulation + BM25 rerank,
inspired by BRaIn/IQLoc and scoped down per docs/relevance_feedback_scoping.md to ~1 LLM
call per bug (file-level relevance judged in a single batched call across the whole
candidate pool, not one call per candidate -- BRaIn's own design costs 150-250 calls/bug
at segment granularity, which this project's budget can't absorb).

Compares three rankings over the same BM25 top-K candidate pool, screened with
evaluation/screening.py so results are directly comparable to every other retrieval-only
result in this project (results/bm25_comparison_*.json, results/hybrid_rrf_*.json):

  - bm25: candidate order as returned by BM25 (symbols+imports representation) -- baseline.
  - relevance_filtered: LLM-relevant candidates promoted to the front (original BM25 order
    preserved within each group) -- isolates whether relevance filtering alone helps,
    without reformulation.
  - reformulated: BM25 re-run over the same candidate pool using a reformulated query
    (original bug report + identifier terms extracted from LLM-relevant files) -- the full
    pipeline from the scoping doc.

All three configs are restricted to the same candidate_pool_size, so Hit@k/MRR/MAP are
directly comparable to each other (though not to full-corpus BM25 screening results at a
different top_k). Cost: 1 LLM call per bug instance.
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
from method.bm25_retriever import rank_files_bm25, rank_files_bm25_with_symbols, extract_query_reformulation_terms
from method.openrouter_localizer import OpenRouterLocalizer
from method.models import RelevanceFeedbackResponse
from method.prompt import PromptGenerator

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def _relevance_feedback(localizer, prompt_gen, bug, candidates, contents):
    prompt = prompt_gen.generate_relevance_feedback_prompt(bug, candidates, contents)
    response = localizer.invoke_structured(prompt, RelevanceFeedbackResponse)
    judged = {j.file: j.relevant for j in response.judgments}
    # Candidates the LLM didn't return a judgment for (schema drift, truncation) are
    # treated as not-relevant here -- they just keep their original BM25 rank in the
    # relevance_filtered config below rather than being dropped from consideration.
    relevant = [c for c in candidates if judged.get(c) is True]
    return relevant, judged


def _run_one(localizer, prompt_gen, bug, candidate_pool_size):
    bm25_candidates = rank_files_bm25_with_symbols(bug, top_k=candidate_pool_size)
    if not bm25_candidates:
        return {
            "bm25": [], "relevance_filtered": [], "reformulated": [],
            "judged": {}, "relevant_count": 0, "reformulation_terms": [],
        }

    contents = get_file_contents_batch(bug.repo, bug.base_commit, bm25_candidates) if is_repo_cached(bug.repo) else {}

    relevant, judged = _relevance_feedback(localizer, prompt_gen, bug, bm25_candidates, contents)
    not_relevant = [c for c in bm25_candidates if c not in relevant]
    relevance_filtered = relevant + not_relevant

    terms = extract_query_reformulation_terms(relevant, contents)
    if terms:
        reformulated_query = bug.bug_report + "\n" + " ".join(terms)
        reformulated = rank_files_bm25(reformulated_query, bm25_candidates, top_k=None)
    else:
        # No relevant files found, or none had extractable identifiers -- nothing to
        # reformulate with, so fall back to the original BM25 ranking rather than a query
        # with zero expansion signal.
        reformulated = bm25_candidates

    return {
        "bm25": bm25_candidates,
        "relevance_filtered": relevance_filtered,
        "reformulated": reformulated,
        "judged": judged,
        "relevant_count": len(relevant),
        "reformulation_terms": terms[:30],
    }


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Prototype LLM relevance feedback + algorithmic query reformulation + "
                    "BM25 rerank (BRaIn/IQLoc-inspired, scoped per "
                    "docs/relevance_feedback_scoping.md). 1 LLM call/bug, real API cost."
    )
    parser.add_argument('--manifest', required=True, help='Path to a manifest JSON')
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox', 'bench4bl'], default=None)
    parser.add_argument('--pool-size', type=int, default=None,
                       help='Override the manifest\'s stored pool_size when re-deriving the pool.')
    parser.add_argument('--candidate-pool-size', type=int, default=50,
                       help='BM25 top-K candidate pool size the relevance-feedback call judges '
                            '(smaller than the usual 100/200 to keep the prompt reasonable).')
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    dataset_name = args.dataset or manifest['dataset']
    if dataset_name == 'swebench':
        instance = SWEBench()
    elif dataset_name == 'bench4bl':
        instance = Bench4BL()
    else:
        instance = BeetleBox()

    pool_size = args.pool_size or manifest.get('pool_size') or manifest['size']
    pool = instance.get_bug_instances(sample_size=pool_size, random_sample=True, random_seed=manifest['seed'])
    wanted = {inst['instance_id'] for inst in manifest['instances']}
    bugs = [b for b in pool if b.instance_id in wanted]
    missing = wanted - {b.instance_id for b in bugs}
    if missing:
        logger.warning(f"{len(missing)} manifest instance(s) not found when re-deriving the pool: {sorted(missing)[:5]}")
    logger.info(
        f"Running relevance-feedback prototype over {len(bugs)}/{manifest['size']} manifest "
        f"instances (manifest {manifest['manifest_id']}), candidate_pool_size={args.candidate_pool_size}, model={args.model}"
    )

    localizer = OpenRouterLocalizer(model=args.model)
    prompt_gen = PromptGenerator()

    per_bug = {}
    for i, bug in enumerate(bugs):
        t0 = time.time()
        result = _run_one(localizer, prompt_gen, bug, args.candidate_pool_size)
        per_bug[bug.instance_id] = result
        logger.info(
            f"[{i + 1}/{len(bugs)}] {bug.instance_id}: {result['relevant_count']}/{len(result['bm25'])} "
            f"judged relevant, {len(result['reformulation_terms'])} reformulation terms, {time.time() - t0:.2f}s"
        )

    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    config_names = ["bm25", "relevance_filtered", "reformulated"]
    results = {}
    for name in config_names:
        rank_fn = lambda bug, _name=name: per_bug[bug.instance_id][_name]
        report = screen_manifest(bugs, token=token, cache=cache, rank_fn=rank_fn)
        summary = summarize_screening(report)
        results[name] = {"screening_report": report, "summary": summary}

    save_cache(cache)

    logger.info(f"=== Summary (macro, candidate_pool_size={args.candidate_pool_size}) ===")
    logger.info(f"{'config':<20} {'Hit@1':>7} {'Hit@5':>7} {'Hit@10':>7} {'MRR':>8} {'MAP':>8}")
    for name in config_names:
        s = results[name]["summary"]
        logger.info(
            f"{name:<20} {s['macro_hit_at'][1]:>7.3f} {s['macro_hit_at'][5]:>7.3f} "
            f"{s['macro_hit_at'][10]:>7.3f} {s['mrr']:>8.4f} {s['map']:>8.4f}"
        )

    bm25_mrr = results["bm25"]["summary"]["mrr"]
    best_name = max(config_names, key=lambda n: results[n]["summary"]["mrr"])
    best_mrr = results[best_name]["summary"]["mrr"]
    logger.info(f"Best MRR: {best_name} ({best_mrr:.4f}); plain bm25: {bm25_mrr:.4f}")
    if best_name != "bm25" and best_mrr > bm25_mrr:
        logger.info(f"VERDICT: {best_name} beats plain BM25 -- relevance feedback / reformulation helps at this scale.")
    else:
        logger.info("VERDICT: neither relevance filtering nor reformulation beats plain BM25 at this scale.")

    stats = localizer.total_stats()
    logger.info(f"LLM call stats: {stats}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "manifest_id": manifest["manifest_id"],
                "candidate_pool_size": args.candidate_pool_size,
                "model": args.model,
                "llm_call_stats": stats,
                "per_bug_detail": per_bug,
                "configs": results,
            }, f, indent=2)
        logger.info(f"Wrote full report to {args.output}")


if __name__ == "__main__":
    main()
