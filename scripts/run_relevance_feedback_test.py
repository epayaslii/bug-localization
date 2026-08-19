"""Prototype: LLM relevance feedback + algorithmic query reformulation + rerank, inspired
by BRaIn/IQLoc, per docs/relevance_feedback_scoping.md.

Update (2026-08-18): originally scoped down to file-level relevance judgments to avoid
needing a code segmenter -- that shape gave a real negative signal at n=12 on SWE-bench
(relevance filtering alone helped, reformulation on top of it hurt). Neither paper actually
validated file-level judgments, though -- BRaIn JavaParser-segments into methods, IQLoc's
cross-encoder scores "each retrieved method". This version defaults to chunk-level judgments
instead (--granularity chunk, reusing method.embedding_retriever._chunk_file_content, no new
parser needed), reformulates the hybrid BM25+embedding RRF retriever instead of BM25 alone
(--retriever hybrid-rrf, the project's stronger production signal), and can run the
relevance-feedback call against a local Ollama model instead of a cloud API (--method
ollama). Per project decision 2026-08-18: Bench4BL only from here on, not SWE-bench.

Compares three rankings over the same initial candidate pool, screened with
evaluation/screening.py so results are directly comparable to every other retrieval-only
result in this project (results/bm25_comparison_*.json, results/hybrid_rrf_*.json):

  - retriever: candidate order as returned by --retriever (bm25 or hybrid-rrf) -- baseline.
  - relevance_filtered: LLM-relevant candidates promoted to the front (original retriever
    order preserved within each group) -- isolates whether relevance filtering alone helps,
    without reformulation. At chunk granularity, a file counts as relevant if any one of its
    judged chunks does (same "max, not mean" logic as rank_files_embedding_chunked).
  - reformulated: retriever re-run over the same candidate pool using a reformulated query
    (original bug report + identifier terms extracted from LLM-relevant files/chunks) -- the
    full pipeline from the scoping doc.

All three configs are restricted to the same candidate_pool_size, so Hit@k/MRR/MAP are
directly comparable to each other (though not to full-corpus screening results at a
different top_k). Cost: 1 LLM call per bug instance (chunk-granularity prompts are larger
than file-granularity ones for the same candidate_pool_size, since a file expands into
several chunks -- see --max-chunks-per-file).
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
from method.bm25_retriever import (
    rank_files_bm25, rank_files_bm25_with_symbols, rank_files_bm25_with_skeleton,
    extract_query_reformulation_terms, _extract_symbol_tokens,
)
from method.hybrid_retriever import rank_files_hybrid
from method.embedding_retriever import _chunk_file_content
from method.openrouter_localizer import OpenRouterLocalizer
from method.ollama_localizer import OllamaLocalizer
from method.models import RelevanceFeedbackResponse, ChunkRelevanceFeedbackResponse
from method.prompt import PromptGenerator

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


_BM25_REPR_FNS = {
    "symbols_with_imports": lambda bug, top_k: rank_files_bm25_with_symbols(bug, top_k=top_k, include_imports=True),
    "symbols_no_imports": lambda bug, top_k: rank_files_bm25_with_symbols(bug, top_k=top_k, include_imports=False),
    "skeleton": lambda bug, top_k: rank_files_bm25_with_skeleton(bug, top_k=top_k),
}


def _initial_ranking(bug, retriever, candidate_pool_size, embedding_model, rrf_weights, bm25_repr):
    bm25_rank_fn = _BM25_REPR_FNS[bm25_repr]
    if retriever == "bm25":
        return bm25_rank_fn(bug, candidate_pool_size)
    ranked, _timing = rank_files_hybrid(
        bug, top_k=candidate_pool_size, candidate_pool_size=candidate_pool_size,
        embedding_model=embedding_model, weights=rrf_weights,
        bm25_rank_fn=lambda b: bm25_rank_fn(b, candidate_pool_size),
    )
    return ranked


def _rerank_with_reformulated_query(bug, candidates, reformulated_query, retriever, embedding_model, rrf_weights, bm25_repr):
    """Re-rank the SAME candidate pool with a reformulated query, using whichever
    retriever produced the original candidates -- a genuine second pass, not just a
    re-ordering, since the query text driving both BM25 and (for hybrid-rrf) the embedding
    step actually changes."""
    bm25_rank_fn = _BM25_REPR_FNS[bm25_repr]
    candidate_bug = bug.model_copy(update={"bug_report": reformulated_query, "code_files": candidates})
    if retriever == "bm25":
        return bm25_rank_fn(candidate_bug, None)
    ranked, _timing = rank_files_hybrid(
        candidate_bug, top_k=None, candidate_pool_size=len(candidates),
        embedding_model=embedding_model, weights=rrf_weights,
        bm25_rank_fn=lambda b: bm25_rank_fn(b, None),
    )
    return ranked


def _relevance_feedback_file(localizer, prompt_gen, bug, candidates, contents):
    prompt = prompt_gen.generate_relevance_feedback_prompt(bug, candidates, contents)
    response = localizer.invoke_structured(prompt, RelevanceFeedbackResponse)
    judged = {j.file: j.relevant for j in response.judgments}
    relevant = [c for c in candidates if judged.get(c) is True]
    terms = extract_query_reformulation_terms(relevant, contents)
    return relevant, judged, terms


def _relevance_feedback_chunked(localizer, prompt_gen, bug, candidates, contents, max_chunks_per_file):
    chunks = []  # list of (file, chunk_index, chunk_text)
    for path in candidates:
        file_chunks = _chunk_file_content(contents.get(path), path=path)
        for idx, chunk_text in enumerate(file_chunks[:max_chunks_per_file]):
            chunks.append((path, idx, chunk_text))

    if not chunks:
        return [], {}, []

    prompt = prompt_gen.generate_chunk_relevance_feedback_prompt(bug, chunks)
    response = localizer.invoke_structured(prompt, ChunkRelevanceFeedbackResponse)
    judged_chunks = {(j.file, j.chunk_index): j.relevant for j in response.judgments}

    chunk_by_key = {(path, idx): chunk_text for path, idx, chunk_text in chunks}
    relevant_files = []
    terms = []
    for path in candidates:
        file_chunk_indices = [idx for (p, idx, _) in chunks if p == path]
        relevant_indices = [idx for idx in file_chunk_indices if judged_chunks.get((path, idx)) is True]
        if relevant_indices:
            relevant_files.append(path)
            for idx in relevant_indices:
                chunk_text = chunk_by_key[(path, idx)]
                try:
                    symbol_tokens, _import_tokens = _extract_symbol_tokens(chunk_text, path)
                except Exception as e:
                    logger.debug(f"Chunk reformulation extraction failed for {path}#{idx}: {e}")
                    continue
                terms += symbol_tokens

    # judged, for the output file, keyed the same shape as the file-level path ("file" ->
    # bool) plus the raw chunk judgments for inspection.
    judged = {"file_level": {p: (p in relevant_files) for p in candidates}, "chunk_level": {f"{f}#{i}": v for (f, i), v in judged_chunks.items()}}
    return relevant_files, judged, terms


def _run_one(localizer, prompt_gen, bug, candidate_pool_size, retriever, embedding_model, rrf_weights, granularity, max_chunks_per_file, bm25_repr):
    candidates = _initial_ranking(bug, retriever, candidate_pool_size, embedding_model, rrf_weights, bm25_repr)
    if not candidates:
        return {
            "retriever": [], "relevance_filtered": [], "reformulated": [],
            "judged": {}, "relevant_count": 0, "reformulation_terms": [],
        }

    contents = get_file_contents_batch(bug.repo, bug.base_commit, candidates) if is_repo_cached(bug.repo) else {}

    if granularity == "file":
        relevant, judged, terms = _relevance_feedback_file(localizer, prompt_gen, bug, candidates, contents)
    else:
        relevant, judged, terms = _relevance_feedback_chunked(localizer, prompt_gen, bug, candidates, contents, max_chunks_per_file)

    not_relevant = [c for c in candidates if c not in relevant]
    relevance_filtered = relevant + not_relevant

    if terms:
        reformulated_query = bug.bug_report + "\n" + " ".join(terms)
        reformulated = _rerank_with_reformulated_query(bug, candidates, reformulated_query, retriever, embedding_model, rrf_weights, bm25_repr)
    else:
        # No relevant chunks/files found, or none had extractable identifiers -- nothing to
        # reformulate with, so fall back to the original ranking rather than a query with
        # zero expansion signal.
        reformulated = candidates

    return {
        "retriever": candidates,
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
                    "rerank (BRaIn/IQLoc-inspired, per docs/relevance_feedback_scoping.md). "
                    "1 LLM call/bug, real API/compute cost."
    )
    parser.add_argument('--manifest', required=True, help='Path to a manifest JSON')
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox', 'bench4bl'], default=None,
                       help='Project decision 2026-08-18: use bench4bl for all new work, not swebench.')
    parser.add_argument('--pool-size', type=int, default=None,
                       help='Override the manifest\'s stored pool_size when re-deriving the pool.')
    parser.add_argument('--candidate-pool-size', type=int, default=50,
                       help='Initial retriever top-K candidate pool size the relevance-feedback call judges '
                            '(smaller than the usual 100/200 to keep the prompt reasonable -- especially at '
                            '--granularity chunk, where each candidate expands into several chunks).')
    parser.add_argument('--retriever', choices=['bm25', 'hybrid-rrf'], default='hybrid-rrf',
                       help='Which retriever produces the initial candidate pool and gets re-run with the '
                            'reformulated query. hybrid-rrf is this project\'s stronger production signal '
                            '(0.714 MRR retrieval-only on Bench4BL) -- bm25 matches what BRaIn/IQLoc '
                            'themselves validated against, kept for comparison.')
    parser.add_argument('--embedding-model', default='Qwen/Qwen3-Embedding-0.6B',
                       help='With --retriever hybrid-rrf: embedding model, default matches the confirmed '
                            'n=30 Bench4BL winner.')
    parser.add_argument('--rrf-weights', default='1,5',
                       help='With --retriever hybrid-rrf: "bm25_weight,embedding_weight" (default "1,5", '
                            'the confirmed n=30 peak on Bench4BL).')
    parser.add_argument('--bm25-repr', choices=list(_BM25_REPR_FNS), default='symbols_with_imports',
                       help='BM25 document representation used both for the initial candidate pool (bm25 '
                            'retriever, or the bm25 half of hybrid-rrf) and the reformulated-query rerank. '
                            'Default matches every prior run for comparability; skeleton scores higher '
                            'standalone and in fusion on the diverse Bench4BL manifest (see '
                            'results/bm25_comparison_bench4bl_30_diverse.json / '
                            'results/hybrid_rrf_weighting_openai_skeleton_bench4bl_30_diverse.json).')
    parser.add_argument('--granularity', choices=['file', 'chunk'], default='chunk',
                       help='file: one relevance judgment per whole candidate file (original scoping, gave a '
                            'negative reformulation signal at n=12 on SWE-bench). chunk: one judgment per '
                            'method/segment (method.embedding_retriever._chunk_file_content), matching what '
                            'BRaIn/IQLoc actually validated -- reformulation terms come only from judged-'
                            'relevant chunks instead of whole files, avoiding dilution from irrelevant code '
                            'in the same file.')
    parser.add_argument('--max-chunks-per-file', type=int, default=5,
                       help='With --granularity chunk: cap chunks judged per candidate file, to bound the '
                            'prompt size growth vs. file-level (a file can have many methods).')
    parser.add_argument('--method', choices=['openrouter', 'ollama'], default='ollama',
                       help='Backend for the relevance-feedback LLM call. ollama runs fully offline (see '
                            'docs/ollama_deployment.md) -- default, per project decision 2026-08-18.')
    parser.add_argument('--model', default=None,
                       help='Defaults to qwen2.5-coder (ollama) or gpt-4o-mini (openrouter) if not passed.')
    parser.add_argument('--ollama-host', default=None)
    parser.add_argument('--num-ctx', type=int, default=16384,
                       help='With --method ollama: context window requested via extra_body (Ollama defaults to '
                            '4096 regardless of the model\'s real max, which chunk-granularity prompts can '
                            'exceed easily -- see method/ollama_localizer.py).')
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

    rrf_weights = [float(w) for w in args.rrf_weights.split(',')] if args.retriever == 'hybrid-rrf' else None

    pool_size = args.pool_size or manifest.get('pool_size') or manifest['size']
    pool = instance.get_bug_instances(sample_size=pool_size, random_sample=True, random_seed=manifest['seed'])
    wanted = {inst['instance_id'] for inst in manifest['instances']}
    bugs = [b for b in pool if b.instance_id in wanted]
    missing = wanted - {b.instance_id for b in bugs}
    if missing:
        logger.warning(f"{len(missing)} manifest instance(s) not found when re-deriving the pool: {sorted(missing)[:5]}")

    model = args.model or ('qwen2.5-coder' if args.method == 'ollama' else 'gpt-4o-mini')
    logger.info(
        f"Running relevance-feedback prototype over {len(bugs)}/{manifest['size']} manifest "
        f"instances (manifest {manifest['manifest_id']}), dataset={dataset_name}, "
        f"retriever={args.retriever}, granularity={args.granularity}, "
        f"candidate_pool_size={args.candidate_pool_size}, method={args.method}, model={model}"
    )

    if args.method == 'ollama':
        localizer = OllamaLocalizer(model=model, host=args.ollama_host, num_ctx=args.num_ctx)
    else:
        localizer = OpenRouterLocalizer(model=model)
    prompt_gen = PromptGenerator()

    per_bug = {}
    for i, bug in enumerate(bugs):
        t0 = time.time()
        result = _run_one(
            localizer, prompt_gen, bug, args.candidate_pool_size, args.retriever,
            args.embedding_model, rrf_weights, args.granularity, args.max_chunks_per_file,
            args.bm25_repr,
        )
        per_bug[bug.instance_id] = result
        logger.info(
            f"[{i + 1}/{len(bugs)}] {bug.instance_id}: {result['relevant_count']}/{len(result['retriever'])} "
            f"judged relevant, {len(result['reformulation_terms'])} reformulation terms, {time.time() - t0:.2f}s"
        )

    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    config_names = ["retriever", "relevance_filtered", "reformulated"]
    results = {}
    for name in config_names:
        rank_fn = lambda bug, _name=name: per_bug[bug.instance_id][_name]
        report = screen_manifest(bugs, token=token, cache=cache, rank_fn=rank_fn)
        summary = summarize_screening(report)
        results[name] = {"screening_report": report, "summary": summary}

    save_cache(cache)

    logger.info(f"=== Summary (macro, retriever={args.retriever}, granularity={args.granularity}, candidate_pool_size={args.candidate_pool_size}) ===")
    logger.info(f"{'config':<20} {'Hit@1':>7} {'Hit@5':>7} {'Hit@10':>7} {'MRR':>8} {'MAP':>8}")
    for name in config_names:
        s = results[name]["summary"]
        logger.info(
            f"{name:<20} {s['macro_hit_at'][1]:>7.3f} {s['macro_hit_at'][5]:>7.3f} "
            f"{s['macro_hit_at'][10]:>7.3f} {s['mrr']:>8.4f} {s['map']:>8.4f}"
        )

    retriever_mrr = results["retriever"]["summary"]["mrr"]
    best_name = max(config_names, key=lambda n: results[n]["summary"]["mrr"])
    best_mrr = results[best_name]["summary"]["mrr"]
    logger.info(f"Best MRR: {best_name} ({best_mrr:.4f}); plain {args.retriever}: {retriever_mrr:.4f}")
    if best_name != "retriever" and best_mrr > retriever_mrr:
        logger.info(f"VERDICT: {best_name} beats plain {args.retriever} -- relevance feedback / reformulation helps at this scale.")
    else:
        logger.info(f"VERDICT: neither relevance filtering nor reformulation beats plain {args.retriever} at this scale.")

    stats = localizer.total_stats()
    logger.info(f"LLM call stats: {stats}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "manifest_id": manifest["manifest_id"],
                "dataset": dataset_name,
                "retriever": args.retriever,
                "granularity": args.granularity,
                "candidate_pool_size": args.candidate_pool_size,
                "method": args.method,
                "model": model,
                "llm_call_stats": stats,
                "per_bug_detail": per_bug,
                "configs": results,
            }, f, indent=2)
        logger.info(f"Wrote full report to {args.output}")


if __name__ == "__main__":
    main()
