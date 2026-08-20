"""Approximation of IQLoc's published pipeline (arxiv.org/html/2510.04468v2), per the
"approximate with what we already have" scope decision in docs/iqloc_replication_scoping.md
-- not a from-scratch retrain of their fine-tuned CodeBERT cross-encoder or domain-pretrained
CodeT5 (see that doc for why that's a materially bigger lift), but the same 5-stage structure
with substitutions from this project's existing toolkit:

  IQLoc stage                          | This approximation
  --------------------------------------|--------------------------------------------------
  BM25 top-100 (Elasticsearch)          | BM25 top-K, skeleton representation (this
                                         | project's confirmed-best BM25 config, see
                                         | results/bm25_comparison_bench4bl_30_diverse.json)
  Fine-tuned CodeBERT cross-encoder,    | Local Ollama LLM, one batched chunk-relevance call
  per-method scoring                    | per bug (this project's existing relevance-feedback
                                         | mechanism, method/method.embedding_retriever's
                                         | _chunk_file_content granularity)
  EmbedRank/MMR keywords (bug report    | Same algorithm (method/keyword_extraction.py,
  side), CodeT5 embeddings              | no KeyBERT dependency), embedding model configurable
  EmbedRank/MMR keywords (code side,    | Same, over LLM-judged-relevant chunk text
  only cross-encoder-relevant segments) |
  Cosine-similarity query reformulation | Same mechanism (keyword_extraction.reformulate_
  (bug-report keywords vs code keywords)| query_iqloc_style)
  BM25 rerank with reformulated query   | BM25 (skeleton) rerank, same candidate pool

Compares three rankings over the same initial candidate pool, screened with
evaluation/screening.py: retriever (BM25 skeleton alone), relevance_filtered (LLM-relevant
chunks' files promoted to front, isolates the cross-encoder-analog stage), iqloc_reformulated
(the full approximated pipeline). Cost: 1 LLM call/bug (Ollama, offline) + up to 4 embedding
calls/bug (doc + candidates, bug-report side and code side) for keyword extraction.
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
from method.bm25_retriever import rank_files_bm25_with_skeleton
from method.embedding_retriever import _chunk_file_content
from method.keyword_extraction import embedrank_mmr_keywords, reformulate_query_iqloc_style
from method.ollama_localizer import OllamaLocalizer
from method.openrouter_localizer import OpenRouterLocalizer
from method.models import ChunkRelevanceFeedbackResponse
from method.prompt import PromptGenerator

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def _relevance_judge_chunked(localizer, prompt_gen, bug, candidates, contents, max_chunks_per_file):
    """Same mechanism as scripts/run_relevance_feedback_test.py's _relevance_feedback_chunked,
    but also returns the judged-relevant chunks' raw text (not just file names) -- IQLoc's
    code-side keyword extraction needs the actual relevant segment text, not just which files
    contain one."""
    chunks = []
    for path in candidates:
        file_chunks = _chunk_file_content(contents.get(path), path=path)
        for idx, chunk_text in enumerate(file_chunks[:max_chunks_per_file]):
            chunks.append((path, idx, chunk_text))

    if not chunks:
        return [], []

    prompt = prompt_gen.generate_chunk_relevance_feedback_prompt(bug, chunks)
    response = localizer.invoke_structured(prompt, ChunkRelevanceFeedbackResponse)
    judged_chunks = {(j.file, j.chunk_index): j.relevant for j in response.judgments}

    relevant_files = []
    relevant_chunk_texts = []
    for path in candidates:
        file_chunk_indices = [(idx, text) for (p, idx, text) in chunks if p == path]
        relevant = [(idx, text) for idx, text in file_chunk_indices if judged_chunks.get((path, idx)) is True]
        if relevant:
            relevant_files.append(path)
            relevant_chunk_texts.extend(text for _idx, text in relevant)

    return relevant_files, relevant_chunk_texts


def _run_one(localizer, prompt_gen, bug, candidate_pool_size, max_chunks_per_file, keyword_model, top_n_keywords):
    candidates = rank_files_bm25_with_skeleton(bug, top_k=candidate_pool_size)
    if not candidates:
        return {
            "retriever": [], "relevance_filtered": [], "iqloc_reformulated": [],
            "relevant_count": 0, "bug_report_keywords": [], "code_keywords": [], "reformulation_terms": [],
        }

    contents = get_file_contents_batch(bug.repo, bug.base_commit, candidates) if is_repo_cached(bug.repo) else {}

    relevant_files, relevant_chunk_texts = _relevance_judge_chunked(
        localizer, prompt_gen, bug, candidates, contents, max_chunks_per_file
    )
    not_relevant = [c for c in candidates if c not in relevant_files]
    relevance_filtered = relevant_files + not_relevant

    bug_report_keywords = embedrank_mmr_keywords(bug.bug_report, model_name=keyword_model, top_n=top_n_keywords)

    if relevant_chunk_texts and bug_report_keywords:
        code_text = "\n".join(relevant_chunk_texts)
        code_keywords = embedrank_mmr_keywords(code_text, model_name=keyword_model, top_n=top_n_keywords)
        reformulation_terms = reformulate_query_iqloc_style(
            bug_report_keywords, code_keywords, model_name=keyword_model, top_matches=top_n_keywords
        )
    else:
        code_keywords, reformulation_terms = [], []

    if reformulation_terms:
        reformulated_query = bug.bug_report + "\n" + " ".join(reformulation_terms)
        candidate_bug = bug.model_copy(update={"bug_report": reformulated_query, "code_files": candidates})
        iqloc_reformulated = rank_files_bm25_with_skeleton(candidate_bug, top_k=None)
    else:
        iqloc_reformulated = candidates

    return {
        "retriever": candidates,
        "relevance_filtered": relevance_filtered,
        "iqloc_reformulated": iqloc_reformulated,
        "relevant_count": len(relevant_files),
        "bug_report_keywords": bug_report_keywords,
        "code_keywords": code_keywords,
        "reformulation_terms": reformulation_terms,
    }


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Approximation of IQLoc's 5-stage pipeline (BM25 -> LLM relevance judge "
                    "in place of a fine-tuned cross-encoder -> EmbedRank/MMR keyword "
                    "extraction -> cosine-similarity reformulation -> BM25 rerank). See "
                    "docs/iqloc_replication_scoping.md for what is/isn't a faithful replication."
    )
    parser.add_argument('--manifest', required=True, help='Path to a manifest JSON')
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox', 'bench4bl'], default=None)
    parser.add_argument('--pool-size', type=int, default=None,
                       help='Override the manifest\'s stored pool_size when re-deriving the pool.')
    parser.add_argument('--candidate-pool-size', type=int, default=50,
                       help='BM25 top-K candidate pool (IQLoc uses K=100; smaller default here '
                            'to bound the chunk-relevance LLM prompt size, matching '
                            'run_relevance_feedback_test.py\'s convention).')
    parser.add_argument('--max-chunks-per-file', type=int, default=5)
    parser.add_argument('--top-n-keywords', type=int, default=15,
                       help='IQLoc\'s own N sweep found 15 the knee of the MAP/MRR-vs-N curve.')
    parser.add_argument('--keyword-model', default='microsoft/unixcoder-base',
                       help='Embedding model for EmbedRank/MMR keyword extraction. IQLoc uses a '
                            'domain-pretrained CodeT5 this project does not have; defaults to a '
                            'free local model to keep this cheap to iterate on.')
    parser.add_argument('--method', choices=['openrouter', 'ollama'], default='ollama')
    parser.add_argument('--model', default=None,
                       help='Defaults to qwen2.5-coder-7b (ollama) or gpt-4o-mini (openrouter).')
    parser.add_argument('--ollama-host', default=None)
    parser.add_argument('--num-ctx', type=int, default=16384)
    parser.add_argument('--max-tokens', type=int, default=8192,
                       help='Raised from OllamaLocalizer\'s own 4096 default -- a batched chunk-relevance '
                            'judgment response (one JSON object per chunk, up to candidate_pool_size * '
                            'max_chunks_per_file of them) can exceed 4096 tokens and get truncated '
                            'mid-string. Confirmed on the n=200 run: 3/200 (1.5%) instances failed to '
                            'parse for exactly this reason before this fix.')
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

    model = args.model or ('qwen2.5-coder-7b' if args.method == 'ollama' else 'gpt-4o-mini')
    logger.info(
        f"Running IQLoc approximation over {len(bugs)}/{manifest['size']} manifest instances "
        f"(manifest {manifest['manifest_id']}), dataset={dataset_name}, "
        f"candidate_pool_size={args.candidate_pool_size}, keyword_model={args.keyword_model}, "
        f"top_n_keywords={args.top_n_keywords}, method={args.method}, model={model}"
    )

    if args.method == 'ollama':
        localizer = OllamaLocalizer(model=model, host=args.ollama_host, num_ctx=args.num_ctx, max_tokens=args.max_tokens)
    else:
        localizer = OpenRouterLocalizer(model=model)
    prompt_gen = PromptGenerator()

    per_bug = {}
    for i, bug in enumerate(bugs):
        t0 = time.time()
        result = _run_one(
            localizer, prompt_gen, bug, args.candidate_pool_size, args.max_chunks_per_file,
            args.keyword_model, args.top_n_keywords,
        )
        per_bug[bug.instance_id] = result
        logger.info(
            f"[{i + 1}/{len(bugs)}] {bug.instance_id}: {result['relevant_count']}/{len(result['retriever'])} "
            f"judged relevant, {len(result['reformulation_terms'])} reformulation terms, {time.time() - t0:.2f}s"
        )

    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    config_names = ["retriever", "relevance_filtered", "iqloc_reformulated"]
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

    retriever_mrr = results["retriever"]["summary"]["mrr"]
    best_name = max(config_names, key=lambda n: results[n]["summary"]["mrr"])
    best_mrr = results[best_name]["summary"]["mrr"]
    logger.info(f"Best MRR: {best_name} ({best_mrr:.4f}); plain BM25 retriever: {retriever_mrr:.4f}")
    if best_name != "retriever" and best_mrr > retriever_mrr:
        logger.info(f"VERDICT: {best_name} beats plain BM25 -- the IQLoc-approximated pipeline helps at this scale.")
    else:
        logger.info("VERDICT: neither relevance filtering nor IQLoc-style reformulation beats plain BM25 at this scale.")

    stats = localizer.total_stats()
    logger.info(f"LLM call stats: {stats}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "manifest_id": manifest["manifest_id"],
                "dataset": dataset_name,
                "candidate_pool_size": args.candidate_pool_size,
                "keyword_model": args.keyword_model,
                "top_n_keywords": args.top_n_keywords,
                "method": args.method,
                "model": model,
                "llm_call_stats": stats,
                "per_bug_detail": per_bug,
                "configs": results,
            }, f, indent=2)
        logger.info(f"Wrote full report to {args.output}")


if __name__ == "__main__":
    main()
