"""Phase 3.1 embedding-model bake-off: compares candidate embedding models against the
UniXCoder baseline used throughout the hybrid-retrieval work (see
scripts/run_hybrid_retrieval_test.py, scripts/run_hybrid_rrf_weighting_test.py), on the same
chunked-embedding ranking method (rank_files_embedding_chunked) so the comparison isolates
the embedding model itself, not the retrieval method around it.

First pass (2026-08-10): microsoft/codebert-base (same mean-pooled HF architecture as the
UniXCoder baseline) and text-embedding-3-small (OpenAI API). Extended same day to cover all
6 models named in the official study plan: BAAI/bge-code-v1 and Qwen/Qwen3-Embedding-0.6B
needed new last-token-pooling + instruction-prefixed-query support in
method/embedding_retriever.py (embed_texts()'s is_query param); Qwen3-Embedding also needed
bumping transformers 4.46.0 -> 4.51.0 (its minimum supported version -- verified UniXCoder/
CodeBERT still work unchanged after the bump, full test suite still green). voyage-code-3
needed a new API backend (plain REST via requests, VOYAGE_AI_API_KEY) once the user added
a Voyage AI key.

Run at a small n first (n=6, matching the original embedding-ceiling test's scale) since
model loading/compute cost is unknown per candidate before this script has actually run them.
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
from dataset.iqloc import IQLocExtended
from dataset.localizability import load_cache, save_cache
from dataset.utils import setup_logging, get_logger
from evaluation.manifest import load_manifest
from evaluation.screening import screen_manifest, summarize_screening
from method.embedding_retriever import rank_files_embedding_chunked

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

MODEL_CONFIGS = [
    ("unixcoder", "microsoft/unixcoder-base"),
    ("codebert", "microsoft/codebert-base"),
    ("openai-3-small", "text-embedding-3-small"),
    # bge-code-v1 was excluded from the original n=6 SWE-bench run (2026-08-10): 2B params,
    # ~16x UniXCoder's size, still mid-instance after ~18 minutes on CPU. Backend confirmed
    # working via a standalone smoke test even then (correct 1536-dim last-token-pooled
    # output) -- re-added 2026-08-18 now that MN5 has a real GPU (H100, ~30-40x speedup
    # confirmed on the embedding step for other models), which makes this model's size no
    # longer prohibitive.
    ("bge-code-v1", "BAAI/bge-code-v1"),
    ("qwen3-embedding-0.6b", "Qwen/Qwen3-Embedding-0.6B"),
    ("voyage-code-3", "voyage-code-3"),
    # Added 2026-08-20: the co-intern's own dense-retrieval candidate (his deck describes
    # runtime-only validation on MN5, no quality numbers yet at the time) -- a real code-
    # specific model neither track had tested. Needs trust_remote_code=True and `einops`
    # (see method/embedding_retriever.py's _TRUST_REMOTE_CODE_MODELS).
    ("jina-embeddings-v2-base-code", "jinaai/jina-embeddings-v2-base-code"),
]


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='Path to a manifest JSON')
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox', 'bench4bl', 'iqloc'], default=None)
    parser.add_argument('--pool-size', type=int, default=None,
                        help='With bench4bl: override the manifest\'s stored pool_size when re-deriving the pool '
                             '-- needed if this environment\'s mirror has a different total instance count than '
                             'the one the manifest was generated against (see docs/bench4bl_result.md).')
    parser.add_argument('--candidate-pool-size', type=int, default=200,
                        help='Files considered per instance (chunked embedding over the full corpus is too slow/wrong per prior hybrid-retrieval work; this caps it the same way)')
    parser.add_argument('--output', default=None)
    parser.add_argument('--models', nargs='+', default=None,
                        help='Subset of MODEL_CONFIGS names to run (default: all). '
                             'Useful to avoid redundantly recomputing models a previous run already covered.')
    args = parser.parse_args()

    model_configs = MODEL_CONFIGS if args.models is None else [
        (name, model_name) for name, model_name in MODEL_CONFIGS if name in args.models
    ]
    unknown = set(args.models or []) - {name for name, _ in MODEL_CONFIGS}
    if unknown:
        raise SystemExit(f"Unknown model name(s) in --models: {sorted(unknown)} (known: {[n for n, _ in MODEL_CONFIGS]})")

    manifest = load_manifest(args.manifest)
    dataset_name = args.dataset or manifest['dataset']
    if dataset_name == 'swebench':
        instance = SWEBench()
    elif dataset_name == 'bench4bl':
        instance = Bench4BL()
    elif dataset_name == 'iqloc':
        instance = IQLocExtended()
    else:
        instance = BeetleBox()

    pool_size = args.pool_size or manifest.get('pool_size') or manifest['size']
    pool = instance.get_bug_instances(sample_size=pool_size, random_sample=True, random_seed=manifest['seed'])
    wanted = {inst['instance_id'] for inst in manifest['instances']}
    bugs = [b for b in pool if b.instance_id in wanted]
    missing = wanted - {b.instance_id for b in bugs}
    if missing:
        logger.warning(f"{len(missing)} manifest instance(s) not found when re-deriving the pool: {sorted(missing)[:5]}")

    # Cap each instance's candidate files the same way the hybrid-retrieval scripts do, via a
    # cheap BM25 pre-filter -- otherwise chunked embedding over an entire repo's files makes
    # slower models (OpenAI API round-trips, or any future larger local model) impractical.
    from method.bm25_retriever import rank_files_bm25_with_symbols
    capped_bugs = []
    for bug in bugs:
        candidates = rank_files_bm25_with_symbols(bug, top_k=args.candidate_pool_size)
        capped_bugs.append(bug.model_copy(update={"code_files": candidates}) if candidates else bug)
    bugs = capped_bugs

    logger.info(f"Bake-off over {len(bugs)}/{manifest['size']} manifest instances (manifest {manifest['manifest_id']}), candidate_pool_size={args.candidate_pool_size}")

    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    results = {}
    for name, model_name in model_configs:
        logger.info(f"--- {name} ({model_name}) ---")
        rankings = {}
        t_start = time.time()
        for i, bug in enumerate(bugs):
            t0 = time.time()
            ranked, timing = rank_files_embedding_chunked(bug, top_k=None, model_name=model_name)
            rankings[bug.instance_id] = ranked
            logger.info(
                f"[{i + 1}/{len(bugs)}] {bug.instance_id}: {time.time() - t0:.2f}s "
                f"({timing.get('num_chunks', 0)} chunks over {timing.get('num_files', 0)} files)"
            )
        model_elapsed = time.time() - t_start

        rank_fn = lambda bug, _name=name: rankings[bug.instance_id]
        report = screen_manifest(bugs, token=token, cache=cache, rank_fn=rank_fn)
        summary = summarize_screening(report)
        results[name] = {"model_name": model_name, "elapsed_s": model_elapsed, "screening_report": report, "summary": summary}
        save_cache(cache)  # incremental -- don't lose localizability cache progress if a later model is slow/fails

        # Write output after EVERY model, not just at the end -- a later model crashing
        # (e.g. an uncaught API error) must not lose already-completed models' results.
        # This crashed a real run on its last model, uncaught, after ~55 minutes of
        # otherwise-successful work (2026-08-10).
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump({"manifest_id": manifest["manifest_id"], "candidate_pool_size": args.candidate_pool_size, "configs": results}, f, indent=2)
            logger.info(f"Wrote partial report ({len(results)}/{len(model_configs)} models) to {args.output}")

    logger.info(f"=== Summary (macro, candidate_pool_size={args.candidate_pool_size}) ===")
    logger.info(f"{'model':<16} {'Hit@1':>7} {'Hit@5':>7} {'Hit@10':>7} {'Hit@100':>8} {'MRR':>8} {'MAP':>8} {'elapsed':>9}")
    for name, _ in model_configs:
        s = results[name]["summary"]
        logger.info(
            f"{name:<16} {s['macro_hit_at'][1]:>7.3f} {s['macro_hit_at'][5]:>7.3f} "
            f"{s['macro_hit_at'][10]:>7.3f} {s['macro_hit_at'][100]:>8.3f} {s['mrr']:>8.4f} {s['map']:>8.4f} "
            f"{results[name]['elapsed_s']:>8.1f}s"
        )
    # (already written incrementally after each model above, including the final one)


if __name__ == "__main__":
    main()
