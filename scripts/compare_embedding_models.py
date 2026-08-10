"""Phase 3.1 embedding-model bake-off: compares candidate embedding models against the
UniXCoder baseline used throughout the hybrid-retrieval work (see
scripts/run_hybrid_retrieval_test.py, scripts/run_hybrid_rrf_weighting_test.py), on the same
chunked-embedding ranking method (rank_files_embedding_chunked) so the comparison isolates
the embedding model itself, not the retrieval method around it.

First pass (2026-08-10, user-scoped via AskUserQuestion): microsoft/codebert-base (same
mean-pooled HF architecture as the UniXCoder baseline, zero code changes needed) and
text-embedding-3-small (OpenAI API, paid but cheap -- embed_texts() dispatches to the API
for this model name). Qwen3-Embedding and BGE-Code-v1 use last-token pooling + an
instruction-prefixed query, not mean pooling -- deliberately out of scope for this pass;
Voyage-Code needs a new API key the user hasn't set up yet.

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
]


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='Path to a manifest JSON')
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox'], default=None)
    parser.add_argument('--candidate-pool-size', type=int, default=200,
                        help='Files considered per instance (chunked embedding over the full corpus is too slow/wrong per prior hybrid-retrieval work; this caps it the same way)')
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
    for name, model_name in MODEL_CONFIGS:
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

    logger.info(f"=== Summary (macro, candidate_pool_size={args.candidate_pool_size}) ===")
    logger.info(f"{'model':<16} {'Hit@1':>7} {'Hit@5':>7} {'Hit@10':>7} {'Hit@100':>8} {'MRR':>8} {'MAP':>8} {'elapsed':>9}")
    for name, _ in MODEL_CONFIGS:
        s = results[name]["summary"]
        logger.info(
            f"{name:<16} {s['macro_hit_at'][1]:>7.3f} {s['macro_hit_at'][5]:>7.3f} "
            f"{s['macro_hit_at'][10]:>7.3f} {s['macro_hit_at'][100]:>8.3f} {s['mrr']:>8.4f} {s['map']:>8.4f} "
            f"{results[name]['elapsed_s']:>8.1f}s"
        )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"manifest_id": manifest["manifest_id"], "candidate_pool_size": args.candidate_pool_size, "configs": results}, f, indent=2)
        logger.info(f"Wrote full report to {args.output}")


if __name__ == "__main__":
    main()
