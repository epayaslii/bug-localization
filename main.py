import os
import json
from dataset.swebench import SWEBench
from dataset.beetlebox import BeetleBox
from dataset.bench4bl import Bench4BL
from method.openai_localizer import OpenAILocalizer
from method.openai_free_localizer import OpenAIFreeLocalizer
# from method.opensource_localizer import OpenSourceLocalizer
from method.openrouter_localizer import OpenRouterLocalizer
from method.ollama_localizer import OllamaLocalizer
from dataset.utils import setup_logging, get_logger
import logging
from method.evaluate import Evaluator
import argparse

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Bug Localization Tool')
    parser.add_argument('--method', choices=['openai', 'openai-free', 'unsloth', 'openrouter', 'ollama'],
                       default='openrouter', help='Localization method to use. "ollama" talks to a local '
                            'Ollama server instead of a cloud API -- see docs/ollama_deployment.md; needed for '
                            'any environment with no outbound internet (e.g. MN5).')
    parser.add_argument('--ollama-host', default=None,
                       help='With --method ollama: override the Ollama server host (default: $OLLAMA_HOST or '
                            'http://localhost:11434).')
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox', 'bench4bl'],
                       default='bench4bl', help='Dataset to use')
    parser.add_argument('--model', default='gpt-oss-20b',
                       help='Model to use (for HuggingFace: gpt-oss, gpt-oss-120b, etc.)')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], 
                       default='auto', help='Device for local inference (HuggingFace only)')
    parser.add_argument('--sample-size', type=int, default=1,
                       help='Number of bug instances to process (ignored if --manifest is passed)')
    parser.add_argument('--manifest', default=None,
                       help='Path to a manifest JSON (evaluation/manifest.py) -- if passed, runs exactly the '
                            'manifest\'s instances instead of a fresh --sample-size random draw, matching the '
                            'pattern used by scripts/compare_bm25_representations.py and '
                            'scripts/run_hybrid_rrf_weighting_test.py.')
    parser.add_argument('--pool-size', type=int, default=None,
                       help='With --manifest: override the manifest\'s stored pool_size when re-deriving the pool '
                            '-- needed if this environment\'s dataset mirror has a different total instance count '
                            'than the one the manifest was generated against (see docs/bench4bl_result.md).')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Max number of code files to send to the model (for cheap smoke-testing)')
    parser.add_argument('--bm25-top-k', type=int, default=None,
                       help='Narrow code files to the top-K most relevant to the bug report via BM25 before prompting (disabled by default)')
    parser.add_argument('--bm25-skeleton', action='store_true',
                       help='Score BM25 using each file\'s content skeleton (docstring + class/function names) instead of just its path (requires the repo to be in the local repo_cache)')
    parser.add_argument('--bm25-symbols', action='store_true',
                       help='Score BM25 using each file\'s extracted class/function/method names instead of just its path (requires the repo to be in the local repo_cache). Takes precedence over --bm25-skeleton if both are passed.')
    parser.add_argument('--bm25-symbols-imports', action='store_true',
                       help='With --bm25-symbols, also include imported module/name tokens (off by default -- symbols-without-imports scored best in n=30 screening)')
    parser.add_argument('--retrieval-top-k', type=int, default=None,
                       help='Narrow code files to the top-K most relevant via hybrid BM25+embedding retrieval '
                            '(--retrieval-mode) before prompting. Separate from --bm25-top-k, which does BM25 alone; '
                            'takes precedence over --bm25-top-k if both are passed.')
    parser.add_argument('--retrieval-mode', choices=['hybrid-rrf', 'embedding'], default='hybrid-rrf',
                       help='With --retrieval-top-k: "hybrid-rrf" fuses BM25 + embedding via weighted Reciprocal '
                            'Rank Fusion (--rrf-weights); "embedding" uses the embedding ranking alone (no fusion). '
                            'hybrid-rrf at 1:5 is the confirmed n=30 winner on both benchmarks tested so far '
                            '(SWE-bench MRR 0.422, Bench4BL MRR 0.714 -- see docs/qwen3_rrf_result.md / '
                            'docs/bench4bl_result.md); "embedding" is offered for comparison, not as the default '
                            'recommendation.')
    parser.add_argument('--rrf-weights', default='1,5',
                       help='With --retrieval-mode hybrid-rrf: "bm25_weight,embedding_weight" (default "1,5", the '
                            'confirmed n=30 peak on both SWE-bench and Bench4BL).')
    parser.add_argument('--embedding-model', default='microsoft/unixcoder-base',
                       help='Embedding model for --retrieval-top-k. Default is UniXCoder (fast, CPU-friendly); '
                            'the 0.422/0.714 MRR numbers cited above for --retrieval-mode/--rrf-weights were both '
                            'measured with "Qwen/Qwen3-Embedding-0.6B" specifically, which scores higher but is a '
                            'slower decoder model -- pass it explicitly to reproduce those numbers.')
    parser.add_argument('--candidate-pool-size', type=int, default=200,
                       help='With --retrieval-top-k: size of the BM25 pre-filter pool reranked by the embedding step.')
    parser.add_argument('--candidates-file', default=None,
                       help='Path to a JSON {instance_id: [file_path, ...]} of precomputed retrieval candidates '
                            '-- skips retrieval entirely and uses these directly, taking precedence over '
                            '--retrieval-top-k/--bm25-top-k. For two-phase runs where retrieval itself needs to '
                            'happen somewhere without live internet (e.g. MN5) before the LLM call, which needs it '
                            '(see scripts/run_hybrid_retrieval_candidates_shard.py). An instance_id missing from '
                            'the file is skipped with a warning, not silently run unfiltered.')
    parser.add_argument('--output', default=None,
                       help='Optional path to write the run config + evaluation results as JSON')

    args = parser.parse_args()
    
    logger.info("Starting...")
    logger.info(f"Method: {args.method}, Dataset: {args.dataset}, Model: {args.model}")
    
    try:
        if args.dataset == 'swebench':
            instance = SWEBench()
        elif args.dataset == 'bench4bl':
            instance = Bench4BL()
        else:
            instance = BeetleBox()
        
        if args.method == 'openai':
            # only override the default (gpt-5-nano) if the user explicitly picked
            # a non-default --model; the flag's own default is tailored to openrouter
            if args.model and args.model != parser.get_default('model'):
                localizer = OpenAILocalizer(model=args.model)
            else:
                localizer = OpenAILocalizer()
        elif args.method == 'openai-free':
            localizer = OpenAIFreeLocalizer(model=args.model)
        elif args.method == 'unsloth':
            device = None if args.device == 'auto' else args.device
            localizer = OpenSourceLocalizer(
                model=args.model,
                device=device
            )
        elif args.method == 'openrouter':
            localizer = OpenRouterLocalizer(model=args.model)
        elif args.method == 'ollama':
            model = args.model if args.model != parser.get_default('model') else 'qwen2.5-coder'
            localizer = OllamaLocalizer(model=model, host=args.ollama_host)
        else:
            raise ValueError(f"Unknown method: {args.method}")
        
        if args.manifest:
            from evaluation.manifest import load_manifest
            manifest = load_manifest(args.manifest)
            pool_size = args.pool_size or manifest.get('pool_size') or manifest['size']
            pool = instance.get_bug_instances(sample_size=pool_size, random_sample=True, random_seed=manifest['seed'])
            wanted = {inst['instance_id'] for inst in manifest['instances']}
            bug_instances = [b for b in pool if b.instance_id in wanted]
            missing = wanted - {b.instance_id for b in bug_instances}
            if missing:
                logger.warning(f"{len(missing)} manifest instance(s) not found when re-deriving the pool: {sorted(missing)[:5]}")
            logger.info(f"Loaded {len(bug_instances)}/{manifest['size']} manifest instances (manifest {manifest['manifest_id']})")
        else:
            bug_instances = instance.get_bug_instances(sample_size=args.sample_size, random_sample=True, random_seed=42)

        logger.info(f"Retrieved {len(bug_instances)} bug instances")
        
        token_stats = instance.get_token_statistics()
        logger.info(f"Token statistics: {token_stats}")

        logger.info(f"Total repo: {len(instance.repos)}")

        precomputed_candidates = None
        if args.candidates_file:
            with open(args.candidates_file) as f:
                precomputed_candidates = json.load(f)
            logger.info(f"Loaded precomputed candidates for {len(precomputed_candidates)} instances from {args.candidates_file}")
            missing_candidates = {b.instance_id for b in bug_instances} - set(precomputed_candidates)
            if missing_candidates:
                logger.warning(f"{len(missing_candidates)} instance(s) missing from --candidates-file, will be skipped: {sorted(missing_candidates)[:5]}")
                bug_instances = [b for b in bug_instances if b.instance_id in precomputed_candidates]

        responses = {}
        for i, bug in enumerate(bug_instances):
            if precomputed_candidates is not None:
                original_count = len(bug.code_files)
                bug.code_files = precomputed_candidates[bug.instance_id]
                logger.info(f"Precomputed-candidates filtered code files from {original_count} to {len(bug.code_files)} (--candidates-file)")
            elif args.retrieval_top_k is not None:
                original_count = len(bug.code_files)
                if args.retrieval_mode == 'embedding':
                    from method.embedding_retriever import rank_files_embedding_chunked
                    from method.bm25_retriever import rank_files_bm25_with_symbols
                    bm25_candidates = rank_files_bm25_with_symbols(bug, top_k=args.candidate_pool_size)
                    candidate_bug = bug.model_copy(update={"code_files": bm25_candidates})
                    ranked, _ = rank_files_embedding_chunked(
                        candidate_bug, top_k=None, model_name=args.embedding_model
                    )
                    bug.code_files = ranked[:args.retrieval_top_k]
                    mode = f"embedding({args.embedding_model})"
                else:
                    from method.hybrid_retriever import rank_files_hybrid
                    weights = [float(w) for w in args.rrf_weights.split(',')]
                    ranked, _ = rank_files_hybrid(
                        bug, top_k=args.retrieval_top_k, candidate_pool_size=args.candidate_pool_size,
                        embedding_model=args.embedding_model, weights=weights,
                    )
                    bug.code_files = ranked
                    mode = f"hybrid-rrf(weights={weights}, embedding={args.embedding_model})"
                logger.info(f"Retrieval-filtered code files from {original_count} to {len(bug.code_files)} (--retrieval-top-k={args.retrieval_top_k}, mode={mode})")
            elif args.bm25_top_k is not None:
                original_count = len(bug.code_files)
                if args.bm25_symbols:
                    from method.bm25_retriever import rank_files_bm25_with_symbols
                    bug.code_files = rank_files_bm25_with_symbols(
                        bug, top_k=args.bm25_top_k, include_imports=args.bm25_symbols_imports
                    )
                    mode = f"symbols(imports={args.bm25_symbols_imports})"
                elif args.bm25_skeleton:
                    from method.bm25_retriever import rank_files_bm25_with_skeleton
                    bug.code_files = rank_files_bm25_with_skeleton(bug, top_k=args.bm25_top_k)
                    mode = "skeleton"
                else:
                    from method.bm25_retriever import rank_files_bm25
                    bug.code_files = rank_files_bm25(bug.bug_report, bug.code_files, top_k=args.bm25_top_k)
                    mode = "path_only"
                logger.info(f"BM25-filtered code files from {original_count} to {len(bug.code_files)} (--bm25-top-k={args.bm25_top_k}, mode={mode})")
            if args.max_files is not None:
                bug.code_files = bug.code_files[:args.max_files]
                logger.info(f"Truncated code files to {len(bug.code_files)} (--max-files={args.max_files})")
            response = localizer.localize(bug)
            responses[bug.instance_id] = {'bug': bug, 'response': response}
            logger.info(f"Response: {response}")
        
        evaluator = Evaluator()
        results = evaluator.evaluate(responses)
        logger.info(f"Results: {results}")

        if args.output:
            bm25_mode = None
            if args.bm25_top_k is not None:
                bm25_mode = "symbols" if args.bm25_symbols else ("skeleton" if args.bm25_skeleton else "path_only")
            report = {
                "method": args.method,
                "dataset": args.dataset,
                "model": args.model,
                "sample_size": args.sample_size,
                "bm25_top_k": args.bm25_top_k,
                "bm25_mode": bm25_mode,
                "bm25_symbols_imports": args.bm25_symbols_imports if args.bm25_symbols else None,
                "retrieval_top_k": args.retrieval_top_k,
                "retrieval_mode": args.retrieval_mode if args.retrieval_top_k is not None else None,
                "rrf_weights": args.rrf_weights if (args.retrieval_top_k is not None and args.retrieval_mode == 'hybrid-rrf') else None,
                "embedding_model": args.embedding_model if args.retrieval_top_k is not None else None,
                "per_bug": {
                    instance_id: {
                        "repo": data["bug"].repo,
                        "ground_truths": data["bug"].ground_truths,
                        "candidate_files": data["response"].candidate_files if data["response"] else [],
                        **results["per_bug"][instance_id],
                    }
                    for instance_id, data in responses.items()
                },
                "overall": results["overall"],
            }
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Wrote full report to {args.output}")

        if hasattr(localizer, 'cleanup'):
            localizer.cleanup()
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        if 'localizer' in locals() and hasattr(localizer, 'cleanup'):
            localizer.cleanup()
        raise


if __name__ == "__main__":
    main()


