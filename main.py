import os
from dataset.swebench import SWEBench
from dataset.beetlebox import BeetleBox
from method.openai_localizer import OpenAILocalizer
from method.openai_free_localizer import OpenAIFreeLocalizer
# from method.opensource_localizer import OpenSourceLocalizer
from method.openrouter_localizer import OpenRouterLocalizer
from dataset.utils import setup_logging, get_logger
import logging
from method.evaluate import Evaluator
import argparse

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Bug Localization Tool')
    parser.add_argument('--method', choices=['openai', 'openai-free', 'unsloth', 'openrouter'], 
                       default='openrouter', help='Localization method to use')
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox'], 
                       default='beetlebox', help='Dataset to use')
    parser.add_argument('--model', default='gpt-oss-20b',
                       help='Model to use (for HuggingFace: gpt-oss, gpt-oss-120b, etc.)')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], 
                       default='auto', help='Device for local inference (HuggingFace only)')
    parser.add_argument('--sample-size', type=int, default=1,
                       help='Number of bug instances to process')
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
    parser.add_argument('--output', default=None,
                       help='Optional path to write the run config + evaluation results as JSON')

    args = parser.parse_args()
    
    logger.info("Starting...")
    logger.info(f"Method: {args.method}, Dataset: {args.dataset}, Model: {args.model}")
    
    try:
        if args.dataset == 'swebench':
            instance = SWEBench()
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
        else:
            raise ValueError(f"Unknown method: {args.method}")
        
        bug_instances = instance.get_bug_instances(sample_size=args.sample_size, random_sample=True, random_seed=42)
        
        logger.info(f"Retrieved {len(bug_instances)} bug instances")
        
        token_stats = instance.get_token_statistics()
        logger.info(f"Token statistics: {token_stats}")

        logger.info(f"Total repo: {len(instance.repos)}")
        responses = {}
        for i, bug in enumerate(bug_instances):
            if args.bm25_top_k is not None:
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
            import json
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


