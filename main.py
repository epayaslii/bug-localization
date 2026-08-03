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
                if args.bm25_skeleton:
                    from method.bm25_retriever import rank_files_bm25_with_skeleton
                    bug.code_files = rank_files_bm25_with_skeleton(bug, top_k=args.bm25_top_k)
                else:
                    from method.bm25_retriever import rank_files_bm25
                    bug.code_files = rank_files_bm25(bug.bug_report, bug.code_files, top_k=args.bm25_top_k)
                logger.info(f"BM25-filtered code files from {original_count} to {len(bug.code_files)} (--bm25-top-k={args.bm25_top_k}, skeleton={args.bm25_skeleton})")
            if args.max_files is not None:
                bug.code_files = bug.code_files[:args.max_files]
                logger.info(f"Truncated code files to {len(bug.code_files)} (--max-files={args.max_files})")
            response = localizer.localize(bug)
            responses[bug.instance_id] = {'bug': bug, 'response': response}
            logger.info(f"Response: {response}")
        
        evaluator = Evaluator()
        results = evaluator.evaluate(responses)
        logger.info(f"Results: {results}")
        
        if hasattr(localizer, 'cleanup'):
            localizer.cleanup()
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        if 'localizer' in locals() and hasattr(localizer, 'cleanup'):
            localizer.cleanup()
        raise


if __name__ == "__main__":
    main()


