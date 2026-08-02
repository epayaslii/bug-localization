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
    
    args = parser.parse_args()
    
    logger.info("Starting...")
    logger.info(f"Method: {args.method}, Dataset: {args.dataset}, Model: {args.model}")
    
    try:
        if args.dataset == 'swebench':
            instance = SWEBench()
        else:
            instance = BeetleBox()
        
        if args.method == 'openai':
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


