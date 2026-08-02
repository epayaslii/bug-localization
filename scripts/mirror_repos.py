import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.swebench import SWEBench
from dataset.beetlebox import BeetleBox
from dataset.repo_cache import mirror_repo, is_repo_cached
from dataset.utils import setup_logging, get_logger
import logging

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Mirror the repos referenced by a dataset sample into the local git cache")
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox'], default='beetlebox')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Only mirror repos touched by this many sampled bug instances (default: whole dataset)')
    args = parser.parse_args()

    instance = SWEBench() if args.dataset == 'swebench' else BeetleBox()

    if args.sample_size is not None:
        bugs = instance.get_bug_instances(sample_size=args.sample_size, random_sample=True, random_seed=42)
    else:
        bugs = instance.get_bug_instances()

    repos = sorted(set(bug.repo for bug in bugs if bug.repo))
    logger.info(f"{len(repos)} unique repos to mirror")

    for i, repo in enumerate(repos):
        if is_repo_cached(repo):
            logger.info(f"[{i+1}/{len(repos)}] already cached, refreshing: {repo}")
        else:
            logger.info(f"[{i+1}/{len(repos)}] cloning: {repo}")
        try:
            mirror_repo(repo)
        except Exception as e:
            logger.error(f"Failed to mirror {repo}: {e}")

    logger.info("Done.")


if __name__ == "__main__":
    main()