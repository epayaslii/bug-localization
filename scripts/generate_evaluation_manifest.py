import os
import sys
import argparse
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.swebench import SWEBench
from dataset.beetlebox import BeetleBox
from dataset.bench4bl import Bench4BL
from dataset.iqloc import IQLocExtended
from evaluation.manifest import build_manifest, save_manifest
from dataset.utils import setup_logging, get_logger

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a deterministic, diversity-constrained evaluation manifest "
                    "(a stable seeded sample so retrieval/reranking comparisons are apples-to-apples)."
    )
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox', 'bench4bl', 'iqloc'], default='swebench')
    parser.add_argument('--size', type=int, default=24)
    parser.add_argument('--pool-size', type=int, default=100,
                       help='How many instances to sample+process before diversity selection (must be >= --size)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-per-repo', type=int, default=2)
    parser.add_argument('--min-distinct-repos', type=int, default=None)
    parser.add_argument('--output', default=None, help='Defaults to results/manifests/<manifest_id>.json')
    args = parser.parse_args()

    if args.pool_size < args.size:
        parser.error("--pool-size must be >= --size")

    instance = {'swebench': SWEBench, 'beetlebox': BeetleBox, 'bench4bl': Bench4BL, 'iqloc': IQLocExtended}[args.dataset]()
    pool = instance.get_bug_instances(sample_size=args.pool_size, random_sample=True, random_seed=args.seed)
    logger.info(f"Pulled a pool of {len(pool)} instances across {len(set(b.repo for b in pool))} repos")

    manifest = build_manifest(
        args.dataset, pool, args.size,
        seed=args.seed, max_per_repo=args.max_per_repo,
        min_distinct_repos=args.min_distinct_repos, pool_size=args.pool_size,
    )
    path = save_manifest(manifest, args.output)
    logger.info(f"Manifest {manifest['manifest_id']}: {manifest['size']} instances, {manifest['distinct_repos']} distinct repos")
    logger.info(f"Saved to {path}")


if __name__ == "__main__":
    main()
