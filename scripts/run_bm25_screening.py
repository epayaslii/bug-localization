import os
import sys
import argparse
import json
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from dataset.swebench import SWEBench
from dataset.beetlebox import BeetleBox
from evaluation.manifest import load_manifest
from evaluation.screening import screen_manifest
from dataset.localizability import load_cache, save_cache
from dataset.utils import setup_logging, get_logger

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run path-only BM25 screening over a saved evaluation manifest, "
                    "reporting the best ground-truth rank and difficulty band per instance."
    )
    parser.add_argument('--manifest', required=True,
                       help='Path to a manifest JSON produced by generate_evaluation_manifest.py')
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox'], default=None,
                       help='Overrides the dataset recorded in the manifest, if needed')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    dataset_name = args.dataset or manifest['dataset']
    instance = SWEBench() if dataset_name == 'swebench' else BeetleBox()

    # Re-derive the exact same pool the manifest was built from (same sample_size/seed),
    # instead of processing the whole dataset, so this stays cheap.
    pool_size = manifest.get('pool_size') or manifest['size']
    pool = instance.get_bug_instances(sample_size=pool_size, random_sample=True, random_seed=manifest['seed'])

    wanted = {inst['instance_id']: inst for inst in manifest['instances']}
    bugs = [b for b in pool if b.instance_id in wanted]

    missing = set(wanted) - {b.instance_id for b in bugs}
    if missing:
        logger.warning(
            f"{len(missing)} manifest instance(s) not found when re-deriving the pool "
            f"(dataset ordering may have changed): {sorted(missing)[:5]}"
        )

    for bug in bugs:
        expected_repo = wanted[bug.instance_id]['repo']
        if bug.repo != expected_repo:
            logger.warning(f"Repo mismatch for {bug.instance_id}: manifest says {expected_repo}, got {bug.repo}")

    logger.info(f"Screening {len(bugs)}/{manifest['size']} manifest instances (manifest {manifest['manifest_id']})")

    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()
    report = screen_manifest(bugs, token=token, cache=cache)
    save_cache(cache)

    logger.info(f"=== BM25 screening: manifest {manifest['manifest_id']} ({len(bugs)} instances) ===")
    for band, count in report["difficulty_distribution"].items():
        logger.info(f"  {band}: {count}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"manifest_id": manifest["manifest_id"], **report}, f, indent=2)
        logger.info(f"Wrote full report to {args.output}")


if __name__ == "__main__":
    main()
