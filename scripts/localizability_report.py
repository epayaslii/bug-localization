import os
import sys
import json
import argparse
import logging
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from dataset.swebench import SWEBench
from dataset.beetlebox import BeetleBox
from dataset.localizability import (
    classify_bug_instance,
    compute_coverage,
    load_cache,
    save_cache,
    EXISTS_BEFORE_FIX,
    DELETED_BY_FIX,
    ADDED_BY_FIX,
    MISSING_UNRESOLVED,
    API_ERROR,
)
from dataset.utils import setup_logging, get_logger

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

CLASS_ORDER = (EXISTS_BEFORE_FIX, DELETED_BY_FIX, ADDED_BY_FIX, MISSING_UNRESOLVED, API_ERROR)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Classify each bug instance's ground-truth files as localizable / "
                    "added-by-fix / deleted-by-fix / unresolved, and report corpus coverage."
    )
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox'], default='swebench')
    parser.add_argument('--sample-size', type=int, default=30)
    parser.add_argument('--output', default=None, help='Optional path to write the full JSON report')
    args = parser.parse_args()

    token = os.getenv("GITHUB_TOKEN")
    instance = SWEBench() if args.dataset == 'swebench' else BeetleBox()
    bugs = instance.get_bug_instances(sample_size=args.sample_size, random_sample=True, random_seed=42)
    logger.info(f"Classifying ground-truth files for {len(bugs)} bug instances...")

    cache = load_cache()
    class_counts = Counter()
    per_bug = {}

    for i, bug in enumerate(bugs):
        classifications = classify_bug_instance(bug, token=token, cache=cache)
        coverage = compute_coverage(classifications)
        per_bug[bug.instance_id] = {
            "repo": bug.repo,
            "classifications": classifications,
            "coverage": coverage,
        }
        class_counts.update(classifications.values())
        if (i + 1) % 10 == 0:
            logger.info(f"Classified {i + 1}/{len(bugs)} instances")

    save_cache(cache)

    total_gt = sum(class_counts.values())
    logger.info("=== Ground-truth classification summary ===")
    for cls in CLASS_ORDER:
        count = class_counts.get(cls, 0)
        pct = (count / total_gt * 100) if total_gt else 0.0
        logger.info(f"  {cls}: {count} ({pct:.1f}%)")

    n = len(per_bug) or 1
    mean_raw = sum(b["coverage"]["raw_coverage"] for b in per_bug.values()) / n
    mean_available = sum(b["coverage"]["available_corpus_coverage"] for b in per_bug.values()) / n
    mean_localizable = sum(b["coverage"]["localizable_coverage"] for b in per_bug.values()) / n
    logger.info(f"Mean raw coverage: {mean_raw:.3f}")
    logger.info(f"Mean available-corpus coverage: {mean_available:.3f}")
    logger.info(f"Mean localizable coverage: {mean_localizable:.3f}")

    if args.output:
        report = {
            "dataset": args.dataset,
            "sample_size": len(bugs),
            "classification_counts": dict(class_counts),
            "mean_raw_coverage": mean_raw,
            "mean_available_corpus_coverage": mean_available,
            "mean_localizable_coverage": mean_localizable,
            "per_bug": per_bug,
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Wrote full report to {args.output}")


if __name__ == "__main__":
    main()
