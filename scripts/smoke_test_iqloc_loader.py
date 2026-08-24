"""Smoke-tests dataset/iqloc.py's IQLocExtended loader against Bench4BLExtended.json,
without needing the full retrieval pipeline. Reports how many of the JSON's bug records
resolve to a usable BugInstance vs. why they don't, so you know whether you need to mirror
more bench4bl_cache/<sub_project> repos before wiring this dataset into the hybrid RRF/BM25
pipeline.

Usage:
    python scripts/smoke_test_iqloc_loader.py --dataset-json Dataset/Bench4BLExtended.json
    python scripts/smoke_test_iqloc_loader.py --dataset-json ... --original-only
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.iqloc import IQLocExtended
from dataset.bench4bl import DEFAULT_CACHE_DIR
from dataset.utils import setup_logging, get_logger

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def diagnose(records, cache_dir, loader):
    """Re-runs the same resolution steps as IQLocExtended._parse_record, record by record,
    to bucket failures by reason -- the loader itself only returns None on failure (same
    contract as Bench4BL._parse_bug), which is fine for the pipeline but not informative
    enough to know what to fix.
    """
    reasons = Counter()
    missing_mirror_subprojects = defaultdict(int)

    for rec in records:
        sub_project = rec["sub_project"]
        proot = os.path.join(cache_dir, sub_project)
        gitrepo = os.path.join(proot, "gitrepo")

        if not os.path.isdir(gitrepo):
            reasons["no_local_mirror"] += 1
            missing_mirror_subprojects[sub_project] += 1
            continue

        tag = rec["version"]  # IQLoc's version field is already the literal git tag

        code_files = loader._list_files_at_commit(gitrepo, tag)
        if not code_files:
            reasons["no_code_files_at_tag"] += 1
            continue

        ground_truths = [
            loader._resolve_dotted_path(dotted, code_files)
            for dotted in rec.get("fixed_files", [])
        ]
        ground_truths = [g for g in ground_truths if g]
        if not ground_truths:
            reasons["no_ground_truths_resolved"] += 1
            continue

        reasons["resolved"] += 1

    return reasons, missing_mirror_subprojects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", required=True, help="Path to Bench4BLExtended.json")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--original-only", action="store_true",
                         help="Only diagnose label==1 (original Bench4BL) records, skip IQLoc's extension bugs")
    args = parser.parse_args()

    with open(args.dataset_json) as f:
        records = json.load(f)
    if args.original_only:
        records = [r for r in records if r.get("label") == 1]

    logger.info(f"Diagnosing {len(records)} records against cache_dir={args.cache_dir}")

    # Reuse IQLocExtended purely for its _list_files_at_commit/_resolve_dotted_path helpers
    # (inherited from Bench4BL); the actual per-record diagnosis below re-derives the same
    # checks so we can bucket *why* a record failed, not just whether it did.
    loader = IQLocExtended(args.dataset_json, cache_dir=args.cache_dir, include_extension=not args.original_only)

    reasons, missing_mirror_subprojects = diagnose(records, args.cache_dir, loader)

    total = sum(reasons.values())
    logger.info("=== Resolution breakdown ===")
    for reason, count in reasons.most_common():
        pct = count / total * 100 if total else 0.0
        logger.info(f"  {reason}: {count} ({pct:.1f}%)")

    logger.info(f"Loader actually produced {len(loader.get_bug_instances())} BugInstance objects "
                f"(cross-check against 'resolved' count above)")

    if missing_mirror_subprojects:
        logger.info("=== sub_projects with no local bench4bl_cache mirror ===")
        for sub_project, count in sorted(missing_mirror_subprojects.items(), key=lambda x: -x[1]):
            logger.info(f"  {sub_project}: {count} records blocked")


if __name__ == "__main__":
    main()
