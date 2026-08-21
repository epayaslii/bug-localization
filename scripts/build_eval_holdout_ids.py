"""Union the instance IDs of every Bench4BL manifest that has produced a reported eval
number into one holdout-IDs artifact -- pass this to Bench4BL.get_bug_instances(
exclude_instance_ids=...) before sampling any training/dev pool, so classifier training
data and prompt-optimization dev sets can never leak instances that later get evaluated
against.
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.manifest import load_manifest, DEFAULT_MANIFEST_DIR
from dataset.utils import setup_logging, get_logger

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

# Every committed Bench4BL manifest that has ever been used to report a real number in
# this project -- including the superseded old-skewed n=30 manifest, since it did produce
# reported numbers historically (better to over-exclude than leak).
DEFAULT_MANIFESTS = [
    "bench4bl-multi-n30-s42-8c4f91c33f21.json",
    "bench4bl-multi-n30-s42-9449c3b8a675.json",
    "bench4bl-multi-n30-s42-mn5-8proj.json",
    "bench4bl-multi-n200-s42.json",
    "bench4bl-multi-n1000-s42-f82f31ccff6f.json",
]

DEFAULT_OUTPUT = os.path.join(DEFAULT_MANIFEST_DIR, "bench4bl_eval_holdout_ids.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifests', nargs='+', default=DEFAULT_MANIFESTS,
        help='Manifest filenames (resolved under results/manifests/) or absolute paths to union.',
    )
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    instance_ids = set()
    source_manifests = []
    for name in args.manifests:
        path = name if os.path.isabs(name) else os.path.join(DEFAULT_MANIFEST_DIR, name)
        manifest = load_manifest(path)
        ids = [inst["instance_id"] for inst in manifest["instances"]]
        instance_ids.update(ids)
        source_manifests.append({"manifest_id": manifest["manifest_id"], "path": path, "size": len(ids)})
        logger.info(f"{manifest['manifest_id']}: {len(ids)} instances")

    holdout = {
        "source_manifests": source_manifests,
        "count": len(instance_ids),
        "instance_ids": sorted(instance_ids),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(holdout, f, indent=2)

    logger.info(f"Union: {len(instance_ids)} unique instance IDs across {len(source_manifests)} manifests")
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
