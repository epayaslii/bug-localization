"""Deterministic, diversity-constrained evaluation manifests.

A manifest is a stable, seeded sample of bug instances used to compare retrieval/
reranking approaches against each other on identical data, instead of ad-hoc
single-run sampling where every comparison is drawn from a different set of
instances. `max_per_repo` prevents one large or easy repository from dominating
the signal.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
import random

from dataset.utils import get_logger

logger = get_logger(__name__)

DEFAULT_MANIFEST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "manifests"
)


def _manifest_id(dataset_name, size, seed, max_per_repo, instance_ids):
    digest_input = f"{dataset_name}|{size}|{seed}|{max_per_repo}|{','.join(sorted(instance_ids))}"
    digest = hashlib.sha256(digest_input.encode()).hexdigest()[:12]
    return f"{dataset_name}-multi-n{size}-s{seed}-{digest}"


def select_diverse_manifest(bugs, size, seed=42, max_per_repo=2, min_distinct_repos=None):
    """Deterministically select `size` bug instances from `bugs`, capping how many can
    come from any one repo (`max_per_repo`).

    `bugs` must already be materialized BugInstance objects (e.g. pulled from a larger
    seeded pool via a dataset loader) -- this function only performs the diversity-
    constrained selection, it does not fetch or process data itself.
    """
    if size > len(bugs):
        raise ValueError(f"Requested manifest size {size} exceeds pool size {len(bugs)}")

    rng = random.Random(seed)
    shuffled = bugs[:]
    rng.shuffle(shuffled)

    per_repo_count = {}
    selected = []
    for bug in shuffled:
        if len(selected) >= size:
            break
        count = per_repo_count.get(bug.repo, 0)
        if count >= max_per_repo:
            continue
        selected.append(bug)
        per_repo_count[bug.repo] = count + 1

    if len(selected) < size:
        logger.warning(
            f"Diversity constraint (max {max_per_repo}/repo) left only {len(selected)}/{size} "
            f"instances selectable from a pool of {len(bugs)}; consider a larger pool."
        )

    distinct_repos = len(set(b.repo for b in selected))
    if min_distinct_repos is not None and distinct_repos < min_distinct_repos:
        logger.warning(
            f"Only {distinct_repos} distinct repos in the manifest, below the requested "
            f"min_distinct_repos={min_distinct_repos}."
        )

    return selected


def build_manifest(dataset_name, bugs, size, seed=42, max_per_repo=2, min_distinct_repos=None, pool_size=None):
    """Build a manifest dict from an already-selected diverse subset of `bugs`.

    `pool_size` (if given) is recorded so the exact same pool can be re-derived later
    (e.g. by a screening script) via the same dataset loader call
    (sample_size=pool_size, random_sample=True, random_seed=seed) without re-processing
    the whole dataset.
    """
    selected = select_diverse_manifest(
        bugs, size, seed=seed, max_per_repo=max_per_repo, min_distinct_repos=min_distinct_repos
    )
    instance_ids = [b.instance_id for b in selected]
    manifest_id = _manifest_id(dataset_name, size, seed, max_per_repo, instance_ids)

    return {
        "manifest_id": manifest_id,
        "dataset": dataset_name,
        "size": len(selected),
        "requested_size": size,
        "seed": seed,
        "max_per_repo": max_per_repo,
        "min_distinct_repos": min_distinct_repos,
        "pool_size": pool_size,
        "distinct_repos": len(set(b.repo for b in selected)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "instances": [
            {"instance_id": b.instance_id, "repo": b.repo, "base_commit": b.base_commit}
            for b in selected
        ],
    }


def save_manifest(manifest, path=None):
    path = path or os.path.join(DEFAULT_MANIFEST_DIR, f"{manifest['manifest_id']}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def load_manifest(path):
    with open(path) as f:
        return json.load(f)
