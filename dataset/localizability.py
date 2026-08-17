"""Ground-truth localizability diagnostics.

Separates "retrieval failure" from "this ground-truth file could never have been
retrieved" by classifying each ground-truth path against the before-fix corpus that
retrieval actually searches (`bug.code_files`, built from `bug.base_commit`).

A ground-truth path introduced by the fixing commit itself did not exist in the
before-fix repository snapshot, so no retrieval method -- however good -- can find it.
Counting such a "miss" as a retrieval failure understates real performance.
"""

import os
import json

from dataset.utils import get_logger, get_code_files, get_diff_hunk

logger = get_logger(__name__)

EXISTS_BEFORE_FIX = "exists_before_fix"
DELETED_BY_FIX = "deleted_by_fix"
ADDED_BY_FIX = "added_by_fix"
MISSING_UNRESOLVED = "missing_unresolved"
API_ERROR = "api_error"

# Present in the before-fix corpus -> a valid retrieval target, regardless of whether
# the fix later deletes it.
LOCALIZABLE_CLASSES = {EXISTS_BEFORE_FIX, DELETED_BY_FIX}

DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "repo_cache", "localizability_cache.json"
)


def load_cache(cache_path=None):
    path = cache_path or DEFAULT_CACHE_PATH
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read localizability cache at {path}, starting fresh: {e}")
    return {}


def save_cache(cache, cache_path=None):
    path = cache_path or DEFAULT_CACHE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def _cache_key(repo, base_commit, path):
    return f"{repo}@{base_commit}:{path}"


def classify_ground_truth_path(bug, path, token=None, cache=None):
    """Classify one ground-truth `path` for `bug` against the before-fix corpus.

    Returns one of EXISTS_BEFORE_FIX, DELETED_BY_FIX, ADDED_BY_FIX, MISSING_UNRESOLVED,
    API_ERROR. Results are cached on disk keyed by (repo, base_commit, path);
    API_ERROR is never cached, so a transient failure is retried on the next run
    instead of being remembered as a final answer.
    """
    key = _cache_key(bug.repo, bug.base_commit, path)
    if cache is not None and key in cache:
        return cache[key]

    if path in bug.code_files:
        hunk = get_diff_hunk(bug.patch, path)
        classification = DELETED_BY_FIX if hunk and "deleted file mode" in hunk else EXISTS_BEFORE_FIX
        if cache is not None:
            cache[key] = classification
        return classification

    # Not in the before-fix corpus. Prefer the patch's own diff header over a network
    # call, when it has a real hunk for this path (SWE-bench patches do; BeetleBox's
    # synthetic "Before/After" patch text does not).
    hunk = get_diff_hunk(bug.patch, path)
    if hunk is not None:
        classification = ADDED_BY_FIX if "new file mode" in hunk else MISSING_UNRESOLVED
        if cache is not None:
            cache[key] = classification
        return classification

    if not bug.after_commit:
        return MISSING_UNRESOLVED

    try:
        extensions = (os.path.splitext(path)[1],)
        after_files = get_code_files(bug.repo, bug.after_commit, extensions, token)
    except Exception as e:
        logger.warning(f"API error resolving {path} @ {bug.repo}#{bug.after_commit}: {e}")
        return API_ERROR

    classification = ADDED_BY_FIX if path in after_files else MISSING_UNRESOLVED
    if cache is not None:
        cache[key] = classification
    return classification


def classify_bug_instance(bug, token=None, cache=None) -> dict[str, str]:
    """Classify every ground-truth path for `bug`. Returns {path: classification}."""
    return {path: classify_ground_truth_path(bug, path, token=token, cache=cache) for path in bug.ground_truths}


def compute_coverage(classifications: dict[str, str]) -> dict[str, float]:
    """Compute the three coverage notions over one bug's ground-truth classifications:

    - raw_coverage: localizable GTs / all GTs.
    - available_corpus_coverage: localizable GTs / GTs that were actually resolved
      (excludes API_ERROR, since those are unknown rather than unlocalizable).
    - localizable_coverage: localizable GTs / GTs that are not confirmed added-by-fix
      or unresolved-due-to-error (i.e. the denominator excludes GTs that could never
      have been localization targets in the first place).
    """
    total = len(classifications)
    if total == 0:
        return {"raw_coverage": 0.0, "available_corpus_coverage": 0.0, "localizable_coverage": 0.0}

    localizable_count = sum(1 for c in classifications.values() if c in LOCALIZABLE_CLASSES)
    resolved = sum(1 for c in classifications.values() if c != API_ERROR)
    denom_localizable = sum(1 for c in classifications.values() if c not in (ADDED_BY_FIX, API_ERROR))

    return {
        "raw_coverage": localizable_count / total,
        "available_corpus_coverage": (localizable_count / resolved) if resolved else 0.0,
        "localizable_coverage": (localizable_count / denom_localizable) if denom_localizable else 0.0,
    }
