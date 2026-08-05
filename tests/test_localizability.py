import pytest

from dataset.models import BugInstance
from dataset.localizability import (
    classify_ground_truth_path,
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

PATCH = """diff --git a/pkg/foo.py b/pkg/foo.py
index abc..def 100644
--- a/pkg/foo.py
+++ b/pkg/foo.py
@@ -1,1 +1,1 @@
-old
+new
diff --git a/pkg/new_file.py b/pkg/new_file.py
new file mode 100644
index 000..abc
--- /dev/null
+++ b/pkg/new_file.py
@@ -0,0 +1,1 @@
+hello
diff --git a/pkg/removed.py b/pkg/removed.py
deleted file mode 100644
index abc..000
--- a/pkg/removed.py
+++ /dev/null
@@ -1,1 +0,0 @@
-bye
"""


def make_bug(**overrides):
    defaults = dict(
        repo="foo/bar",
        instance_id="1",
        base_commit="deadbeef",
        patch=PATCH,
        hints_text="",
        ground_truths=["pkg/foo.py", "pkg/new_file.py", "pkg/removed.py", "pkg/unknown.py"],
        bug_report="",
        code_files=["pkg/foo.py", "pkg/removed.py", "pkg/other.py"],
    )
    defaults.update(overrides)
    return BugInstance(**defaults)


def test_classify_bug_instance_all_classes():
    bug = make_bug()
    classifications = classify_bug_instance(bug)

    assert classifications["pkg/foo.py"] == EXISTS_BEFORE_FIX
    assert classifications["pkg/new_file.py"] == ADDED_BY_FIX
    assert classifications["pkg/removed.py"] == DELETED_BY_FIX
    assert classifications["pkg/unknown.py"] == MISSING_UNRESOLVED


def test_classify_falls_back_to_after_commit_when_patch_has_no_real_hunk():
    # BeetleBox-style synthetic patch text has no diff --git headers at all.
    bug = make_bug(
        patch="Before: deadbeef\nAfter: cafebabe",
        ground_truths=["pkg/mystery.py"],
        code_files=[],
        after_commit="cafebabe",
        repo="this-repo-does-not-exist-xyz/nope",
    )
    result = classify_ground_truth_path(bug, "pkg/mystery.py", token=None, cache=None)
    assert result == API_ERROR


def test_classify_missing_unresolved_when_no_after_commit_and_no_hunk():
    bug = make_bug(
        patch="Before: deadbeef\nAfter: ",
        ground_truths=["pkg/mystery.py"],
        code_files=[],
        after_commit=None,
    )
    result = classify_ground_truth_path(bug, "pkg/mystery.py", token=None, cache=None)
    assert result == MISSING_UNRESOLVED


def test_api_error_is_never_cached():
    bug = make_bug(
        patch="Before: deadbeef\nAfter: cafebabe",
        ground_truths=["pkg/mystery.py"],
        code_files=[],
        after_commit="cafebabe",
        repo="this-repo-does-not-exist-xyz/nope",
    )
    cache = {}
    result = classify_ground_truth_path(bug, "pkg/mystery.py", token=None, cache=cache)
    assert result == API_ERROR
    assert cache == {}


def test_non_error_classification_is_cached_and_reused():
    bug = make_bug()
    cache = {}
    first = classify_ground_truth_path(bug, "pkg/foo.py", token=None, cache=cache)
    assert first == EXISTS_BEFORE_FIX
    assert len(cache) == 1

    # Mutate the bug so a fresh (uncached) classification would give a different answer;
    # the cached result should be returned unchanged, proving the cache was actually used.
    bug.code_files = []
    second = classify_ground_truth_path(bug, "pkg/foo.py", token=None, cache=cache)
    assert second == EXISTS_BEFORE_FIX


def test_cache_save_and_load_round_trip(tmp_path):
    cache_path = str(tmp_path / "localizability_cache.json")
    cache = load_cache(cache_path)
    assert cache == {}

    cache["foo/bar@sha:pkg/foo.py"] = EXISTS_BEFORE_FIX
    save_cache(cache, cache_path)

    reloaded = load_cache(cache_path)
    assert reloaded == cache


@pytest.mark.parametrize(
    "classifications,expected",
    [
        (
            {"a": EXISTS_BEFORE_FIX, "b": DELETED_BY_FIX, "c": ADDED_BY_FIX, "d": MISSING_UNRESOLVED},
            {"raw_coverage": 0.5, "available_corpus_coverage": 0.5, "localizable_coverage": 2 / 3},
        ),
        (
            {"a": EXISTS_BEFORE_FIX},
            {"raw_coverage": 1.0, "available_corpus_coverage": 1.0, "localizable_coverage": 1.0},
        ),
        (
            {"a": ADDED_BY_FIX},
            {"raw_coverage": 0.0, "available_corpus_coverage": 0.0, "localizable_coverage": 0.0},
        ),
        (
            {},
            {"raw_coverage": 0.0, "available_corpus_coverage": 0.0, "localizable_coverage": 0.0},
        ),
    ],
)
def test_compute_coverage(classifications, expected):
    coverage = compute_coverage(classifications)
    assert coverage == pytest.approx(expected)


def test_compute_coverage_excludes_api_error_from_available_corpus_denominator():
    classifications = {"a": EXISTS_BEFORE_FIX, "b": API_ERROR}
    coverage = compute_coverage(classifications)
    # raw_coverage counts API_ERROR in the denominator (it's still a "dataset GT")
    assert coverage["raw_coverage"] == pytest.approx(0.5)
    # available_corpus_coverage excludes it, since it was never actually resolved
    assert coverage["available_corpus_coverage"] == pytest.approx(1.0)
