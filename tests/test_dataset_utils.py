import pytest

from dataset.utils import (
    is_code_file,
    filter_code_paths,
    get_token_count,
    chunk_code_files,
    get_diff_hunk,
    parse_line_ranges,
)
from dataset.models import LineRange


def test_is_code_file_accepts_matching_extension():
    assert is_code_file("pkg/foo.py", (".py",)) is True


def test_is_code_file_rejects_non_matching_extension():
    assert is_code_file("pkg/foo.txt", (".py",)) is False


def test_is_code_file_accepts_single_string_extension():
    assert is_code_file("pkg/foo.py", ".py") is True


def test_is_code_file_rejects_non_string_input():
    assert is_code_file(None, (".py",)) is False
    assert is_code_file(123, (".py",)) is False


def test_filter_code_paths_keeps_only_matching_extensions():
    paths = ["a.py", "b.java", "c.py", "README.md"]
    assert filter_code_paths(paths, (".py",)) == ["a.py", "c.py"]


_SAMPLE_PATCH = """diff --git a/pkg/foo.py b/pkg/foo.py
--- a/pkg/foo.py
+++ b/pkg/foo.py
@@ -10,3 +10,4 @@ def foo():
     pass
+    pass
diff --git a/pkg/bar.py b/pkg/bar.py
--- a/pkg/bar.py
+++ b/pkg/bar.py
@@ -1,2 +1,2 @@
-old
+new
@@ -20,5 +20,6 @@ def bar():
     pass
"""


def test_get_diff_hunk_returns_matching_files_hunk_only():
    hunk = get_diff_hunk(_SAMPLE_PATCH, "pkg/foo.py")
    assert "@@ -10,3 +10,4 @@" in hunk
    assert "pkg/bar.py" not in hunk


def test_get_diff_hunk_returns_none_for_path_not_in_patch():
    assert get_diff_hunk(_SAMPLE_PATCH, "pkg/missing.py") is None


def test_get_diff_hunk_returns_none_for_synthetic_beetlebox_patch():
    assert get_diff_hunk("Before: abc123\nAfter: def456", "pkg/foo.py") is None


def test_parse_line_ranges_single_hunk():
    hunk = get_diff_hunk(_SAMPLE_PATCH, "pkg/foo.py")
    ranges = parse_line_ranges(hunk)
    assert ranges == [LineRange(old_start=10, old_lines=3, new_start=10, new_lines=4)]


def test_parse_line_ranges_multiple_hunks_same_file():
    hunk = get_diff_hunk(_SAMPLE_PATCH, "pkg/bar.py")
    ranges = parse_line_ranges(hunk)
    assert ranges == [
        LineRange(old_start=1, old_lines=2, new_start=1, new_lines=2),
        LineRange(old_start=20, old_lines=5, new_start=20, new_lines=6),
    ]


def test_parse_line_ranges_missing_count_implies_one_line():
    # unified diff omits the count when it's 1: "@@ -5 +5 @@" means a single-line range
    ranges = parse_line_ranges("@@ -5 +5 @@ def f():\n-old\n+new\n")
    assert ranges == [LineRange(old_start=5, old_lines=1, new_start=5, new_lines=1)]


def test_parse_line_ranges_empty_hunk_returns_empty_list():
    assert parse_line_ranges("") == []


def test_filter_code_paths_empty_input():
    assert filter_code_paths([], (".py",)) == []
    assert filter_code_paths(None, (".py",)) == []


def test_get_token_count_returns_positive_int_for_nonempty_text():
    count = get_token_count("hello world, this is a bug report", model="gpt-4")
    assert isinstance(count, int)
    assert count > 0


def test_get_token_count_zero_for_empty_text():
    assert get_token_count("", model="gpt-4") == 0


def test_get_token_count_longer_text_has_more_tokens():
    short = get_token_count("short text", model="gpt-4")
    long = get_token_count("short text " * 50, model="gpt-4")
    assert long > short


def test_chunk_code_files_splits_by_token_budget():
    # Each file's own text is short, so many should share a chunk under a small budget.
    files = [f"file_{i}.py" for i in range(20)]
    chunks = chunk_code_files(files, max_chunk_tokens=10, model="gpt-4")
    assert len(chunks) >= 1
    # every input file should appear somewhere across the chunks
    flattened = [f for chunk in chunks for f in chunk]
    assert sorted(flattened) == sorted(files)


def test_chunk_code_files_empty_input():
    assert chunk_code_files([], max_chunk_tokens=100) == []
