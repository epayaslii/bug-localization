from dataset.models import BugInstance
from method.bm25_retriever import (
    _tokenize_path,
    _tokenize_query,
    _extract_skeleton_tokens,
    _extract_symbol_tokens,
    rank_files_bm25,
    rank_files_bm25_with_skeleton,
    rank_files_bm25_with_symbols,
)

SAMPLE_SOURCE = """
import os
from collections import Counter as MyCounter

class ShapeUtil:
    \"\"\"Utilities for cartesian shape handling.\"\"\"

    def compute_area(self):
        pass

def cartesian_transform(x, y):
    return x, y
"""


def make_bug(code_files, bug_report="fix the bug", **overrides):
    defaults = dict(
        repo="org/repo",
        instance_id="1",
        base_commit="sha",
        patch="",
        hints_text="",
        ground_truths=[],
        bug_report=bug_report,
        code_files=code_files,
    )
    defaults.update(overrides)
    return BugInstance(**defaults)


def test_tokenize_path_splits_separators_and_camel_case():
    assert _tokenize_path("pkg/CartesianShapeQueryTests.java") == [
        "pkg", "cartesian", "shape", "query", "tests", "java",
    ]


def test_tokenize_query_extracts_word_tokens():
    assert _tokenize_query("Uploading a file removes trailing_newline!") == [
        "uploading", "a", "file", "removes", "trailing_newline",
    ]


def test_extract_skeleton_tokens_includes_docstring_and_names():
    tokens = _extract_skeleton_tokens(SAMPLE_SOURCE)
    assert "cartesian" in tokens  # from the docstring
    assert "shape" in tokens and "util" in tokens  # from ShapeUtil, split like a path
    assert "compute" in tokens and "area" in tokens
    assert "transform" in tokens


def test_extract_skeleton_tokens_returns_empty_on_syntax_error():
    assert _extract_skeleton_tokens("def broken(:\n") == []


def test_extract_symbol_tokens_separates_symbols_from_imports():
    symbol_tokens, import_tokens = _extract_symbol_tokens(SAMPLE_SOURCE)

    assert "shape" in symbol_tokens and "util" in symbol_tokens
    assert "compute" in symbol_tokens and "area" in symbol_tokens
    assert "cartesian" in symbol_tokens and "transform" in symbol_tokens

    assert "os" in import_tokens
    assert "collections" in import_tokens
    assert "counter" in import_tokens
    # docstring text should NOT leak into either -- unlike the skeleton variant
    assert "utilities" not in symbol_tokens and "utilities" not in import_tokens


def test_extract_symbol_tokens_returns_empty_pair_on_syntax_error():
    assert _extract_symbol_tokens("def broken(:\n") == ([], [])


def test_rank_files_bm25_truncates_to_top_k():
    bug_report = "fix cartesian shape transform bug"
    files = ["cartesian_shape.py", "unrelated_a.py", "unrelated_b.py", "unrelated_c.py"]
    result = rank_files_bm25(bug_report, files, top_k=2)
    assert len(result) == 2
    assert result[0] == "cartesian_shape.py"


def test_rank_files_bm25_returns_input_unchanged_when_within_top_k():
    files = ["a.py", "b.py"]
    result = rank_files_bm25("anything", files, top_k=10)
    assert result == files


def test_rank_files_bm25_top_k_none_returns_full_ranking():
    bug_report = "fix cartesian shape transform bug"
    files = ["cartesian_shape.py", "unrelated_a.py", "unrelated_b.py"]
    result = rank_files_bm25(bug_report, files, top_k=None)
    assert len(result) == len(files)
    assert set(result) == set(files)
    assert result[0] == "cartesian_shape.py"


def test_rank_files_bm25_with_symbols_falls_back_to_path_tokens_when_repo_not_cached():
    # Repo isn't mirrored locally, so no content can be fetched -- should behave like
    # path-only ranking rather than erroring.
    bug = make_bug(
        code_files=["cartesian_shape.py", "a.py", "b.py"],
        bug_report="fix cartesian shape transform bug",
        repo="this-repo-is-definitely-not-mirrored-xyz/nope",
    )
    result = rank_files_bm25_with_symbols(bug, top_k=1)
    assert result == ["cartesian_shape.py"]


def test_rank_files_bm25_with_skeleton_falls_back_to_path_tokens_when_repo_not_cached():
    bug = make_bug(
        code_files=["cartesian_shape.py", "a.py", "b.py"],
        bug_report="fix cartesian shape transform bug",
        repo="this-repo-is-definitely-not-mirrored-xyz/nope",
    )
    result = rank_files_bm25_with_skeleton(bug, top_k=1)
    assert result == ["cartesian_shape.py"]


def test_rank_files_bm25_with_symbols_returns_input_unchanged_when_within_top_k():
    bug = make_bug(code_files=["a.py", "b.py"])
    result = rank_files_bm25_with_symbols(bug, top_k=10)
    assert result == ["a.py", "b.py"]
