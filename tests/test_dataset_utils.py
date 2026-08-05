import pytest

from dataset.utils import (
    is_code_file,
    filter_code_paths,
    get_token_count,
    chunk_code_files,
)


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
