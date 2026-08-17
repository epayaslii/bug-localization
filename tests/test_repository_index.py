import json
import os

import numpy as np
import pytest

from dataset.models import BugInstance
import method.repository_index as repository_index
from method.repository_index import (
    extract_symbols_and_imports,
    _language_for_path,
    _index_paths,
    is_indexed,
    build_repository_index,
    load_repository_index,
    rank_files_from_index,
)

SAMPLE_SOURCE = """import os
from collections import Counter

class ShapeUtil:
    \"\"\"Utilities.\"\"\"

    def compute_area(self):
        return 42

def cartesian_transform(x, y):
    return x, y
"""


def make_bug(repo, base_commit, code_files, bug_report="fix the bug", **overrides):
    defaults = dict(
        repo=repo, instance_id="1", base_commit=base_commit, patch="",
        hints_text="", ground_truths=[], bug_report=bug_report, code_files=code_files,
    )
    defaults.update(overrides)
    return BugInstance(**defaults)


def test_extract_symbols_and_imports_real_source():
    # ast.walk() is breadth-first: top-level siblings (ShapeUtil, cartesian_transform)
    # come before descending into ShapeUtil's body (compute_area).
    symbols, imports = extract_symbols_and_imports(SAMPLE_SOURCE)
    assert symbols == ["ShapeUtil", "cartesian_transform", "compute_area"]
    assert imports == ["os", "collections"]


def test_extract_symbols_and_imports_syntax_error_returns_empty():
    assert extract_symbols_and_imports("def broken(:\n") == ([], [])


def test_language_for_path():
    assert _language_for_path("pkg/foo.py") == "python"
    assert _language_for_path("pkg/Foo.java") == "java"
    assert _language_for_path("pkg/foo.unknown_ext") == "unknown"


def test_index_paths_replaces_slashes_and_includes_model(tmp_path):
    index_path, meta_path = _index_paths("org/repo", "abc123", "microsoft/unixcoder-base", str(tmp_path))
    assert "org__repo" in index_path
    assert "abc123" in index_path
    assert "microsoft__unixcoder-base" in index_path
    assert index_path.endswith(".faiss")
    assert meta_path.endswith(".meta.json")


def test_is_indexed_false_when_nothing_built(tmp_path):
    assert is_indexed("org/repo", "abc123", index_root=str(tmp_path)) is False


def _fake_embed_texts(texts, model_name="x", batch_size=32):
    """Deterministic fake embeddings: one-hot-ish on text length mod 4, so ranking is
    predictable without loading a real model or calling an API."""
    import torch
    dim = 4
    vecs = []
    for t in texts:
        v = [0.0] * dim
        v[len(t) % dim] = 1.0
        vecs.append(v)
    return torch.tensor(vecs)


@pytest.fixture
def mocked_repo(monkeypatch):
    """Two files, each a single chunk (no top-level defs -> falls back to one
    fixed-window chunk each), served without touching the real repo_cache or a real
    embedding model/API."""
    contents = {
        "pkg/a.py": "x = 1\n",
        "pkg/b.py": "y = 22\n",
    }
    monkeypatch.setattr(repository_index, "is_repo_cached", lambda repo: True)
    monkeypatch.setattr(repository_index, "get_code_files_local", lambda repo, commit, ext: list(contents))
    monkeypatch.setattr(repository_index, "get_file_contents_batch", lambda repo, commit, paths: contents)
    monkeypatch.setattr(repository_index, "embed_texts", _fake_embed_texts)
    return contents


def test_build_repository_index_creates_index_and_metadata(tmp_path, mocked_repo):
    stats = build_repository_index("org/repo", "abc123", index_root=str(tmp_path))
    assert stats is not None
    assert os.path.isfile(stats["index_path"])
    assert is_indexed("org/repo", "abc123", index_root=str(tmp_path)) is True

    _, meta_path = _index_paths("org/repo", "abc123", repository_index.DEFAULT_EMBEDDING_MODEL, str(tmp_path))
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["repo"] == "org/repo"
    assert meta["commit"] == "abc123"
    assert len(meta["chunks"]) == 2
    assert {c["path"] for c in meta["chunks"]} == {"pkg/a.py", "pkg/b.py"}


def test_build_repository_index_returns_throughput_and_storage_stats(tmp_path, mocked_repo):
    stats = build_repository_index("org/repo", "abc123", index_root=str(tmp_path))
    assert stats["num_files"] == 2
    assert stats["num_chunks"] == 2
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] == 2
    assert stats["embed_elapsed_s"] >= 0
    assert stats["total_elapsed_s"] >= 0
    assert stats["index_bytes"] > 0
    assert stats["cache_bytes"] > 0


def test_build_repository_index_reindexing_same_content_is_all_cache_hits(tmp_path, mocked_repo):
    build_repository_index("org/repo", "abc123", index_root=str(tmp_path))
    # Same repo, same file content, different commit SHA -- the chunk cache is keyed by
    # content hash, not by commit, so this should hit the cache entirely (the actual point
    # of incremental indexing: unchanged content never gets re-embedded).
    stats = build_repository_index("org/repo", "def456", index_root=str(tmp_path))
    assert stats["cache_hits"] == 2
    assert stats["cache_misses"] == 0


def test_build_repository_index_force_recompute_bypasses_cache(tmp_path, mocked_repo):
    build_repository_index("org/repo", "abc123", index_root=str(tmp_path))
    stats = build_repository_index("org/repo", "def456", index_root=str(tmp_path), force_recompute=True)
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] == 2


def test_build_repository_index_duplicate_chunk_across_files_shares_one_cache_entry(tmp_path, monkeypatch):
    # Two different files with byte-identical content -- content-hash-keyed caching means
    # the second file's chunk is a cache hit even within the SAME build, not just across
    # separate builds/commits.
    identical_content = "x = 1\n"
    contents = {"pkg/a.py": identical_content, "pkg/b.py": identical_content}
    monkeypatch.setattr(repository_index, "is_repo_cached", lambda repo: True)
    monkeypatch.setattr(repository_index, "get_code_files_local", lambda repo, commit, ext: list(contents))
    monkeypatch.setattr(repository_index, "get_file_contents_batch", lambda repo, commit, paths: contents)
    monkeypatch.setattr(repository_index, "embed_texts", _fake_embed_texts)

    stats = build_repository_index("org/repo", "abc123", index_root=str(tmp_path))
    assert stats["num_chunks"] == 2
    assert stats["cache_misses"] == 1  # only one distinct chunk text across both files
    assert stats["cache_hits"] == 1


def test_build_repository_index_no_tmp_files_left_behind(tmp_path, mocked_repo):
    build_repository_index("org/repo", "abc123", index_root=str(tmp_path))
    leftovers = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")] if os.path.isdir(tmp_path) else []
    # walk the whole tree since files land in nested repo/commit/model dirs
    for root, _, files in os.walk(tmp_path):
        leftovers += [f for f in files if f.endswith(".tmp")]
    assert leftovers == []


def test_build_repository_index_raises_if_repo_not_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(repository_index, "is_repo_cached", lambda repo: False)
    with pytest.raises(ValueError):
        build_repository_index("org/uncached", "abc123", index_root=str(tmp_path))


def test_load_repository_index_returns_none_when_not_built(tmp_path):
    index, chunks = load_repository_index("org/repo", "abc123", index_root=str(tmp_path))
    assert index is None
    assert chunks is None


def test_rank_files_from_index_returns_none_when_not_indexed(tmp_path, monkeypatch):
    monkeypatch.setattr(repository_index, "embed_texts", _fake_embed_texts)
    bug = make_bug("org/repo", "abc123", ["pkg/a.py"])
    assert rank_files_from_index(bug, index_root=str(tmp_path)) is None


def test_rank_files_from_index_ranks_and_filters_to_code_files(tmp_path, mocked_repo):
    build_repository_index("org/repo", "abc123", index_root=str(tmp_path))
    bug = make_bug("org/repo", "abc123", ["pkg/a.py", "pkg/b.py"], bug_report="a")
    ranked = rank_files_from_index(bug, index_root=str(tmp_path))
    assert set(ranked) == {"pkg/a.py", "pkg/b.py"}


def test_rank_files_from_index_excludes_files_outside_code_files(tmp_path, mocked_repo):
    build_repository_index("org/repo", "abc123", index_root=str(tmp_path))
    # code_files only lists one of the two indexed files -- the other should never appear.
    bug = make_bug("org/repo", "abc123", ["pkg/a.py"], bug_report="a")
    ranked = rank_files_from_index(bug, index_root=str(tmp_path))
    assert ranked == ["pkg/a.py"]


def test_rank_files_from_index_respects_top_k(tmp_path, mocked_repo):
    build_repository_index("org/repo", "abc123", index_root=str(tmp_path))
    bug = make_bug("org/repo", "abc123", ["pkg/a.py", "pkg/b.py"], bug_report="a")
    ranked = rank_files_from_index(bug, index_root=str(tmp_path), top_k=1)
    assert len(ranked) == 1
