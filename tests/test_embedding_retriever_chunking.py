from method.embedding_retriever import _chunk_file_content

SAMPLE_SOURCE = '''import os
from collections import Counter

class ShapeUtil:
    """Utilities for cartesian shape handling."""

    def compute_area(self):
        return 42

def cartesian_transform(x, y):
    return x, y
'''


def test_chunk_file_content_splits_by_top_level_definitions():
    chunks = _chunk_file_content(SAMPLE_SOURCE)
    assert len(chunks) == 3
    assert "import os" in chunks[0]
    assert "class ShapeUtil" in chunks[1]
    assert "def cartesian_transform" in chunks[2]


def test_chunk_file_content_falls_back_to_character_windows_on_syntax_error():
    broken = "def broken(:\n" + ("x = 1\n" * 500)
    chunks = _chunk_file_content(broken, max_chunk_chars=200, overlap_chars=20)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_file_content_returns_empty_for_none_or_empty():
    assert _chunk_file_content(None) == []
    assert _chunk_file_content("") == []


def test_chunk_file_content_falls_back_when_no_top_level_definitions():
    # Valid Python, but nothing to chunk by structurally (no functions/classes).
    plain = "x = 1\ny = 2\n" * 200
    chunks = _chunk_file_content(plain, max_chunk_chars=500, overlap_chars=50)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
