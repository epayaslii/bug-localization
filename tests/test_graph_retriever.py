import method.graph_retriever as graph_retriever
from dataset.models import BugInstance
from method.graph_retriever import (
    _extract_imports,
    _resolve_import_to_path,
    build_import_graph,
    rank_files_graph_traversal,
)

SAMPLE_WITH_IMPORTS = """
import os
from pkg.sub import mod
from . import relative_thing

def f():
    pass
"""


def make_bug(code_files, **overrides):
    defaults = dict(
        repo="org/repo",
        instance_id="1",
        base_commit="sha",
        patch="",
        hints_text="",
        ground_truths=[],
        bug_report="fix the bug",
        code_files=code_files,
    )
    defaults.update(overrides)
    return BugInstance(**defaults)


def test_extract_imports_returns_raw_module_names():
    imports = _extract_imports(SAMPLE_WITH_IMPORTS)
    assert "os" in imports
    assert "pkg.sub" in imports
    # a bare "from . import x" has no node.module -- should be skipped, not crash
    assert len(imports) == 2


def test_extract_imports_returns_empty_on_syntax_error():
    assert _extract_imports("def broken(:\n") == []


def test_resolve_import_to_path_dot_to_slash():
    files = {"pkg/sub/mod.py", "other.py"}
    assert _resolve_import_to_path("pkg.sub.mod", files) == "pkg/sub/mod.py"


def test_resolve_import_to_path_falls_back_to_init_py():
    files = {"pkg/sub/__init__.py"}
    assert _resolve_import_to_path("pkg.sub", files) == "pkg/sub/__init__.py"


def test_resolve_import_to_path_returns_none_when_not_found():
    files = {"other.py"}
    assert _resolve_import_to_path("pkg.sub.mod", files) is None


def test_build_import_graph_returns_empty_when_repo_not_cached(monkeypatch):
    monkeypatch.setattr(graph_retriever, "is_repo_cached", lambda repo: False)
    bug = make_bug(code_files=["a.py", "b.py"])
    graph = build_import_graph(bug)
    assert graph == {"a.py": set(), "b.py": set()}


def test_build_import_graph_builds_undirected_edges(monkeypatch):
    monkeypatch.setattr(graph_retriever, "is_repo_cached", lambda repo: True)
    monkeypatch.setattr(
        graph_retriever, "get_file_contents_batch",
        lambda repo, commit, paths: {"pkg/a.py": "import pkg.b\n", "pkg/b.py": ""},
    )
    bug = make_bug(code_files=["pkg/a.py", "pkg/b.py"])
    graph = build_import_graph(bug)
    assert graph["pkg/a.py"] == {"pkg/b.py"}
    assert graph["pkg/b.py"] == {"pkg/a.py"}  # undirected, even though only a.py imports b.py


def test_rank_files_graph_traversal_orders_by_hop_distance(monkeypatch):
    # chain: seed -> b (1 hop) -> c (2 hops) -> d (3 hops, out of reach at hops=2)
    monkeypatch.setattr(graph_retriever, "is_repo_cached", lambda repo: True)
    monkeypatch.setattr(
        graph_retriever, "get_file_contents_batch",
        lambda repo, commit, paths: {
            "seed.py": "import b\n", "b.py": "import c\n", "c.py": "import d\n", "d.py": "",
        },
    )
    bug = make_bug(code_files=["seed.py", "b.py", "c.py", "d.py"])
    # resolve_import_to_path needs matching filenames -- use flat top-level names so
    # "import b" -> "b.py" resolves directly. seed_size=1 so only seed.py is an actual
    # seed -- b.py/c.py/d.py must be discovered via BFS, not auto-seeded.
    ranked = rank_files_graph_traversal(
        bug, seed_ranking=["seed.py", "d.py", "c.py", "b.py"], seed_size=1, hops=2
    )
    # seed.py (0 hops), b.py (1 hop), c.py (2 hops) should all outrank d.py (unreached at hops=2)
    assert ranked.index("seed.py") < ranked.index("d.py")
    assert ranked.index("b.py") < ranked.index("d.py")
    assert ranked.index("c.py") < ranked.index("d.py")
    assert ranked[0] == "seed.py"


def test_rank_files_graph_traversal_falls_back_to_seed_order_when_repo_not_cached(monkeypatch):
    monkeypatch.setattr(graph_retriever, "is_repo_cached", lambda repo: False)
    bug = make_bug(code_files=["a.py", "b.py", "c.py"])
    seed_ranking = ["a.py", "b.py", "c.py"]
    ranked = rank_files_graph_traversal(bug, seed_ranking=seed_ranking)
    assert ranked == seed_ranking


def test_rank_files_graph_traversal_truncates_to_top_k(monkeypatch):
    monkeypatch.setattr(graph_retriever, "is_repo_cached", lambda repo: False)
    bug = make_bug(code_files=["a.py", "b.py", "c.py"])
    ranked = rank_files_graph_traversal(bug, seed_ranking=["a.py", "b.py", "c.py"], top_k=2)
    assert ranked == ["a.py", "b.py"]


def test_rank_files_graph_traversal_appends_files_outside_seed_ranking(monkeypatch):
    # bug.code_files can be a superset of seed_ranking if seed_ranking was pre-truncated
    monkeypatch.setattr(graph_retriever, "is_repo_cached", lambda repo: False)
    bug = make_bug(code_files=["a.py", "b.py", "leftover.py"])
    ranked = rank_files_graph_traversal(bug, seed_ranking=["a.py", "b.py"])
    assert ranked == ["a.py", "b.py", "leftover.py"]
