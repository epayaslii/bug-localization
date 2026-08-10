import pytest

from dataset.models import BugInstance
import method.fusion_signals as fusion_signals
from method.fusion_signals import (
    _resolve_import_to_path,
    rank_files_ast_similarity,
    rank_files_dependency_graph,
    rank_files_commit_recency,
)


def make_bug(code_files, bug_report="fix the bug", repo="org/repo", base_commit="sha", **overrides):
    defaults = dict(
        repo=repo, instance_id="1", base_commit=base_commit, patch="",
        hints_text="", ground_truths=[], bug_report=bug_report, code_files=code_files,
    )
    defaults.update(overrides)
    return BugInstance(**defaults)


# ---------------- rank_files_ast_similarity ----------------

def test_rank_files_ast_similarity_favors_symbol_name_overlap(monkeypatch):
    contents = {
        "pkg/shape_util.py": "class ShapeUtil:\n    def compute_area(self):\n        return 1\n",
        "pkg/unrelated.py": "class Unrelated:\n    def noop(self):\n        pass\n",
    }
    monkeypatch.setattr(fusion_signals, "is_repo_cached", lambda repo: True)
    monkeypatch.setattr(fusion_signals, "get_file_contents_batch", lambda repo, commit, paths: contents)

    bug = make_bug(["pkg/unrelated.py", "pkg/shape_util.py"], bug_report="ShapeUtil.compute_area is broken")
    ranked = rank_files_ast_similarity(bug)
    assert ranked[0] == "pkg/shape_util.py"


def test_rank_files_ast_similarity_uncached_repo_scores_everything_zero(monkeypatch):
    monkeypatch.setattr(fusion_signals, "is_repo_cached", lambda repo: False)
    bug = make_bug(["a.py", "b.py"], bug_report="fix Foo")
    ranked = rank_files_ast_similarity(bug)
    assert set(ranked) == {"a.py", "b.py"}  # no crash, stable fallback order


def test_rank_files_ast_similarity_syntax_error_file_scores_zero_not_crash(monkeypatch):
    contents = {"broken.py": "def broken(:\n", "ok.py": "class Foo:\n    pass\n"}
    monkeypatch.setattr(fusion_signals, "is_repo_cached", lambda repo: True)
    monkeypatch.setattr(fusion_signals, "get_file_contents_batch", lambda repo, commit, paths: contents)

    bug = make_bug(["broken.py", "ok.py"], bug_report="Foo is broken")
    ranked = rank_files_ast_similarity(bug)
    assert ranked[0] == "ok.py"


def test_rank_files_ast_similarity_respects_top_k(monkeypatch):
    monkeypatch.setattr(fusion_signals, "is_repo_cached", lambda repo: False)
    bug = make_bug(["a.py", "b.py", "c.py"])
    assert len(rank_files_ast_similarity(bug, top_k=1)) == 1


# ---------------- _resolve_import_to_path ----------------

def test_resolve_import_to_path_module_file():
    assert _resolve_import_to_path("pkg.utils", {"pkg/utils.py"}) == "pkg/utils.py"


def test_resolve_import_to_path_package_init():
    assert _resolve_import_to_path("pkg.sub", {"pkg/sub/__init__.py"}) == "pkg/sub/__init__.py"


def test_resolve_import_to_path_unresolvable_returns_none():
    assert _resolve_import_to_path("numpy", {"pkg/utils.py"}) is None


# ---------------- rank_files_dependency_graph ----------------

def test_rank_files_dependency_graph_boosts_file_that_imports_a_seed(monkeypatch):
    contents = {
        "pkg/seed.py": "class Seed:\n    pass\n",
        "pkg/importer.py": "import pkg.seed\n\nclass Importer:\n    pass\n",
        "pkg/unrelated.py": "class Unrelated:\n    pass\n",
    }
    monkeypatch.setattr(fusion_signals, "is_repo_cached", lambda repo: True)
    monkeypatch.setattr(fusion_signals, "get_file_contents_batch", lambda repo, commit, paths: contents)

    bug = make_bug(["pkg/unrelated.py", "pkg/importer.py", "pkg/seed.py"])
    ranked = rank_files_dependency_graph(bug, seed_ranking=["pkg/seed.py"])
    assert ranked.index("pkg/importer.py") < ranked.index("pkg/unrelated.py")


def test_rank_files_dependency_graph_boosts_file_imported_by_a_seed(monkeypatch):
    contents = {
        "pkg/target.py": "class Target:\n    pass\n",
        "pkg/seed.py": "import pkg.target\n\nclass Seed:\n    pass\n",
        "pkg/unrelated.py": "class Unrelated:\n    pass\n",
    }
    monkeypatch.setattr(fusion_signals, "is_repo_cached", lambda repo: True)
    monkeypatch.setattr(fusion_signals, "get_file_contents_batch", lambda repo, commit, paths: contents)

    bug = make_bug(["pkg/unrelated.py", "pkg/target.py", "pkg/seed.py"])
    ranked = rank_files_dependency_graph(bug, seed_ranking=["pkg/seed.py"])
    assert ranked.index("pkg/target.py") < ranked.index("pkg/unrelated.py")


def test_rank_files_dependency_graph_seed_size_limits_seed_pool(monkeypatch):
    contents = {
        "pkg/a.py": "class A:\n    pass\n",
        "pkg/importer.py": "import pkg.a\n\nclass Importer:\n    pass\n",
    }
    monkeypatch.setattr(fusion_signals, "is_repo_cached", lambda repo: True)
    monkeypatch.setattr(fusion_signals, "get_file_contents_batch", lambda repo, commit, paths: contents)

    bug = make_bug(["pkg/importer.py", "pkg/a.py"])
    # seed_size=0 -> no seeds at all -> importer gets no adjacency boost over a
    ranked = rank_files_dependency_graph(bug, seed_ranking=["pkg/a.py"], seed_size=0)
    assert set(ranked) == {"pkg/importer.py", "pkg/a.py"}


# ---------------- rank_files_commit_recency ----------------

def test_rank_files_commit_recency_favors_more_recent_timestamps(monkeypatch):
    monkeypatch.setattr(fusion_signals, "is_repo_cached", lambda repo: True)
    monkeypatch.setattr(
        fusion_signals, "get_recent_commit_timestamps",
        lambda repo, commit, max_commits=2000: {"old.py": 100, "new.py": 999},
    )
    bug = make_bug(["old.py", "new.py"])
    assert rank_files_commit_recency(bug) == ["new.py", "old.py"]


def test_rank_files_commit_recency_uncached_repo_returns_original_order(monkeypatch):
    monkeypatch.setattr(fusion_signals, "is_repo_cached", lambda repo: False)
    bug = make_bug(["b.py", "a.py"])
    assert rank_files_commit_recency(bug) == ["b.py", "a.py"]


def test_rank_files_commit_recency_untouched_file_falls_to_bottom(monkeypatch):
    monkeypatch.setattr(fusion_signals, "is_repo_cached", lambda repo: True)
    monkeypatch.setattr(
        fusion_signals, "get_recent_commit_timestamps",
        lambda repo, commit, max_commits=2000: {"touched.py": 500},
    )
    bug = make_bug(["never_touched.py", "touched.py"])
    ranked = rank_files_commit_recency(bug)
    assert ranked[0] == "touched.py"
