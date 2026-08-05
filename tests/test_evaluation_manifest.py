import pytest

from dataset.models import BugInstance
from evaluation.manifest import build_manifest, select_diverse_manifest, save_manifest, load_manifest


def make_bugs(repo_sizes):
    """repo_sizes: {repo_name: instance_count}"""
    bugs = []
    for repo, n in repo_sizes.items():
        for i in range(n):
            bugs.append(BugInstance(
                repo=repo, instance_id=f"{repo}-{i}", base_commit=f"c{i}",
                patch="", hints_text="", ground_truths=["foo.py"],
                bug_report="fix the bug", code_files=["foo.py", "bar.py"],
            ))
    return bugs


def test_select_diverse_manifest_respects_max_per_repo():
    bugs = make_bugs({"org/a": 5, "org/b": 3, "org/c": 1})
    selected = select_diverse_manifest(bugs, size=5, seed=42, max_per_repo=2)

    counts = {}
    for b in selected:
        counts[b.repo] = counts.get(b.repo, 0) + 1
    assert all(c <= 2 for c in counts.values())


def test_select_diverse_manifest_caps_below_requested_size_when_pool_too_concentrated():
    # 3 repos capped at 2 each -> max achievable is 5, even though pool has 9 instances.
    bugs = make_bugs({"org/a": 5, "org/b": 3, "org/c": 1})
    selected = select_diverse_manifest(bugs, size=5, seed=42, max_per_repo=2)
    assert len(selected) == 5


def test_select_diverse_manifest_raises_when_size_exceeds_pool():
    bugs = make_bugs({"org/a": 2})
    with pytest.raises(ValueError):
        select_diverse_manifest(bugs, size=10, seed=42, max_per_repo=2)


def test_build_manifest_is_deterministic_for_same_seed():
    bugs = make_bugs({"org/a": 5, "org/b": 3, "org/c": 1})
    m1 = build_manifest("testds", bugs, size=5, seed=42, max_per_repo=2)
    m2 = build_manifest("testds", bugs, size=5, seed=42, max_per_repo=2)
    assert m1["manifest_id"] == m2["manifest_id"]
    assert [i["instance_id"] for i in m1["instances"]] == [i["instance_id"] for i in m2["instances"]]


def test_build_manifest_id_changes_with_different_seed():
    bugs = make_bugs({"org/a": 5, "org/b": 3, "org/c": 1})
    m1 = build_manifest("testds", bugs, size=5, seed=1, max_per_repo=2)
    m2 = build_manifest("testds", bugs, size=5, seed=2, max_per_repo=2)
    assert m1["manifest_id"] != m2["manifest_id"]


def test_build_manifest_records_metadata():
    bugs = make_bugs({"org/a": 5, "org/b": 3, "org/c": 1})
    manifest = build_manifest("testds", bugs, size=5, seed=42, max_per_repo=2, pool_size=9)
    assert manifest["dataset"] == "testds"
    assert manifest["size"] == 5
    assert manifest["requested_size"] == 5
    assert manifest["seed"] == 42
    assert manifest["max_per_repo"] == 2
    assert manifest["pool_size"] == 9
    assert manifest["distinct_repos"] == 3
    assert len(manifest["instances"]) == 5
    for inst in manifest["instances"]:
        assert set(inst.keys()) == {"instance_id", "repo", "base_commit"}


def test_save_and_load_manifest_round_trip(tmp_path):
    bugs = make_bugs({"org/a": 5, "org/b": 3, "org/c": 1})
    manifest = build_manifest("testds", bugs, size=5, seed=42, max_per_repo=2)
    path = save_manifest(manifest, str(tmp_path / "manifest.json"))
    loaded = load_manifest(path)
    assert loaded == manifest


def test_save_manifest_default_path_uses_manifest_id(tmp_path, monkeypatch):
    import evaluation.manifest as manifest_mod
    monkeypatch.setattr(manifest_mod, "DEFAULT_MANIFEST_DIR", str(tmp_path))

    bugs = make_bugs({"org/a": 3})
    manifest = build_manifest("testds", bugs, size=2, seed=42, max_per_repo=2)
    path = save_manifest(manifest)
    assert path == str(tmp_path / f"{manifest['manifest_id']}.json")
    assert load_manifest(path) == manifest
