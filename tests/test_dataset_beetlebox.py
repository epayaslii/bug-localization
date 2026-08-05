from datasets import Dataset

from dataset.beetlebox import BeetleBox


def _sample_dataset():
    return Dataset.from_dict({
        "status": ["closed"],
        "repo_name": ["org/repo"],
        "repo_url": ["https://github.com/org/repo"],
        "issue_id": [1],
        "updated_files": [["foo.py"]],
        "title": ["fix bug"],
        "body": ["something broke"],
        "issue_url": ["https://github.com/org/repo/issues/1"],
        "pull_url": ["https://github.com/org/repo/pull/2"],
        "before_fix_sha": ["abc123"],
        "after_fix_sha": ["def456"],
        "report_datetime": ["2024-01-01"],
        "language": ["python"],
        "commit_datetime": ["2024-01-02"],
    })


def test_beetlebox_loads_from_local_path(tmp_path, monkeypatch):
    """BEETLEBOX_LOCAL_PATH should load via load_from_disk instead of hitting the Hub --
    the same offline pattern SWEBENCH_LOCAL_PATH already provides in dataset/swebench.py,
    confirmed missing here by a real MN5 offline-execution run (see memory)."""
    local_path = tmp_path / "beetlebox_local"
    _sample_dataset().save_to_disk(str(local_path))

    monkeypatch.setenv("BEETLEBOX_LOCAL_PATH", str(local_path))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    bb = BeetleBox()

    assert bb.data is not None
    assert len(bb.data) == 1
    assert bb.data[0]["repo_name"] == "org/repo"


def test_beetlebox_local_path_never_calls_load_dataset(tmp_path, monkeypatch):
    """When BEETLEBOX_LOCAL_PATH is set, the network path (load_dataset against the Hub)
    must never be invoked -- this is the whole point of the offline path."""
    local_path = tmp_path / "beetlebox_local2"
    _sample_dataset().save_to_disk(str(local_path))
    monkeypatch.setenv("BEETLEBOX_LOCAL_PATH", str(local_path))

    import dataset.beetlebox as beetlebox_mod

    def fail_if_called(*args, **kwargs):
        raise AssertionError("load_dataset should not be called when BEETLEBOX_LOCAL_PATH is set")

    monkeypatch.setattr(beetlebox_mod, "load_dataset", fail_if_called)

    bb = BeetleBox()
    assert len(bb.data) == 1
