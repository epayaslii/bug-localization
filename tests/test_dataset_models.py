from dataset.models import BugInstance


def make_bug(**overrides):
    defaults = dict(
        repo="org/repo",
        instance_id="1",
        base_commit="sha",
        patch="diff --git a/foo.py b/foo.py",
        hints_text="some hints",
        ground_truths=["foo.py"],
        bug_report="the bug report text",
        code_files=["foo.py", "bar.py"],
    )
    defaults.update(overrides)
    return BugInstance(**defaults)


def test_after_commit_defaults_to_none():
    bug = make_bug()
    assert bug.after_commit is None


def test_after_commit_can_be_set():
    bug = make_bug(after_commit="cafebabe")
    assert bug.after_commit == "cafebabe"


def test_to_string_includes_all_key_fields():
    bug = make_bug()
    text = bug.to_string()
    assert "org/repo" in text
    assert "sha" in text
    assert "some hints" in text
    assert "the bug report text" in text
    assert "foo.py" in text and "bar.py" in text


def test_get_token_count_returns_expected_keys():
    bug = make_bug()
    stats = bug.get_token_count(model="gpt-4o")
    assert set(stats.keys()) == {
        "bug_report_tokens", "hints_tokens", "code_files_tokens", "total_prompt_tokens",
    }
    assert all(isinstance(v, int) and v >= 0 for v in stats.values())
    # total should be at least as large as any individual component
    assert stats["total_prompt_tokens"] >= stats["bug_report_tokens"]


def test_get_token_count_scales_with_more_code_files():
    small = make_bug(code_files=["a.py"])
    large = make_bug(code_files=[f"file_{i}.py" for i in range(50)])
    assert large.get_token_count()["code_files_tokens"] > small.get_token_count()["code_files_tokens"]
