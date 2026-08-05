import pytest

from dataset.models import BugInstance
from evaluation.screening import screen_bug_instance, screen_manifest, summarize_screening


def make_bug(ground_truths, code_files, patch="", bug_report="fix bar", **overrides):
    defaults = dict(
        repo="org/repo", instance_id="1", base_commit="c", patch=patch,
        hints_text="", ground_truths=ground_truths, bug_report=bug_report,
        code_files=code_files,
    )
    defaults.update(overrides)
    return BugInstance(**defaults)


NEW_FILE_PATCH = "diff --git a/new.py b/new.py\nnew file mode 100644\n"


def test_screen_bug_instance_easy_band_for_top_ranked_gt():
    bug = make_bug(
        ground_truths=["bar.py"],
        code_files=["bar.py", "unrelated1.py", "unrelated2.py"],
        bug_report="fix bar",
    )
    result = screen_bug_instance(bug)
    assert result["difficulty"] == "easy"
    assert result["best_rank"] == 1
    assert result["hit_at"][1] == 1


def test_screen_bug_instance_no_localizable_gt_band():
    bug = make_bug(
        ground_truths=["new.py"],
        code_files=["bar.py"],
        patch=NEW_FILE_PATCH,
        bug_report="fix new",
    )
    result = screen_bug_instance(bug)
    assert result["difficulty"] == "no_localizable_gt"
    assert result["best_rank"] is None
    assert result["gt_ranks"] == {}
    assert all(v == 0 for v in result["hit_at"].values())
    assert all(v == 0.0 for v in result["recall_at"].values())


def test_screen_bug_instance_difficulty_bands_by_rank():
    # 250 noise files plus the GT buried deep in an unrelated-looking position.
    noise = [f"noise_{i}.py" for i in range(250)]
    bug = make_bug(
        ground_truths=["zzz_target.py"],
        code_files=noise + ["zzz_target.py"],
        bug_report="completely unrelated query terms",
    )
    result = screen_bug_instance(bug)
    assert result["difficulty"] == "outside_top200"
    assert result["best_rank"] > 200


def test_screen_bug_instance_uses_custom_rank_fn():
    bug = make_bug(
        ground_truths=["bar.py"],
        code_files=["bar.py", "unrelated.py"],
    )
    # Reverse the natural ranking to prove rank_fn is actually being used.
    reversed_rank_fn = lambda b: list(reversed(b.code_files))
    result = screen_bug_instance(bug, rank_fn=reversed_rank_fn)
    assert result["gt_ranks"]["bar.py"] == 2


def test_screen_manifest_aggregates_difficulty_distribution():
    bugs = [
        make_bug(ground_truths=["bar.py"], code_files=["bar.py", "x.py"], instance_id="easy"),
        make_bug(ground_truths=["new.py"], code_files=["bar.py"], patch=NEW_FILE_PATCH, instance_id="unlocalizable"),
    ]
    report = screen_manifest(bugs)
    assert report["total"] == 2
    assert report["difficulty_distribution"]["easy"] == 1
    assert report["difficulty_distribution"]["no_localizable_gt"] == 1


def test_summarize_screening_macro_metrics():
    bugs = [
        make_bug(ground_truths=["bar.py"], code_files=["bar.py", "x.py"], instance_id="a"),
        make_bug(ground_truths=["bar.py"], code_files=["x.py", "bar.py"], instance_id="b"),
    ]
    report = screen_manifest(bugs)
    summary = summarize_screening(report)
    assert summary["n"] == 2
    # instance a: rank 1 -> hit@1=1; instance b: rank 2 -> hit@1=0, hit@5=1
    assert summary["macro_hit_at"][1] == pytest.approx(0.5)
    assert summary["macro_hit_at"][5] == pytest.approx(1.0)
    assert summary["mrr"] == pytest.approx((1 / 1 + 1 / 2) / 2)


def test_summarize_screening_handles_empty_report():
    report = {"per_instance": [], "difficulty_distribution": {}, "total": 0}
    summary = summarize_screening(report)
    assert summary["n"] == 0
    assert summary["mrr"] == 0.0
