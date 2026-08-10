import pytest

from dataset.models import BugInstance
from evaluation.screening import screen_bug_instance, screen_manifest, summarize_screening, _average_precision


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
    assert result["average_precision"] == 0.0


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


def test_screen_bug_instance_recall_at_1_5_10_are_fractional_not_binary():
    # Two localizable GTs, ranked 1 and 8. Hit@10 is binary ("found at least one") = 1
    # for BOTH GTs found within 10. Recall@10 is fractional ("found what share of all
    # GTs") = 2/2 = 1.0 here too, but Recall@5 must diverge: only 1 of 2 GTs is within
    # rank 5, so Recall@5 = 0.5 while Hit@5 stays 1 (at least one GT found).
    bug = make_bug(
        ground_truths=["first.py", "eighth.py"],
        code_files=["first.py", "a.py", "b.py", "c.py", "d.py", "e.py", "f.py", "eighth.py"],
    )
    rank_fn = lambda b: b.code_files  # natural order: first.py=rank1, eighth.py=rank8
    result = screen_bug_instance(bug, rank_fn=rank_fn)

    assert result["hit_at"][5] == 1        # at least one GT within top 5 -> binary hit
    assert result["recall_at"][5] == 0.5   # but only 1 of 2 GTs within top 5 -> fractional
    assert result["hit_at"][10] == 1
    assert result["recall_at"][10] == 1.0  # both GTs within top 10 -> full recall


def test_screen_bug_instance_recall_at_1_zero_when_only_second_gt_ranks_first():
    bug = make_bug(
        ground_truths=["first.py", "second.py"],
        code_files=["first.py", "second.py"],
    )
    result = screen_bug_instance(bug, rank_fn=lambda b: b.code_files)
    assert result["recall_at"][1] == 0.5  # only 1 of 2 GTs at rank 1


def test_screen_bug_instance_reports_latency_for_a_real_rank_fn_call():
    import time
    bug = make_bug(ground_truths=["bar.py"], code_files=["bar.py", "x.py"])

    def slow_rank_fn(b):
        time.sleep(0.05)
        return b.code_files

    result = screen_bug_instance(bug, rank_fn=slow_rank_fn)
    assert result["latency_s"] >= 0.05


def test_screen_bug_instance_latency_zero_when_no_localizable_gt_skips_rank_fn():
    bug = make_bug(ground_truths=["new.py"], code_files=["bar.py"], patch=NEW_FILE_PATCH)
    calls = []
    result = screen_bug_instance(bug, rank_fn=lambda b: calls.append(1) or b.code_files)
    assert result["latency_s"] == 0.0
    assert calls == []  # rank_fn must never be called -- there's nothing it could find


def test_summarize_screening_reports_mean_latency():
    import time
    bug = make_bug(ground_truths=["bar.py"], code_files=["bar.py"])

    def slow_rank_fn(b):
        time.sleep(0.02)
        return b.code_files

    report = screen_manifest([bug], rank_fn=slow_rank_fn)
    summary = summarize_screening(report)
    assert summary["mean_latency_s"] >= 0.02


def test_summarize_screening_mean_latency_excludes_no_localizable_gt_instances():
    # A slow real hit alongside a skipped (no-GT) instance shouldn't have its mean latency
    # diluted toward zero by an instance where rank_fn was never even called.
    import time
    real_bug = make_bug(ground_truths=["bar.py"], code_files=["bar.py"], instance_id="real")
    skipped_bug = make_bug(ground_truths=["new.py"], code_files=["bar.py"], patch=NEW_FILE_PATCH, instance_id="skipped")

    def slow_rank_fn(b):
        time.sleep(0.03)
        return b.code_files

    report = screen_manifest([real_bug, skipped_bug], rank_fn=slow_rank_fn)
    summary = summarize_screening(report)
    assert summary["mean_latency_s"] >= 0.03  # not diluted by the skipped instance's 0.0


def test_summarize_screening_includes_recall_at_1_5_10():
    bugs = [make_bug(ground_truths=["bar.py"], code_files=["bar.py", "x.py"])]
    report = screen_manifest(bugs)
    summary = summarize_screening(report)
    assert set(summary["macro_recall_at"].keys()) >= {1, 5, 10}


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
    # single-GT AP per instance equals 1/rank, so this matches MRR here
    assert summary["map"] == pytest.approx((1 / 1 + 1 / 2) / 2)


def test_summarize_screening_handles_empty_report():
    report = {"per_instance": [], "difficulty_distribution": {}, "total": 0}
    summary = summarize_screening(report)
    assert summary["n"] == 0
    assert summary["mrr"] == 0.0
    assert summary["map"] == 0.0


def test_average_precision_all_gts_found():
    # GT at ranks 2 and 5, both found: AP = ((1/2) + (2/5)) / 2
    ap = _average_precision({"a.py": 2, "b.py": 5}, num_localizable_gts=2)
    assert ap == pytest.approx((1 / 2 + 2 / 5) / 2)


def test_average_precision_penalizes_gts_not_found():
    # Only 1 of 2 localizable GTs appears in gt_ranks (e.g. rank_fn returned a truncated
    # list) -- the missing one still counts in the denominator, dragging AP down.
    ap = _average_precision({"a.py": 2}, num_localizable_gts=2)
    assert ap == pytest.approx((1 / 2) / 2)


def test_average_precision_perfect_top_rank():
    assert _average_precision({"a.py": 1}, num_localizable_gts=1) == pytest.approx(1.0)


def test_average_precision_zero_localizable_gts():
    assert _average_precision({}, num_localizable_gts=0) == 0.0
