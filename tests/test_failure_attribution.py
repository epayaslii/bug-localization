from dataset.models import BugInstance
from evaluation.screening import screen_bug_instance, screen_manifest
from evaluation.failure_attribution import (
    classify_retrieval_reach,
    summarize_retrieval_reach,
    prepare_oracle_candidate_set,
    REACHED_CANDIDATE_SET,
    RETRIEVAL_FAILURE,
)


def make_bug(ground_truths, code_files, bug_report="fix bar", **overrides):
    defaults = dict(
        repo="org/repo", instance_id="1", base_commit="c", patch="",
        hints_text="", ground_truths=ground_truths, bug_report=bug_report,
        code_files=code_files,
    )
    defaults.update(overrides)
    return BugInstance(**defaults)


def test_classify_retrieval_reach_splits_by_candidate_size():
    near = make_bug(ground_truths=["bar.py"], code_files=["bar.py", "x1.py", "x2.py"], bug_report="fix bar issue")
    far = make_bug(
        ground_truths=["zzz_unrelated_name.py"],
        code_files=["zzz_unrelated_name.py"] + [f"noise_bar_{i}.py" for i in range(20)],
        bug_report="fix bar issue",
    )

    near_result = screen_bug_instance(near)
    far_result = screen_bug_instance(far)

    near_reach = classify_retrieval_reach(near_result, candidate_size=5)
    far_reach = classify_retrieval_reach(far_result, candidate_size=5)

    assert near_reach["bar.py"] == REACHED_CANDIDATE_SET
    assert far_reach["zzz_unrelated_name.py"] == RETRIEVAL_FAILURE


def test_classify_retrieval_reach_empty_for_no_localizable_gt():
    bug = make_bug(ground_truths=["new.py"], code_files=["bar.py"],
                    patch="diff --git a/new.py b/new.py\nnew file mode 100644\n")
    result = screen_bug_instance(bug)
    reach = classify_retrieval_reach(result, candidate_size=100)
    assert reach == {}


def test_summarize_retrieval_reach_counts_fully_unreachable_instances():
    near = make_bug(ground_truths=["bar.py"], code_files=["bar.py", "x1.py", "x2.py"],
                     bug_report="fix bar issue", instance_id="near")
    far = make_bug(
        ground_truths=["zzz_unrelated_name.py"],
        code_files=["zzz_unrelated_name.py"] + [f"noise_bar_{i}.py" for i in range(20)],
        bug_report="fix bar issue", instance_id="far",
    )
    report = screen_manifest([near, far])
    summary = summarize_retrieval_reach(report, candidate_size=5)

    assert summary["file_level_counts"][REACHED_CANDIDATE_SET] == 1
    assert summary["file_level_counts"][RETRIEVAL_FAILURE] == 1
    assert summary["instances_with_localizable_gt"] == 2
    assert summary["instances_fully_unreachable"] == 1


def test_prepare_oracle_candidate_set_injects_missing_gt():
    bug = make_bug(
        ground_truths=["zzz_unrelated_name.py"],
        code_files=["zzz_unrelated_name.py"] + [f"noise_bar_{i}.py" for i in range(20)],
        bug_report="fix bar issue",
    )
    candidate_set, injected = prepare_oracle_candidate_set(bug, candidate_size=5)
    assert "zzz_unrelated_name.py" in candidate_set
    assert injected == ["zzz_unrelated_name.py"]
    assert len(candidate_set) == 6  # 5 BM25 top candidates + 1 injected GT


def test_prepare_oracle_candidate_set_injects_nothing_when_already_top_ranked():
    bug = make_bug(ground_truths=["bar.py"], code_files=["bar.py", "x1.py", "x2.py"],
                    bug_report="fix bar issue")
    candidate_set, injected = prepare_oracle_candidate_set(bug, candidate_size=5)
    assert injected == []
    assert "bar.py" in candidate_set
