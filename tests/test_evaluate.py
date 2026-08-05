import pytest

from method.evaluate import Evaluator


@pytest.fixture
def evaluator():
    return Evaluator()


def test_evaluate_candidate_files_perfect_hit(evaluator):
    result = evaluator._evaluate_candidate_files(["a.py", "b.py"], ["a.py"], k=2)
    assert result["accuracy"] == 1
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(1.0)


def test_evaluate_candidate_files_complete_miss(evaluator):
    result = evaluator._evaluate_candidate_files(["a.py", "b.py"], ["c.py"], k=2)
    assert result["accuracy"] == 0
    assert result["tp"] == 0
    assert result["fp"] == 2
    assert result["fn"] == 1
    assert result["precision"] == 0
    assert result["recall"] == 0
    assert result["f1"] == 0


def test_evaluate_candidate_files_respects_k_truncation():
    evaluator = Evaluator()
    # Ground truth is ranked 3rd; with k=2 it's outside the truncated candidate list.
    result = evaluator._evaluate_candidate_files(["x.py", "y.py", "target.py"], ["target.py"], k=2)
    assert result["accuracy"] == 0
    assert result["fn"] == 1

    # With k=3 (or unrestricted) it should be a hit.
    result_full = evaluator._evaluate_candidate_files(["x.py", "y.py", "target.py"], ["target.py"], k=3)
    assert result_full["accuracy"] == 1
    assert result_full["tp"] == 1


def test_evaluate_candidate_files_empty_predictions(evaluator):
    result = evaluator._evaluate_candidate_files([], ["a.py"], k=5)
    assert result["accuracy"] == 0
    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["fn"] == 1
    assert result["precision"] == 0


def test_calculate_overall_metrics_aggregates_across_bugs(evaluator):
    per_bug_results = {
        "bug1": {"accuracy": 1, "tp": 1, "fp": 0, "fn": 0},
        "bug2": {"accuracy": 0, "tp": 0, "fp": 1, "fn": 1},
    }
    overall = evaluator._calculate_overall_metrics(per_bug_results)
    assert overall["total_bugs"] == 2
    assert overall["accuracy"] == pytest.approx(0.5)
    assert overall["total_tp"] == 1
    assert overall["total_fp"] == 1
    assert overall["total_fn"] == 1
    assert overall["precision"] == pytest.approx(0.5)
    assert overall["recall"] == pytest.approx(0.5)


def test_calculate_overall_metrics_empty_results(evaluator):
    assert evaluator._calculate_overall_metrics({}) == {}


class FakeResponse:
    def __init__(self, candidate_files):
        self.candidate_files = candidate_files


def test_evaluate_reads_ground_truths_from_bug_object(evaluator):
    class FakeBug:
        ground_truths = ["a.py"]

    responses = {
        "1": {"response": FakeResponse(["a.py"]), "bug": FakeBug()},
    }
    results = evaluator.evaluate(responses, k=1)
    assert results["per_bug"]["1"]["accuracy"] == 1
    assert results["overall"]["total_bugs"] == 1


def test_evaluate_reads_ground_truths_from_dict_key(evaluator):
    responses = {
        "1": {"response": FakeResponse(["a.py"]), "ground_truths": ["a.py"]},
    }
    results = evaluator.evaluate(responses, k=1)
    assert results["per_bug"]["1"]["accuracy"] == 1
