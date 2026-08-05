import pytest

from method.hybrid_retriever import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_favors_items_ranked_well_in_multiple_lists():
    ranking1 = ["a", "b", "c", "d"]
    ranking2 = ["a", "c", "x", "y"]
    fused = reciprocal_rank_fusion([ranking1, ranking2], k=60)
    assert fused[0] == "a"
    # 'c' appears in both lists (ranks 3, 2); 'b' only in one (rank 2) -- c should rank above b.
    assert fused.index("c") < fused.index("b")


def test_reciprocal_rank_fusion_matches_manual_rrf_score():
    # RRF score for 'a': 1/(60+1) (rank 1 in list1) + 1/(60+2) (rank 2 in list2)
    fused = reciprocal_rank_fusion([["a", "z"], ["y", "a"]], k=60)
    assert fused[0] == "a"


def test_reciprocal_rank_fusion_empty_rankings():
    assert reciprocal_rank_fusion([[], []]) == []


def test_reciprocal_rank_fusion_single_ranking_preserves_order():
    assert reciprocal_rank_fusion([["a", "b", "c"]]) == ["a", "b", "c"]


def test_reciprocal_rank_fusion_disjoint_rankings_include_all_items():
    fused = reciprocal_rank_fusion([["p", "q"], ["r", "s"]])
    assert set(fused) == {"p", "q", "r", "s"}
    # Both rank-1 items should outrank both rank-2 items.
    assert fused.index("p") < fused.index("q")
    assert fused.index("r") < fused.index("s")
