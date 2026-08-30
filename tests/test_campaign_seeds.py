import pytest

from rand_isopeps.campaign import derive_seed, seed_hierarchy, seed_stream


def test_seed_hierarchy_pairs_problem_and_score_draws():
    first = seed_hierarchy(4926, 3, sketch_index=0, score_index=2)
    second = seed_hierarchy(4926, 3, sketch_index=1, score_index=2)
    assert first == seed_hierarchy(4926, 3, sketch_index=0, score_index=2)
    assert first["problem"] == second["problem"]
    assert first["score"] == second["score"]
    assert first["sketch"] != second["sketch"]


def test_seed_streams_are_stable_and_distinct():
    seeds = seed_stream(4926, "replicate", 16)
    assert seeds == seed_stream(4926, "replicate", 16)
    assert len(set(seeds)) == 16
    assert all(0 <= value < 2**63 for value in seeds)
    assert derive_seed(4926, "a") != derive_seed(4926, "b")


def test_seed_inputs_are_validated():
    with pytest.raises(ValueError):
        derive_seed(-1, "problem")
    with pytest.raises(ValueError):
        seed_stream(1, "problem", -1)
