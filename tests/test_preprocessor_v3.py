from __future__ import annotations

from training.scripts.preprocessor_v3 import split_households


def test_household_splits_are_deterministic_and_disjoint():
    splits = split_households([f"hhv3_{index}" for index in range(20)])
    groups = [set(values) for values in splits.values()]

    assert set.union(*groups) == {f"hhv3_{index}" for index in range(20)}
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert splits == split_households([f"hhv3_{index}" for index in range(20)])
