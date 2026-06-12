"""Frame-stack constants exist in both features.py and policy.py."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import features as F
import policy as P


def test_features_stack_constants():
    assert F.STACK == 5
    assert F.STACKED_GRID_CHANNELS == F.GRID_CHANNELS * F.STACK == 85
    assert F.STACKED_SCALARS == F.FEATURE_SCALARS * F.STACK == 50


def test_policy_stack_constants():
    assert P.STACK == 5
    assert P.STACKED_GRID_CHANNELS == P.GRID_CHANNELS * P.STACK == 85
    assert P.STACKED_SCALARS == P.SYMBOLIC_SCALARS * P.STACK == 50


def test_constants_agree_across_modules():
    assert F.STACK == P.STACK
    assert F.STACKED_GRID_CHANNELS == P.STACKED_GRID_CHANNELS
    assert F.STACKED_SCALARS == P.STACKED_SCALARS
