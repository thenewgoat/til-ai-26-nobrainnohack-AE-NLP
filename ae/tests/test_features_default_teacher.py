"""FeatureBuilder default teacher matches the BC default (balanced_extreme_opening)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from features import FeatureBuilder


def test_default_teacher_is_balanced_extreme_opening():
    fb = FeatureBuilder()
    assert fb.strategy.name == "balanced_extreme_opening"
