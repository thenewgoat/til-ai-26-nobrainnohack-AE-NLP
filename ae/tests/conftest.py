"""Pytest configuration for AE tests.

Tests live under `ae/tests/`. They import from `ae/src/` (the runtime package
— `scripted/`, `ae_manager.py`, `policy.py`, `features.py`) and from
`ae/src/training/` (the training-side modules — `bc.py`, `measure_tournament.py`,
`evaluate.py`, `critic.py`, etc.). Both directories are put on `sys.path` here.
"""

import sys
from pathlib import Path

# ae/src — runtime package: scripted/, ae_manager.py, policy.py, features.py.
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

# ae/src/training — training-side modules imported by name in some tests
# (e.g. measure_tournament, evaluate, bc, critic).
_TRAINING = _SRC / "training"
sys.path.insert(0, str(_TRAINING))
