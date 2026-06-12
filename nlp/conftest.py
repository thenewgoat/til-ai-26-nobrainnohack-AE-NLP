"""Pytest configuration: make modules under src/ importable as top-level."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
