"""The only code path allowed to load test-split ground truth (protocol §13.1, §9.7).

Built in WP6, after all selected checkpoints exist and the final validation
manipulation checks are complete. tests/test_isolation.py enforces that no
other module in this repository references test MIDI.
"""
from __future__ import annotations


def evaluate_final(*args, **kwargs):
    raise NotImplementedError("WP6")
