"""Validation-only evaluation entry point, used by training (spec WP2 §8).

Refuses manifest="test": only src/vapt/eval/final.py (WP6) may score test
ground truth. The full wiring of predictions -> ClipScores/voice_metrics
against val_manifest.json is completed in WP3, once infer.py exists.
"""
from __future__ import annotations

from pathlib import Path


def evaluate_predictions(pred_dir: Path, manifest: str = "val") -> dict:
    if manifest == "test":
        raise ValueError(
            "evaluate_predictions refuses manifest='test': test scoring happens only in "
            "src/vapt/eval/final.py (WP6), per protocol §9.7"
        )
    raise NotImplementedError("WP3: wires predictions through metrics.py/melody.py against val_manifest.json")
