from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/train/run_skin_optics_so1c_r2_overfit16_gate.py"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_r2_protocol_freezes_only_authorized_training_changes() -> None:
    source = _source()
    assert "BATCH_SIZE = 8" in source
    assert "MAX_STEPS = 3000" in source
    assert "EVALUATION_INTERVAL = 50" in source
    assert "AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)" in source
    assert "masked_smooth_l1_loss" in source


def test_r2_requires_exact_original_id_hash_and_milestones() -> None:
    source = _source()
    assert "ORIGINAL_IDS_PATH" in source
    assert "Copied original overfit16 IDs hash mismatch" in source
    assert "MILESTONES = {1000, 1500, 2000, 2500, 3000}" in source
    assert "step_{step}.pt" in source


def test_r2_gate_and_stability_use_single_evaluation_records() -> None:
    source = _source()
    assert 'record["selection_metric"] = record["M_MAE"] + record["H_MAE"]' in source
    assert 'record["formal_gate_pass"] = gate(summary, reduction)' in source
    assert "first_gate_pass_step" in source
    assert "stable_gate_pass" in source


def test_r2_output_must_be_new() -> None:
    assert "R2 output directory must be new and empty" in _source()
