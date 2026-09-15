from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/so1c_r2_overfit16_gate"
ORIGINAL_IDS = ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/preflight/overfit16_ids.txt"
ORIGINAL_CONFIG = ROOT / "config/train/skin_optics_so1/so1_decomposition_overfit_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate(record: dict, channels: dict) -> bool:
    return bool(
        float(record["loss_reduction"]) >= 0.9
        and float(channels["M"]["MAE"]) <= 0.02
        and float(channels["H"]["MAE"]) <= 0.02
        and float(channels["S"]["MAE"]) <= 0.03
        and float(channels["P"]["pred"]["std"]) > 0
        and float(channels["P"]["active_region_recall"]) > 0
    )


def test_r2_reuses_original_ids_and_only_changes_authorized_protocol_fields() -> None:
    assert RUN.is_dir(), "Run SO-1C-R2 before checking its artifacts"
    assert _sha256(ORIGINAL_IDS) == _sha256(RUN / "original_overfit16_ids.txt")
    assert (RUN / "original_overfit16_ids_sha256.txt").read_text(encoding="utf-8").strip() == _sha256(ORIGINAL_IDS)
    original = yaml.safe_load(ORIGINAL_CONFIG.read_text(encoding="utf-8"))
    resolved = yaml.safe_load((RUN / "config_resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["train"]["batch_size"] == 8
    assert resolved["train"]["max_steps"] == 3000
    for key in ("seed", "learning_rate", "weight_decay", "scheduler", "amp", "early_stopping_patience"):
        assert resolved["train"][key] == original["train"][key]


def test_r2_milestones_runtime_and_same_checkpoint_gate_are_consistent() -> None:
    for step in (1000, 1500, 2000, 2500, 3000):
        assert (RUN / "checkpoints" / f"step_{step}.pt").is_file()
        assert (RUN / "metrics" / f"step_{step}_metrics.json").is_file()
    runtime = json.loads((RUN / "runtime.json").read_text(encoding="utf-8"))
    assert runtime["optimizer_steps"] == 3000
    assert runtime["first_gate_pass_step"] is None
    assert runtime["formal_gate_pass"] is False
    assert runtime["stable_gate_pass"] is False
    best = json.loads((RUN / "metrics/best_metrics.json").read_text(encoding="utf-8"))
    assert best["record"]["formal_gate_pass"] is False
    assert _gate(best["record"], best["summary"]["channels"]) is False
    with (RUN / "gate_history.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    assert len(history) == 60
    assert not any(row["formal_gate_pass"] == "True" for row in history)
