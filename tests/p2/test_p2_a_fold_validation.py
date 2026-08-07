from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from p2_counterfactual.assets import PRESET_NAMES
from p2_counterfactual.config import config_fingerprint, resolve_p2_a_config
from p2_counterfactual.fold_validation import validate_p2_a_fold


def _config(tmp_path: Path) -> dict:
    config = resolve_p2_a_config("config/p2/p2_a/p2_a1_colorjitter.yaml")
    config["output_root"] = str(tmp_path)
    return config


def _write_fold_outputs(fold_dir: Path, config: dict, *, duplicate_original: bool = False, drop_preset: bool = False) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "checkpoints").mkdir(exist_ok=True)
    (fold_dir / "logs").mkdir(exist_ok=True)
    cases = ["100037382", "100960732"]
    original = pd.DataFrame(
        [
            {"case_id": cid, "patient_group_id": cid, "fold": 0, "label": 1, "prob_control": 0.4, "prob_patient": 0.6, "predicted_label": 1}
            for cid in cases
        ]
    )
    if duplicate_original:
        original = pd.concat([original, original.iloc[[0]]], ignore_index=True)
    original.to_csv(fold_dir / "val_predictions_original.csv", index=False)
    rows = []
    presets = list(PRESET_NAMES)[:-1] if drop_preset else list(PRESET_NAMES)
    for cid in cases:
        for preset in presets:
            rows.append({"case_id": cid, "patient_group_id": cid, "fold": 0, "label": 1, "preset_name": preset, "prob_control": 0.45, "prob_patient": 0.55, "predicted_label": 1})
    pd.DataFrame(rows).to_csv(fold_dir / "val_predictions_relighted.csv", index=False)
    np.savez_compressed(fold_dir / "val_features_original.npz", case_ids=np.asarray(cases), features=np.ones((2, 512), np.float32))
    np.savez_compressed(fold_dir / "val_features_relighted.npz", case_ids=np.asarray(cases), preset_names=np.asarray(PRESET_NAMES), features=np.ones((2, 6, 512), np.float32))
    (fold_dir / "config_resolved.yaml").write_text("experiment_id: p2_a1_colorjitter\n", encoding="utf-8")
    (fold_dir / "fold_metadata.json").write_text(json.dumps({"train_class_counts": {"0": 92, "1": 308}}), encoding="utf-8")
    pd.DataFrame([{"epoch": 1, "val_macro_auc": 0.5}]).to_csv(fold_dir / "training_history.csv", index=False)
    (fold_dir / "metrics_original.json").write_text("{}", encoding="utf-8")
    pd.DataFrame([[0, 0], [0, 2]]).to_csv(fold_dir / "confusion_matrix_original.csv")
    (fold_dir / "fold_summary.json").write_text("{}", encoding="utf-8")
    payload = {"model_state_dict": {}, "fold": 0, "config_fingerprint": config_fingerprint(config)}
    torch.save(payload, fold_dir / "checkpoints/best_macro_auc.pth")
    torch.save(payload, fold_dir / "checkpoints/last.pth")


def test_fold_validation_writes_success_marker(tmp_path: Path) -> None:
    config = _config(tmp_path)
    fold_dir = tmp_path / "fold_0"
    _write_fold_outputs(fold_dir, config)
    result = validate_p2_a_fold(fold_dir=fold_dir, config=config, fold=0, manifest_path=config["manifest_path"], expected_full_fold=False)
    assert result["status"] == "P2_A_FOLD_SUCCESS"
    assert (fold_dir / "_FOLD_SUCCESS.json").is_file()


def test_fold_validation_fails_when_checkpoint_missing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    fold_dir = tmp_path / "fold_0"
    _write_fold_outputs(fold_dir, config)
    (fold_dir / "checkpoints/best_macro_auc.pth").unlink()
    with pytest.raises(FileNotFoundError):
        validate_p2_a_fold(fold_dir=fold_dir, config=config, fold=0, manifest_path=config["manifest_path"], expected_full_fold=False)


def test_fold_validation_fails_on_duplicate_original_or_missing_preset(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dup = tmp_path / "dup"
    _write_fold_outputs(dup, config, duplicate_original=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_p2_a_fold(fold_dir=dup, config=config, fold=0, manifest_path=config["manifest_path"], expected_full_fold=False)
    missing = tmp_path / "missing"
    _write_fold_outputs(missing, config, drop_preset=True)
    with pytest.raises(ValueError, match="six"):
        validate_p2_a_fold(fold_dir=missing, config=config, fold=0, manifest_path=config["manifest_path"], expected_full_fold=False)
