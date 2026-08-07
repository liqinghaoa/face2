from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from p2_counterfactual.assets import PRESET_NAMES
from p2_counterfactual.config import config_fingerprint
from p2_counterfactual.dataset import load_p2_manifest
from p2_counterfactual.io_utils import sha256_file, utc_now, write_json_atomic


REQUIRED_FOLD_FILES = (
    "checkpoints/best_macro_auc.pth",
    "checkpoints/last.pth",
    "config_resolved.yaml",
    "fold_metadata.json",
    "training_history.csv",
    "val_predictions_original.csv",
    "val_predictions_relighted.csv",
    "val_features_original.npz",
    "val_features_relighted.npz",
    "metrics_original.json",
    "confusion_matrix_original.csv",
    "fold_summary.json",
)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_probabilities(frame: pd.DataFrame, path: Path) -> None:
    probs = frame[["prob_control", "prob_patient"]].to_numpy(dtype=float)
    if not np.isfinite(probs).all():
        raise ValueError(f"non-finite probabilities in {path}")
    if (probs < -1e-7).any() or (probs > 1.0 + 1e-7).any():
        raise ValueError(f"probabilities outside [0,1] in {path}")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"probability rows do not sum to one in {path}")


def validate_p2_a_fold(
    *,
    fold_dir: str | Path,
    config: dict[str, Any],
    fold: int,
    manifest_path: str | Path,
    write_success: bool = True,
    expected_full_fold: bool = True,
) -> dict[str, Any]:
    fold_dir = Path(fold_dir)
    fold = int(fold)
    missing = [name for name in REQUIRED_FOLD_FILES if not (fold_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"P2-A fold {fold} missing required outputs: {missing}")

    manifest = load_p2_manifest(manifest_path)
    manifest_fold = manifest[manifest["fold"].astype(int) == fold].copy()
    original = pd.read_csv(fold_dir / "val_predictions_original.csv", dtype={"case_id": str, "patient_group_id": str})
    relighted = pd.read_csv(fold_dir / "val_predictions_relighted.csv", dtype={"case_id": str, "patient_group_id": str})
    expected_cases = set(manifest_fold["case_id"].astype(str))
    actual_cases = original["case_id"].astype(str).tolist()

    if original["fold"].astype(int).nunique() != 1 or int(original["fold"].astype(int).iloc[0]) != fold:
        raise ValueError(f"original predictions contain wrong fold for fold={fold}")
    if original["case_id"].duplicated().any():
        raise ValueError(f"original predictions contain duplicate case_id for fold={fold}")
    if expected_full_fold and len(original) != len(manifest_fold):
        raise ValueError(f"fold={fold} original row count {len(original)} != manifest fold count {len(manifest_fold)}")
    if expected_full_fold and set(actual_cases) != expected_cases:
        missing_cases = sorted(expected_cases - set(actual_cases))[:10]
        extra_cases = sorted(set(actual_cases) - expected_cases)[:10]
        raise ValueError(f"fold={fold} case mismatch missing={missing_cases} extra={extra_cases}")
    if not set(actual_cases).issubset(expected_cases):
        raise ValueError(f"fold={fold} contains predictions outside held-out manifest fold")

    manifest_key = manifest.set_index("case_id")
    for row in original.itertuples(index=False):
        manifest_row = manifest_key.loc[str(row.case_id)]
        if int(row.label) != int(manifest_row["binary_label"]):
            raise ValueError(f"label mismatch for case_id={row.case_id}")
        if str(row.patient_group_id) != str(manifest_row["patient_group_id"]):
            raise ValueError(f"patient_group_id mismatch for case_id={row.case_id}")

    _validate_probabilities(original, fold_dir / "val_predictions_original.csv")
    _validate_probabilities(relighted, fold_dir / "val_predictions_relighted.csv")

    if relighted["fold"].astype(int).nunique() != 1 or int(relighted["fold"].astype(int).iloc[0]) != fold:
        raise ValueError(f"relighted predictions contain wrong fold for fold={fold}")
    rel_counts = relighted.groupby("case_id")["preset_name"].nunique()
    if not (rel_counts == len(PRESET_NAMES)).all():
        raise ValueError(f"fold={fold} not every case has six unique relighting presets")
    rel_size = relighted.groupby("case_id").size()
    if not (rel_size == len(PRESET_NAMES)).all():
        raise ValueError(f"fold={fold} not every case has exactly six relighting rows")
    for case_id, names in relighted.groupby("case_id")["preset_name"]:
        if set(names.astype(str)) != set(PRESET_NAMES):
            raise ValueError(f"case_id={case_id} preset set is incomplete")

    with np.load(fold_dir / "val_features_original.npz", allow_pickle=False) as features_o:
        original_features = features_o["features"]
        feature_case_ids = [str(x) for x in features_o["case_ids"].tolist()]
    with np.load(fold_dir / "val_features_relighted.npz", allow_pickle=False) as features_r:
        relighted_features = features_r["features"]
        relighted_feature_case_ids = [str(x) for x in features_r["case_ids"].tolist()]
        preset_names = [str(x) for x in features_r["preset_names"].tolist()]
    if original_features.shape != (len(original), 512):
        raise ValueError(f"original feature shape {original_features.shape} != ({len(original)},512)")
    if relighted_features.shape != (len(original), len(PRESET_NAMES), 512):
        raise ValueError(f"relighted feature shape {relighted_features.shape} != ({len(original)},6,512)")
    if feature_case_ids != actual_cases or relighted_feature_case_ids != actual_cases:
        raise ValueError("feature case order does not match original prediction case order")
    if preset_names != list(PRESET_NAMES):
        raise ValueError(f"feature preset order mismatch: {preset_names}")

    checkpoint = _load_checkpoint(fold_dir / "checkpoints/best_macro_auc.pth")
    if int(checkpoint.get("fold", -1)) != fold:
        raise ValueError("checkpoint fold metadata mismatch")
    if checkpoint.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError("checkpoint config fingerprint differs from resolved formal config")

    metadata = json.loads((fold_dir / "fold_metadata.json").read_text(encoding="utf-8"))
    train_counts = metadata.get("train_class_counts", {})
    expected_train = manifest[manifest["fold"].astype(int) != fold]["binary_label"].value_counts().sort_index()
    expected_train_counts = {str(int(k)): int(v) for k, v in expected_train.to_dict().items()}
    if {str(k): int(v) for k, v in train_counts.items()} != expected_train_counts and expected_full_fold:
        raise ValueError("fold class weights metadata does not match training folds only")

    history = pd.read_csv(fold_dir / "training_history.csv")
    if history.empty:
        raise ValueError("training_history.csv is empty")
    best_idx = pd.to_numeric(history["val_macro_auc"], errors="coerce").idxmax()
    best_epoch = int(history.loc[best_idx, "epoch"])
    best_auc = float(history.loc[best_idx, "val_macro_auc"])
    payload = {
        "experiment_id": str(config["experiment_id"]),
        "fold": fold,
        "status": "P2_A_FOLD_SUCCESS",
        "case_count": int(len(original)),
        "relighted_row_count": int(len(relighted)),
        "best_epoch": best_epoch,
        "best_val_macro_auc": best_auc,
        "manifest_sha256": sha256_file(manifest_path),
        "pair_cycle_schedule_sha256": sha256_file(config["pair_cycle_schedule_path"]) if config.get("pair_cycle_schedule_path") else None,
        "config_sha256": config_fingerprint(config),
        "checkpoint_sha256": sha256_file(fold_dir / "checkpoints/best_macro_auc.pth"),
        "original_prediction_sha256": sha256_file(fold_dir / "val_predictions_original.csv"),
        "relighted_prediction_sha256": sha256_file(fold_dir / "val_predictions_relighted.csv"),
        "completed_at": utc_now(),
        "expected_full_fold": bool(expected_full_fold),
    }
    if write_success:
        write_json_atomic(fold_dir / "_FOLD_SUCCESS.json", payload)
    return payload


def fold_success_valid(fold_dir: str | Path, *, config: dict[str, Any], fold: int) -> bool:
    success_path = Path(fold_dir) / "_FOLD_SUCCESS.json"
    if not success_path.is_file():
        return False
    try:
        success = json.loads(success_path.read_text(encoding="utf-8"))
        return (
            success.get("status") == "P2_A_FOLD_SUCCESS"
            and int(success.get("fold", -1)) == int(fold)
            and success.get("experiment_id") == config["experiment_id"]
            and success.get("config_sha256") == config_fingerprint(config)
            and (Path(fold_dir) / "checkpoints/best_macro_auc.pth").is_file()
        )
    except Exception:
        return False
