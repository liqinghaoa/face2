from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p2_counterfactual.assets import PRESET_NAMES
from p2_counterfactual.config import config_fingerprint
from p2_counterfactual.dataset import load_p2_manifest
from p2_counterfactual.fold_validation import validate_p2_a_fold
from p2_counterfactual.io_utils import json_safe, sha256_file, utc_now, write_json_atomic
from p2_counterfactual.metrics import compute_p2_a_metrics, scalar_metrics
from p2_counterfactual.stability_metrics import compute_stability_outputs


def _string_array(values: Any, *, max_chars: int = 128) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=f"<U{int(max_chars)}")


def _write_confusion_matrix(path: Path, matrix: Any) -> None:
    pd.DataFrame(np.asarray(matrix, dtype=int), index=["true_control", "true_patient"], columns=["pred_control", "pred_patient"]).to_csv(
        path, encoding="utf-8-sig"
    )


def aggregate_p2_a_experiment(
    *,
    experiment_dir: str | Path,
    config: dict[str, Any],
    manifest_path: str | Path,
    folds: list[int] | tuple[int, ...] = (0, 1, 2, 3, 4),
    expected_case_count: int = 500,
    expected_full_folds: bool = True,
) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir)
    oof_dir = experiment_dir / "oof"
    summary_dir = experiment_dir / "summary"
    oof_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_p2_manifest(manifest_path)
    fold_success = []
    originals = []
    relighteds = []
    original_features = []
    relighted_features = []
    feature_case_ids = []
    relighted_feature_case_ids = []

    for fold in folds:
        fold_dir = experiment_dir / f"fold_{int(fold)}"
        fold_success.append(
            validate_p2_a_fold(
                fold_dir=fold_dir,
                config=config,
                fold=int(fold),
                manifest_path=manifest_path,
                write_success=True,
                expected_full_fold=expected_full_folds,
            )
        )
        original = pd.read_csv(fold_dir / "val_predictions_original.csv", dtype={"case_id": str, "patient_group_id": str})
        relighted = pd.read_csv(fold_dir / "val_predictions_relighted.csv", dtype={"case_id": str, "patient_group_id": str})
        originals.append(original)
        relighteds.append(relighted)
        with np.load(fold_dir / "val_features_original.npz", allow_pickle=False) as fo:
            original_features.append(fo["features"].astype(np.float32))
            feature_case_ids.extend([str(x) for x in fo["case_ids"].tolist()])
        with np.load(fold_dir / "val_features_relighted.npz", allow_pickle=False) as fr:
            relighted_features.append(fr["features"].astype(np.float32))
            relighted_feature_case_ids.extend([str(x) for x in fr["case_ids"].tolist()])
            preset_names = [str(x) for x in fr["preset_names"].tolist()]
            if preset_names != list(PRESET_NAMES):
                raise ValueError("relighted feature preset order mismatch during aggregation")

    oof_original = pd.concat(originals, ignore_index=True).reset_index(drop=True)
    oof_relighted = pd.concat(relighteds, ignore_index=True).reset_index(drop=True)
    if expected_full_folds and len(oof_original) != expected_case_count:
        raise ValueError(f"OOF original row count {len(oof_original)} != {expected_case_count}")
    if oof_original["case_id"].duplicated().any():
        raise ValueError("OOF original contains duplicate case_id")
    if expected_full_folds and set(oof_original["case_id"]) != set(manifest["case_id"].astype(str)):
        raise ValueError("OOF original case set differs from manifest")
    if expected_full_folds and len(oof_relighted) != expected_case_count * len(PRESET_NAMES):
        raise ValueError(f"OOF relighted row count {len(oof_relighted)} != {expected_case_count * len(PRESET_NAMES)}")
    if not (oof_relighted.groupby("case_id").size() == len(PRESET_NAMES)).all():
        raise ValueError("OOF relighted must contain exactly six rows per case")
    if not (oof_relighted.groupby("case_id")["preset_name"].nunique() == len(PRESET_NAMES)).all():
        raise ValueError("OOF relighted preset names incomplete")

    manifest_key = manifest.set_index("case_id")
    for row in oof_original.itertuples(index=False):
        manifest_row = manifest_key.loc[str(row.case_id)]
        if int(row.fold) != int(manifest_row["fold"]) or int(row.label) != int(manifest_row["binary_label"]):
            raise ValueError(f"OOF original fold/label mismatch for case_id={row.case_id}")
        if str(row.patient_group_id) != str(manifest_row["patient_group_id"]):
            raise ValueError(f"OOF original patient_group_id mismatch for case_id={row.case_id}")

    oof_original.to_csv(oof_dir / "oof_predictions_original.csv", index=False, encoding="utf-8-sig")
    oof_relighted.to_csv(oof_dir / "oof_predictions_relighted.csv", index=False, encoding="utf-8-sig")
    features_o = np.concatenate(original_features, axis=0).astype(np.float32)
    features_r = np.concatenate(relighted_features, axis=0).astype(np.float32)
    if feature_case_ids != oof_original["case_id"].astype(str).tolist():
        raise ValueError("original OOF feature case order mismatch")
    if relighted_feature_case_ids != oof_original["case_id"].astype(str).tolist():
        raise ValueError("relighted OOF feature case order mismatch")
    np.savez_compressed(oof_dir / "oof_features_original.npz", case_ids=_string_array(feature_case_ids), features=features_o)
    np.savez_compressed(
        oof_dir / "oof_features_relighted.npz",
        case_ids=_string_array(relighted_feature_case_ids),
        preset_names=_string_array(PRESET_NAMES, max_chars=64),
        features=features_r,
    )

    metrics = compute_p2_a_metrics(
        oof_original["label"].astype(int).to_numpy(),
        oof_original[["prob_control", "prob_patient"]].to_numpy(dtype=float),
    )
    (summary_dir / "metrics_original_oof.json").write_text(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_confusion_matrix(summary_dir / "confusion_matrix_original_oof.csv", metrics["confusion_matrix"])
    stability = compute_stability_outputs(
        original_predictions=oof_original,
        relighted_predictions=oof_relighted,
        original_features_npz=oof_dir / "oof_features_original.npz",
        relighted_features_npz=oof_dir / "oof_features_relighted.npz",
        output_dir=summary_dir,
    )
    success = {
        "experiment_id": config["experiment_id"],
        "status": "P2_A_EXPERIMENT_SUCCESS",
        "fold_count": int(len(folds)),
        "oof_case_count": int(len(oof_original)),
        "relighted_view_count": int(len(oof_relighted)),
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": config_fingerprint(config),
        "oof_sha256": sha256_file(oof_dir / "oof_predictions_original.csv"),
        "summary_metrics": scalar_metrics(metrics),
        "stability_metrics": json_safe(stability["metrics"]),
        "fold_success": fold_success,
        "completed_at": utc_now(),
        "expected_full_folds": bool(expected_full_folds),
    }
    write_json_atomic(experiment_dir / "_EXPERIMENT_SUCCESS.json", success)
    return success
