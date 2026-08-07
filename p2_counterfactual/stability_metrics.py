from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from p2_counterfactual.assets import PRESET_NAMES
from p2_counterfactual.metrics import compute_p2_a_metrics
from p2_counterfactual.io_utils import json_safe


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def build_case_probability_table(original: pd.DataFrame, relighted: pd.DataFrame) -> pd.DataFrame:
    base = original[["case_id", "patient_group_id", "fold", "label", "prob_patient", "predicted_label"]].copy()
    base = base.rename(columns={"prob_patient": "original_probability", "predicted_label": "original_predicted_label"})
    pivot_prob = relighted.pivot(index="case_id", columns="preset_name", values="prob_patient").reset_index()
    pivot_pred = relighted.pivot(index="case_id", columns="preset_name", values="predicted_label").reset_index()
    for preset in PRESET_NAMES:
        if preset not in pivot_prob.columns:
            raise ValueError(f"relighted predictions missing preset={preset}")
    pivot_prob = pivot_prob.rename(columns={preset: f"prob_{preset}" for preset in PRESET_NAMES})
    pivot_pred = pivot_pred.rename(columns={preset: f"pred_{preset}" for preset in PRESET_NAMES})
    return base.merge(pivot_prob, on="case_id", how="inner").merge(pivot_pred, on="case_id", how="inner")


def compute_feature_cosines(original_features_npz: str | Path, relighted_features_npz: str | Path) -> tuple[pd.DataFrame, dict[str, float]]:
    with np.load(original_features_npz, allow_pickle=False) as fo:
        case_ids = np.asarray([str(x) for x in fo["case_ids"].tolist()])
        original = fo["features"].astype(np.float32)
    with np.load(relighted_features_npz, allow_pickle=False) as fr:
        relighted_case_ids = np.asarray([str(x) for x in fr["case_ids"].tolist()])
        presets = [str(x) for x in fr["preset_names"].tolist()]
        relighted = fr["features"].astype(np.float32)
    if case_ids.tolist() != relighted_case_ids.tolist():
        raise ValueError("original and relighted feature case order mismatch")
    if presets != list(PRESET_NAMES):
        raise ValueError("relighted feature preset order mismatch")
    numerator = (original[:, None, :] * relighted).sum(axis=2)
    denom = np.linalg.norm(original, axis=1)[:, None] * np.linalg.norm(relighted, axis=2)
    cosines = numerator / np.clip(denom, 1e-12, None)
    rows = []
    for index, case_id in enumerate(case_ids):
        row = {"case_id": case_id}
        row.update({f"feature_cosine_{preset}": float(cosines[index, j]) for j, preset in enumerate(PRESET_NAMES)})
        row["mean_feature_cosine"] = float(np.mean(cosines[index]))
        rows.append(row)
    per_case = pd.DataFrame(rows)
    per_preset = {preset: float(np.mean(cosines[:, j])) for j, preset in enumerate(PRESET_NAMES)}
    return per_case, per_preset


def compute_relighted_pairwise_feature_cosines(relighted_features_npz: str | Path) -> pd.DataFrame:
    with np.load(relighted_features_npz, allow_pickle=False) as fr:
        case_ids = np.asarray([str(x) for x in fr["case_ids"].tolist()])
        presets = [str(x) for x in fr["preset_names"].tolist()]
        relighted = fr["features"].astype(np.float32)
    if presets != list(PRESET_NAMES):
        raise ValueError("relighted feature preset order mismatch")
    denom = np.linalg.norm(relighted, axis=2)
    rows = []
    pair_names = []
    for left in range(len(PRESET_NAMES)):
        for right in range(left + 1, len(PRESET_NAMES)):
            pair_names.append((left, right, f"{PRESET_NAMES[left]}__{PRESET_NAMES[right]}"))
    for index, case_id in enumerate(case_ids):
        row = {"case_id": case_id}
        values = []
        for left, right, name in pair_names:
            numerator = float((relighted[index, left, :] * relighted[index, right, :]).sum())
            value = numerator / max(1e-12, float(denom[index, left] * denom[index, right]))
            row[f"relighted_pairwise_feature_cosine_{name}"] = float(value)
            values.append(float(value))
        row["mean_relighted_pairwise_feature_cosine"] = float(np.mean(values))
        row["mean_relighted_pairwise_feature_distance"] = float(1.0 - np.mean(values))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_stability_outputs(
    *,
    original_predictions: pd.DataFrame,
    relighted_predictions: pd.DataFrame,
    original_features_npz: str | Path,
    relighted_features_npz: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    table = build_case_probability_table(original_predictions, relighted_predictions)
    probability_columns = ["original_probability", *[f"prob_{preset}" for preset in PRESET_NAMES]]
    relight_probability_columns = [f"prob_{preset}" for preset in PRESET_NAMES]
    table["prediction_std"] = table[probability_columns].astype(float).std(axis=1, ddof=0)
    table["min_relighted_probability"] = table[relight_probability_columns].astype(float).min(axis=1)
    table["max_relighted_probability"] = table[relight_probability_columns].astype(float).max(axis=1)
    table["probability_range"] = table["max_relighted_probability"] - table["min_relighted_probability"]
    table["relighted_prediction_std"] = table[relight_probability_columns].astype(float).std(axis=1, ddof=0)
    pairwise_probability_diffs = []
    for left in range(len(PRESET_NAMES)):
        for right in range(left + 1, len(PRESET_NAMES)):
            left_col = f"prob_{PRESET_NAMES[left]}"
            right_col = f"prob_{PRESET_NAMES[right]}"
            pairwise_probability_diffs.append((table[left_col].astype(float) - table[right_col].astype(float)).abs())
    table["mean_relighted_pairwise_abs_probability_diff"] = pd.concat(pairwise_probability_diffs, axis=1).mean(axis=1)
    for preset in PRESET_NAMES:
        table[f"abs_probability_delta_{preset}"] = (table[f"prob_{preset}"].astype(float) - table["original_probability"].astype(float)).abs()
    pred_cols = [f"pred_{preset}" for preset in PRESET_NAMES]
    table["flip_preset_count"] = (table[pred_cols].astype(int).ne(table["original_predicted_label"].astype(int), axis=0)).sum(axis=1)
    table["case_flip"] = (table["flip_preset_count"] > 0).astype(int)

    feature_case, per_preset_cosine = compute_feature_cosines(original_features_npz, relighted_features_npz)
    table = table.merge(feature_case, on="case_id", how="inner")
    relighted_pairwise_feature = compute_relighted_pairwise_feature_cosines(relighted_features_npz)
    table = table.merge(relighted_pairwise_feature, on="case_id", how="inner")
    labels = table["label"].astype(int).to_numpy()
    per_preset_rows = []
    preset_aucs = []
    for preset in PRESET_NAMES:
        probs = table[f"prob_{preset}"].to_numpy(dtype=float)
        auc = _safe_auc(labels, probs)
        relighted_subset = relighted_predictions[relighted_predictions["preset_name"].astype(str) == preset].copy()
        preset_metrics = compute_p2_a_metrics(
            relighted_subset["label"].astype(int).to_numpy(),
            relighted_subset[["prob_control", "prob_patient"]].to_numpy(dtype=float),
        )
        preset_aucs.append(auc)
        per_preset_rows.append(
            {
                "preset_name": preset,
                "macro_auc": auc,
                "macro_f1": float(preset_metrics["macro_f1"]),
                "balanced_accuracy": float(preset_metrics["balanced_accuracy"]),
                "mean_probability": float(np.mean(probs)),
                "mean_abs_probability_delta": float(table[f"abs_probability_delta_{preset}"].mean()),
                "feature_cosine": per_preset_cosine[preset],
            }
        )
    finite_aucs = [value for value in preset_aucs if np.isfinite(value)]
    stability_metrics = {
        "prediction_std": _summary_stats(table["prediction_std"].to_numpy(dtype=float)),
        "case_level_label_flip_rate": float(table["case_flip"].mean()),
        "flip_preset_count_mean": float(table["flip_preset_count"].mean()),
        "per_preset_auc": {preset: float(value) for preset, value in zip(PRESET_NAMES, preset_aucs)},
        "worst_light_auc": float(np.min(finite_aucs)) if finite_aucs else float("nan"),
        "mean_relighted_auc": float(np.mean(finite_aucs)) if finite_aucs else float("nan"),
        "auc_range": float(np.max(finite_aucs) - np.min(finite_aucs)) if finite_aucs else float("nan"),
        "mean_feature_cosine": float(table["mean_feature_cosine"].mean()),
        "median_feature_cosine": float(table["mean_feature_cosine"].median()),
        "per_preset_feature_cosine": per_preset_cosine,
        "mean_abs_probability_delta": {
            preset: float(table[f"abs_probability_delta_{preset}"].mean()) for preset in PRESET_NAMES
        },
        "relighted_prediction_std": _summary_stats(table["relighted_prediction_std"].to_numpy(dtype=float)),
        "mean_relighted_pairwise_abs_probability_diff": float(table["mean_relighted_pairwise_abs_probability_diff"].mean()),
        "mean_relighted_pairwise_feature_cosine": float(table["mean_relighted_pairwise_feature_cosine"].mean()),
        "mean_relighted_pairwise_feature_distance": float(table["mean_relighted_pairwise_feature_distance"].mean()),
        "case_count": int(len(table)),
    }
    per_preset = pd.DataFrame(per_preset_rows)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "case_level_stability.csv", index=False, encoding="utf-8-sig")
        per_preset.to_csv(out / "per_preset_metrics.csv", index=False, encoding="utf-8-sig")
        (out / "stability_metrics.json").write_text(json.dumps(json_safe(stability_metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"case_level": table, "per_preset": per_preset, "metrics": stability_metrics}
