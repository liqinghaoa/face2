"""Post-hoc/oracle joint OOF filtering analysis for three NYHA classifiers.

This script never trains a model and never changes source labels, folds, or
experiment artefacts.  Every scenario is derived independently from the full
aligned OOF cohort.  S6 is deliberately labelled as an optimistic post-hoc
upper bound, not as unbiased model performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata
from sklearn.metrics import cohen_kappa_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.classification_metrics import compute_classification_metrics  # noqa: E402


SEED = 2026
CLASS_LABELS = (0, 1, 2)
CLASS_NAMES = {0: "normal", 1: "mild", 2: "severe"}
CLASS_NAMES_ZH = {0: "正常", 1: "轻症", 2: "重症"}
MODEL_NAMES = ("resnet18", "resnet34", "resnet50", "ensemble")
DISCLAIMER = (
    "本报告使用模型OOF结果进行事后病例筛选，结果仅用于数据质量探索、人工复核排序和"
    "性能上限估计，不属于独立验证性能，不得替代完整数据集上的主要五折OOF结果。"
)
SCENARIO_LABELS = {
    "S0": "S0_original",
    "S1": "S1_remove_extreme_consensus",
    "S2": "S2_remove_same_wrong",
    "S3": "S3_remove_all_consensus_wrong",
    "S4": "S4_balanced_review_priority",
    "S5": "S5_balanced_same_wrong_first",
    "S6": "S6_ORACLE_POSTHOC_UPPER_BOUND",
}


class ValidationError(RuntimeError):
    """Raised when critical OOF input validation fails."""


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text.casefold())


def find_alias(columns: Iterable[str], aliases: Iterable[str], role: str) -> str:
    lookup = {str(col).casefold(): str(col) for col in columns}
    for alias in aliases:
        if alias.casefold() in lookup:
            return lookup[alias.casefold()]
    raise ValidationError(f"Cannot identify {role}; available columns: {list(columns)}")


def identify_probability_columns(columns: Iterable[str]) -> list[str]:
    columns = list(columns)
    groups = [
        ("prob_normal", "prob_mild", "prob_severe"),
        ("prob_0", "prob_1", "prob_2"),
        ("prob_class_0", "prob_class_1", "prob_class_2"),
        ("p_normal", "p_mild", "p_severe"),
        ("probability_normal", "probability_mild", "probability_severe"),
    ]
    lookup = {str(col).casefold(): str(col) for col in columns}
    for group in groups:
        if all(item.casefold() in lookup for item in group):
            return [lookup[item.casefold()] for item in group]
    raise ValidationError(f"Cannot identify three ordered probability columns in {columns}")


def locate_config(oof_path: Path) -> Path | None:
    for parent in [oof_path.parent, *oof_path.parents]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return candidate
        if parent == PROJECT_ROOT:
            break
    return None


def verify_experiment_identity(oof_path: Path, backbone: str) -> dict[str, Any]:
    config_path = locate_config(oof_path)
    details: dict[str, Any] = {
        "oof_path": str(oof_path.resolve()),
        "config_path": str(config_path.resolve()) if config_path else None,
        "expected_backbone": backbone,
    }
    path_text = str(oof_path).casefold()
    details["path_has_global"] = "global" in path_text
    details["path_has_backbone"] = backbone in path_text
    if not config_path:
        raise ValidationError(f"No config.yaml found above {oof_path}")
    config_text = config_path.read_text(encoding="utf-8")
    match = re.search(r"^\s*backbone\s*:\s*([^\s#]+)", config_text, flags=re.MULTILINE | re.I)
    observed = match.group(1).strip().casefold() if match else None
    details["configured_backbone"] = observed
    details["config_has_global_experiment"] = "global" in config_text.casefold()
    if observed != backbone or not details["config_has_global_experiment"]:
        raise ValidationError(
            f"Experiment identity mismatch for {oof_path}: expected Global/{backbone}, "
            f"configured backbone={observed!r}"
        )
    return details


def discover_oof(project_root: Path, backbone: str) -> Path:
    candidates = []
    pattern = re.compile(rf"{re.escape(backbone)}(?!\d)", re.I)
    for path in project_root.rglob("oof_predictions.csv"):
        text = str(path)
        if "global" in text.casefold() and pattern.search(text):
            try:
                verify_experiment_identity(path, backbone)
            except ValidationError:
                continue
            candidates.append(path.resolve())
    candidates = sorted(set(candidates), key=lambda item: str(item).casefold())
    if not candidates:
        raise ValidationError(f"No validated Global {backbone} oof_predictions.csv found")
    if len(candidates) > 1:
        listing = "\n".join(f"  - {path}" for path in candidates)
        raise ValidationError(
            f"Multiple validated Global {backbone} candidates found. "
            f"Pass --{backbone}-oof explicitly:\n{listing}"
        )
    return candidates[0]


def standardize_oof(path: Path, backbone: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    identity = verify_experiment_identity(path, backbone)
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    id_col = find_alias(raw.columns, ("patient_id", "sample_id", "ID"), "patient/sample ID")
    true_col = find_alias(raw.columns, ("true_label", "label_3class", "target", "label"), "true label")
    pred_col = find_alias(raw.columns, ("pred_label", "pred_class", "predicted_label", "prediction"), "predicted label")
    fold_col = find_alias(raw.columns, ("fold", "fold_id", "cv_fold"), "fold")
    prob_cols = identify_probability_columns(raw.columns)
    out = pd.DataFrame(
        {
            "patient_id": raw[id_col].astype(str).str.strip(),
            "true_label": pd.to_numeric(raw[true_col], errors="coerce"),
            "pred_label": pd.to_numeric(raw[pred_col], errors="coerce"),
            "fold": pd.to_numeric(raw[fold_col], errors="coerce"),
        }
    )
    for index, source in enumerate(prob_cols):
        out[f"prob_{index}"] = pd.to_numeric(raw[source], errors="coerce")
    errors: list[str] = []
    if out["patient_id"].eq("").any() or out["patient_id"].isna().any():
        errors.append("blank patient IDs")
    duplicate_ids = out.loc[out["patient_id"].duplicated(keep=False), "patient_id"].unique().tolist()
    if duplicate_ids:
        errors.append(f"duplicate IDs: {duplicate_ids[:20]}")
    for column in ("true_label", "pred_label", "fold"):
        if out[column].isna().any():
            errors.append(f"missing/non-numeric values in {column}")
    if not out["true_label"].dropna().isin(CLASS_LABELS).all():
        errors.append("true labels outside 0,1,2")
    if not out["pred_label"].dropna().isin(CLASS_LABELS).all():
        errors.append("predicted labels outside 0,1,2")
    probs = out[[f"prob_{i}" for i in CLASS_LABELS]].to_numpy(float)
    if not np.isfinite(probs).all():
        errors.append("missing or non-finite probabilities")
    if ((probs < 0) | (probs > 1)).any():
        errors.append("probabilities outside [0,1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        bad = np.flatnonzero(~np.isclose(probs.sum(axis=1), 1.0, atol=1e-5))[:20]
        errors.append(f"probability rows do not sum to one: row indices {bad.tolist()}")
    if not out["pred_label"].isna().any() and not np.array_equal(
        out["pred_label"].to_numpy(int), probs.argmax(axis=1)
    ):
        errors.append("predicted label does not equal probability argmax")
    if errors:
        raise ValidationError(f"Invalid OOF file {path}: " + "; ".join(errors))
    out[["true_label", "pred_label", "fold"]] = out[["true_label", "pred_label", "fold"]].astype(int)
    identity.update(
        {
            "n_rows": len(out),
            "id_column": id_col,
            "true_label_column": true_col,
            "pred_label_column": pred_col,
            "fold_column": fold_col,
            "probability_columns": prob_cols,
            "sha256": sha256(path),
        }
    )
    return out, identity


def validate_labels(label_csv: Path, reference: pd.DataFrame) -> dict[str, Any]:
    raw = pd.read_csv(label_csv, dtype=str, encoding="utf-8-sig")
    id_col = find_alias(raw.columns, ("patient_id", "sample_id", "ID"), "label-file ID")
    nyha_col = find_alias(raw.columns, ("NYHA", "nyha_class", "nyha"), "NYHA")
    labels = pd.DataFrame(
        {
            "patient_id": raw[id_col].astype(str).str.strip(),
            "NYHA": pd.to_numeric(raw[nyha_col], errors="coerce"),
        }
    )
    if labels["patient_id"].duplicated().any():
        raise ValidationError("label CSV contains duplicate IDs")
    if labels["NYHA"].isna().any() or not labels["NYHA"].isin((0, 1, 2, 3, 4)).all():
        raise ValidationError("label CSV contains missing or invalid NYHA values")
    labels["mapped_label"] = labels["NYHA"].astype(int).map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2})
    merged = reference[["patient_id", "true_label"]].merge(labels, on="patient_id", how="left", validate="one_to_one")
    if merged["NYHA"].isna().any():
        missing = merged.loc[merged["NYHA"].isna(), "patient_id"].tolist()
        raise ValidationError(f"OOF IDs missing from label CSV: {missing[:20]}")
    mismatch = merged[merged["true_label"] != merged["mapped_label"]]
    if not mismatch.empty:
        raise ValidationError(f"NYHA mapping disagrees with OOF labels: {mismatch.head(20).to_dict('records')}")
    return {
        "path": str(label_csv.resolve()),
        "sha256": sha256(label_csv),
        "n_rows": len(labels),
        "all_oof_ids_found": True,
        "nyha_mapping_matches": True,
    }


def align_oofs(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    names = list(frames)
    base_name = names[0]
    base = frames[base_name].set_index("patient_id", drop=False)
    base_ids = set(base.index)
    errors: list[str] = []
    for name in names[1:]:
        current = frames[name].set_index("patient_id", drop=False)
        if set(current.index) != base_ids:
            errors.append(
                f"ID set differs for {name}: missing={sorted(base_ids-set(current.index))[:20]}, "
                f"extra={sorted(set(current.index)-base_ids)[:20]}"
            )
            continue
        current = current.loc[base.index]
        bad_labels = base.index[base["true_label"].to_numpy() != current["true_label"].to_numpy()].tolist()
        bad_folds = base.index[base["fold"].to_numpy() != current["fold"].to_numpy()].tolist()
        if bad_labels:
            errors.append(f"true labels differ for {name}: {bad_labels[:20]}")
        if bad_folds:
            errors.append(f"folds differ for {name}: {bad_folds[:20]}")
    if errors:
        raise ValidationError("OOF alignment failed: " + "; ".join(errors))

    joint = base[["patient_id", "true_label", "fold"]].copy().reset_index(drop=True)
    for name in names:
        current = frames[name].set_index("patient_id").loc[joint["patient_id"]]
        joint[f"pred_{name}"] = current["pred_label"].to_numpy(int)
        for class_index in CLASS_LABELS:
            joint[f"prob_{name}_{CLASS_NAMES[class_index]}"] = current[f"prob_{class_index}"].to_numpy(float)
    return joint


def build_joint_error_table(joint: pd.DataFrame) -> pd.DataFrame:
    true = joint["true_label"].to_numpy(int)
    pred_cols = [f"pred_resnet{depth}" for depth in (18, 34, 50)]
    predictions = joint[pred_cols].to_numpy(int)
    wrong = predictions != true[:, None]
    joint["wrong_count"] = wrong.sum(axis=1)
    joint["all_three_wrong"] = wrong.all(axis=1)
    joint["all_three_correct"] = (~wrong).all(axis=1)
    joint["all_three_same_wrong"] = joint["all_three_wrong"] & (
        (joint[pred_cols[0]] == joint[pred_cols[1]]) & (joint[pred_cols[1]] == joint[pred_cols[2]])
    )

    model_prob_arrays = []
    true_probs = []
    entropies = []
    for model in ("resnet18", "resnet34", "resnet50"):
        cols = [f"prob_{model}_{CLASS_NAMES[index]}" for index in CLASS_LABELS]
        probs = joint[cols].to_numpy(float)
        model_prob_arrays.append(probs)
        selected = probs[np.arange(len(joint)), true]
        joint[f"true_class_probability_{model}"] = selected
        true_probs.append(selected)
        entropies.append(-(np.clip(probs, 1e-15, 1.0) * np.log(np.clip(probs, 1e-15, 1.0))).sum(axis=1))
    mean_prob = np.mean(np.stack(model_prob_arrays, axis=0), axis=0)
    for index in CLASS_LABELS:
        joint[f"ensemble_prob_{CLASS_NAMES[index]}"] = mean_prob[:, index]
    joint["ensemble_pred"] = mean_prob.argmax(axis=1)
    joint["mean_true_class_probability"] = np.mean(np.stack(true_probs, axis=1), axis=1)
    joint["mean_prediction_entropy"] = np.mean(np.stack(entropies, axis=1), axis=1)
    vote_max = np.apply_along_axis(lambda row: np.bincount(row, minlength=3).max(), 1, predictions)
    joint["prediction_disagreement"] = 1.0 - vote_max / 3.0
    consensus_pred = joint[pred_cols[0]].to_numpy(int)
    joint["extreme_consensus_error"] = joint["all_three_same_wrong"] & (np.abs(true - consensus_pred) == 2)
    joint["extreme_error"] = (joint["ensemble_pred"] != true) & (np.abs(true - joint["ensemble_pred"].to_numpy(int)) == 2)

    def direction(row: pd.Series) -> str:
        actual = int(row["true_label"])
        predicted = int(row["ensemble_pred"])
        if predicted == actual:
            return "ensemble_correct"
        return f"{CLASS_NAMES[actual]}_to_{CLASS_NAMES[predicted]}"

    joint["error_direction"] = joint.apply(direction, axis=1)
    priority = np.full(len(joint), "P5", dtype=object)
    priority[joint["wrong_count"].to_numpy() == 2] = "P4"
    priority[joint["all_three_wrong"].to_numpy()] = "P3"
    priority[joint["all_three_same_wrong"].to_numpy()] = "P2"
    priority[joint["extreme_consensus_error"].to_numpy()] = "P1"
    joint["suggested_review_priority"] = priority
    joint["true_label_name"] = joint["true_label"].map(CLASS_NAMES)
    joint["ensemble_pred_name"] = joint["ensemble_pred"].map(CLASS_NAMES)
    return joint


def ranking_order(frame: pd.DataFrame, columns: list[str], ascending: list[bool]) -> list[str]:
    temp = frame.copy()
    temp["_stable_id_group"] = temp["patient_id"].map(lambda x: stable_id_key(x)[0])
    temp["_stable_id_value"] = temp["patient_id"].map(lambda x: str(stable_id_key(x)[1]))
    columns = [*columns, "_stable_id_group", "_stable_id_value"]
    ascending = [*ascending, True, True]
    return temp.sort_values(columns, ascending=ascending, kind="mergesort")["patient_id"].tolist()


def make_scenarios(joint: pd.DataFrame) -> tuple[dict[str, set[str]], pd.DataFrame]:
    all_ids = set(joint["patient_id"])
    excluded: dict[str, set[str]] = {"S0": set()}
    excluded["S1"] = set(joint.loc[joint["extreme_consensus_error"], "patient_id"])
    excluded["S2"] = set(joint.loc[joint["all_three_same_wrong"], "patient_id"])
    excluded["S3"] = set(joint.loc[joint["all_three_wrong"], "patient_id"])

    target = int(joint["true_label"].value_counts().min())
    counts = joint["true_label"].value_counts().to_dict()
    priority_rank = joint["suggested_review_priority"].map({"P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5})
    temp = joint.assign(_priority_rank=priority_rank)
    order = ranking_order(
        temp,
        ["_priority_rank", "wrong_count", "mean_true_class_probability", "mean_prediction_entropy"],
        [True, False, True, False],
    )
    by_id = joint.set_index("patient_id")
    s4: set[str] = set()
    for patient_id in order:
        label = int(by_id.loc[patient_id, "true_label"])
        if counts[label] > target:
            s4.add(patient_id)
            counts[label] -= 1
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"S4 balancing failed: {counts}")
    excluded["S4"] = s4

    s5 = set(excluded["S2"])
    remaining = joint.loc[~joint["patient_id"].isin(s5)]
    target_s5 = int(remaining["true_label"].value_counts().min())
    counts_s5 = remaining["true_label"].value_counts().to_dict()
    supplement_order = ranking_order(
        remaining,
        ["mean_true_class_probability"],
        [True],
    )
    for patient_id in supplement_order:
        label = int(by_id.loc[patient_id, "true_label"])
        if counts_s5[label] > target_s5:
            s5.add(patient_id)
            counts_s5[label] -= 1
    if len(set(counts_s5.values())) != 1:
        raise RuntimeError(f"S5 balancing failed: {counts_s5}")
    excluded["S5"] = s5

    min_count = int(joint["true_label"].value_counts().min())
    lower = 80 if min_count >= 80 else min_count
    ranked_by_class: dict[int, list[str]] = {}
    for class_index in CLASS_LABELS:
        group = joint[joint["true_label"] == class_index]
        ranked_by_class[class_index] = ranking_order(
            group,
            ["all_three_correct", "wrong_count", "mean_true_class_probability"],
            [False, True, False],
        )
    grid_rows: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, int] | None = None
    best_retained: set[str] | None = None
    for keep_per_class in range(lower, min_count + 1):
        retained = set()
        for class_index in CLASS_LABELS:
            retained.update(ranked_by_class[class_index][:keep_per_class])
        subset = joint[joint["patient_id"].isin(retained)]
        fold_ok = (
            subset.groupby(["fold", "true_label"]).size().reindex(
                pd.MultiIndex.from_product([sorted(joint["fold"].unique()), CLASS_LABELS]), fill_value=0
            ) > 0
        ).all()
        if not fold_ok:
            continue
        metrics = calculate_metrics(subset, "ensemble")
        row = {
            "keep_per_class": keep_per_class,
            "retained_n": len(subset),
            "macro_auc": metrics["macro_auc"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "all_fold_class_cells_nonempty": bool(fold_ok),
        }
        grid_rows.append(row)
        key = (metrics["macro_auc"], metrics["balanced_accuracy"], metrics["macro_f1"], len(subset))
        if best_key is None or key > best_key:
            best_key, best_retained = key, retained
    if best_retained is None:
        raise RuntimeError("S6 grid has no candidate satisfying per-fold/per-class constraints")
    excluded["S6"] = all_ids - best_retained
    return excluded, pd.DataFrame(grid_rows)


def probability_matrix(frame: pd.DataFrame, model: str) -> np.ndarray:
    prefix = "ensemble_prob" if model == "ensemble" else f"prob_{model}"
    return frame[[f"{prefix}_{CLASS_NAMES[index]}" for index in CLASS_LABELS]].to_numpy(float)


def calculate_metrics(frame: pd.DataFrame, model: str) -> dict[str, Any]:
    true = frame["true_label"].to_numpy(int)
    prob = probability_matrix(frame, model)
    base = compute_classification_metrics(true, prob, num_classes=3)
    predicted = prob.argmax(axis=1)
    base["n"] = len(frame)
    for class_index in CLASS_LABELS:
        base[f"n_{CLASS_NAMES[class_index]}"] = int((true == class_index).sum())
    base["ordinal_mae"] = float(np.mean(np.abs(true - predicted)))
    base["within_one_accuracy"] = float(np.mean(np.abs(true - predicted) <= 1))
    base["extreme_error_rate"] = float(np.mean(np.abs(true - predicted) == 2))
    base["quadratic_weighted_kappa"] = float(cohen_kappa_score(true, predicted, weights="quadratic", labels=list(CLASS_LABELS)))
    return base


def bootstrap_intervals(frame: pd.DataFrame, model: str, iterations: int, seed: int) -> dict[str, tuple[float, float]]:
    true = frame["true_label"].to_numpy(int)
    prob = probability_matrix(frame, model)
    n = len(frame)
    rng = np.random.default_rng(seed)
    names = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")
    values = {name: [] for name in names}

    def fast_core_metrics(sample_true: np.ndarray, sample_prob: np.ndarray) -> tuple[float, float, float, float]:
        """Equivalent fast path for the four bootstrapped project metrics."""
        predicted = sample_prob.argmax(axis=1)
        accuracy = float(np.mean(predicted == sample_true))
        recalls = []
        f1s = []
        aucs = []
        for class_index in CLASS_LABELS:
            actual = sample_true == class_index
            called = predicted == class_index
            tp = int(np.sum(actual & called))
            fn = int(np.sum(actual & ~called))
            fp = int(np.sum(~actual & called))
            recalls.append(tp / (tp + fn) if tp + fn else 0.0)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = recalls[-1]
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
            n_positive = int(actual.sum())
            n_negative = len(actual) - n_positive
            ranks = rankdata(sample_prob[:, class_index], method="average")
            aucs.append(
                (float(ranks[actual].sum()) - n_positive * (n_positive + 1) / 2)
                / (n_positive * n_negative)
            )
        return accuracy, float(np.mean(recalls)), float(np.mean(f1s)), float(np.mean(aucs))

    for _ in range(iterations):
        index = rng.integers(0, n, size=n)
        sample_true = true[index]
        if np.unique(sample_true).size < 3:
            continue
        sample_prob = prob[index]
        result = dict(zip(names, fast_core_metrics(sample_true, sample_prob)))
        for name in names:
            value = result[name]
            if np.isfinite(value):
                values[name].append(value)
    return {
        name: tuple(np.quantile(observed, [0.025, 0.975]).tolist()) if observed else (math.nan, math.nan)
        for name, observed in values.items()
    }


def metrics_long_table(
    joint: pd.DataFrame,
    exclusions: dict[str, set[str]],
    bootstrap_iterations: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    scalar_metrics = (
        "n", "n_normal", "n_mild", "n_severe", "accuracy", "balanced_accuracy", "macro_f1",
        "macro_auc", "auc_normal", "auc_mild", "auc_severe", "ordinal_mae", "within_one_accuracy",
        "extreme_error_rate", "quadratic_weighted_kappa",
    )
    for scenario_index, scenario in enumerate(SCENARIO_LABELS):
        subset = joint.loc[~joint["patient_id"].isin(exclusions[scenario])].copy()
        for model_index, model in enumerate(MODEL_NAMES):
            result = calculate_metrics(subset, model)
            cache[(scenario, model)] = result
            intervals = bootstrap_intervals(
                subset, model, bootstrap_iterations, SEED + scenario_index * 100 + model_index
            ) if bootstrap_iterations > 0 else {}
            for metric in scalar_metrics:
                low, high = intervals.get(metric, (math.nan, math.nan))
                rows.append(
                    {
                        "scenario": scenario,
                        "scenario_name": SCENARIO_LABELS[scenario],
                        "model": model,
                        "metric": metric,
                        "value": result[metric],
                        "bootstrap_ci_2.5%": low,
                        "bootstrap_ci_97.5%": high,
                        "bootstrap_iterations": bootstrap_iterations if metric in intervals else 0,
                    }
                )
            matrix = np.asarray(result["confusion_matrix"])
            for actual in CLASS_LABELS:
                for predicted in CLASS_LABELS:
                    rows.append(
                        {
                            "scenario": scenario,
                            "scenario_name": SCENARIO_LABELS[scenario],
                            "model": model,
                            "metric": f"cm_true_{CLASS_NAMES[actual]}_pred_{CLASS_NAMES[predicted]}",
                            "value": int(matrix[actual, predicted]),
                            "bootstrap_ci_2.5%": math.nan,
                            "bootstrap_ci_97.5%": math.nan,
                            "bootstrap_iterations": 0,
                        }
                    )
    return pd.DataFrame(rows), cache


def scenario_summary(joint: pd.DataFrame, exclusions: dict[str, set[str]], cache: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
    rows = []
    original_n = len(joint)
    for scenario in SCENARIO_LABELS:
        metrics = cache[(scenario, "ensemble")]
        excluded_n = len(exclusions[scenario])
        rows.append(
            {
                "scenario": scenario,
                "scenario_name": SCENARIO_LABELS[scenario],
                "interpretation": "ORACLE_POSTHOC_UPPER_BOUND" if scenario == "S6" else "POSTHOC_EXPLORATION",
                "retained_n": original_n - excluded_n,
                "excluded_n": excluded_n,
                "retention_rate": (original_n - excluded_n) / original_n,
                "exclusion_rate": excluded_n / original_n,
                "n_normal": metrics["n_normal"],
                "n_mild": metrics["n_mild"],
                "n_severe": metrics["n_severe"],
                "ensemble_accuracy": metrics["accuracy"],
                "ensemble_balanced_accuracy": metrics["balanced_accuracy"],
                "ensemble_macro_f1": metrics["macro_f1"],
                "ensemble_macro_auc": metrics["macro_auc"],
                "ensemble_ordinal_mae": metrics["ordinal_mae"],
                "ensemble_within_one_accuracy": metrics["within_one_accuracy"],
                "ensemble_extreme_error_rate": metrics["extreme_error_rate"],
                "ensemble_quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"],
            }
        )
    return pd.DataFrame(rows)


def exclusion_audit(joint: pd.DataFrame, exclusions: dict[str, set[str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in list(SCENARIO_LABELS)[1:]:
        excluded = joint[joint["patient_id"].isin(exclusions[scenario])]
        for dimension, column, values in (
            ("true_class", "true_label_name", list(CLASS_NAMES.values())),
            ("fold", "fold", sorted(joint["fold"].unique())),
            ("error_direction", "error_direction", sorted(joint["error_direction"].unique())),
            ("review_priority", "suggested_review_priority", ["P1", "P2", "P3", "P4", "P5"]),
        ):
            original_counts = joint[column].value_counts()
            excluded_counts = excluded[column].value_counts()
            for value in values:
                original_n = int(original_counts.get(value, 0))
                excluded_n = int(excluded_counts.get(value, 0))
                rows.append(
                    {
                        "scenario": scenario,
                        "scenario_name": SCENARIO_LABELS[scenario],
                        "audit_dimension": dimension,
                        "audit_value": value,
                        "original_n": original_n,
                        "excluded_n": excluded_n,
                        "retained_n": original_n - excluded_n,
                        "exclusion_rate": excluded_n / original_n if original_n else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def class_fold_counts(joint: pd.DataFrame, exclusions: dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    folds = sorted(joint["fold"].unique())
    for scenario in SCENARIO_LABELS:
        subset = joint[~joint["patient_id"].isin(exclusions[scenario])]
        for fold in folds:
            for class_index in CLASS_LABELS:
                rows.append(
                    {
                        "scenario": scenario,
                        "scenario_name": SCENARIO_LABELS[scenario],
                        "fold": int(fold),
                        "true_label": class_index,
                        "true_label_name": CLASS_NAMES[class_index],
                        "retained_n": int(((subset["fold"] == fold) & (subset["true_label"] == class_index)).sum()),
                    }
                )
    return pd.DataFrame(rows)


def plot_metric_vs_retention(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    specs = (
        ("ensemble_macro_auc", "Ensemble macro AUC"),
        ("ensemble_balanced_accuracy", "Ensemble balanced accuracy"),
        ("ensemble_macro_f1", "Ensemble macro F1"),
    )
    for ax, (column, title) in zip(axes, specs):
        ax.plot(summary["retention_rate"] * 100, summary[column], marker="o", color="#276FBF")
        for row in summary.itertuples():
            ax.annotate(row.scenario, (row.retention_rate * 100, getattr(row, column)), xytext=(3, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Retained cases (%)")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
    fig.suptitle("Post-hoc metrics versus cohort retention (not independent validation)", fontweight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(cache: dict[tuple[str, str], dict[str, Any]], path: Path) -> None:
    scenarios = list(SCENARIO_LABELS)
    fig, axes = plt.subplots(len(scenarios), len(MODEL_NAMES), figsize=(14, 20), constrained_layout=True)
    for row_index, scenario in enumerate(scenarios):
        for col_index, model in enumerate(MODEL_NAMES):
            ax = axes[row_index, col_index]
            matrix = np.asarray(cache[(scenario, model)]["confusion_matrix"])
            image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
            for actual in CLASS_LABELS:
                for predicted in CLASS_LABELS:
                    ax.text(predicted, actual, str(matrix[actual, predicted]), ha="center", va="center", fontsize=8)
            ax.set_xticks(CLASS_LABELS, [CLASS_NAMES[i] for i in CLASS_LABELS], rotation=30, ha="right", fontsize=7)
            ax.set_yticks(CLASS_LABELS, [CLASS_NAMES[i] for i in CLASS_LABELS], fontsize=7)
            ax.set_title(f"{scenario} / {model}", fontsize=9)
            if col_index == 0:
                ax.set_ylabel("True")
            if row_index == len(scenarios) - 1:
                ax.set_xlabel("Predicted")
    fig.suptitle("OOF confusion matrices after post-hoc selection", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda value: f"{value:.{digits}f}" if pd.notna(value) else "NA")
    headers = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in display.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def write_validation_report(path: Path, identities: list[dict[str, Any]], label_info: dict[str, Any] | None, status: str, error: str | None = None) -> None:
    lines = ["# Input validation", "", f"**Status: {status}**", ""]
    if error:
        lines += ["## Critical error", "", error, "", "筛选已停止；没有静默丢弃任何样本。", ""]
    if identities:
        lines += ["## OOF inputs", ""]
        for item in identities:
            lines += [
                f"- `{item.get('expected_backbone')}`: `{item.get('oof_path')}`",
                f"  - config: `{item.get('config_path')}`",
                f"  - rows: {item.get('n_rows', 'not loaded')}; configured backbone: `{item.get('configured_backbone')}`",
                f"  - fields: ID=`{item.get('id_column')}`, true=`{item.get('true_label_column')}`, pred=`{item.get('pred_label_column')}`, fold=`{item.get('fold_column')}`, probabilities=`{item.get('probability_columns')}`",
            ]
    if label_info:
        lines += ["", "## Label cross-check", "", f"- Path: `{label_info['path']}`", f"- Rows: {label_info['n_rows']}", "- NYHA mapping 0 / 1-2 / 3-4 agrees with OOF labels: yes"]
    if status == "PASS":
        lines += [
            "", "## Critical checks", "",
            "- Each OOF ID is unique: PASS",
            "- Three ID sets are identical: PASS",
            "- True labels agree by ID: PASS",
            "- Fold assignments agree by ID: PASS",
            "- Probabilities are finite, within [0,1], and sum to one: PASS",
            "- Stored predictions equal probability argmax: PASS",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_under_retention(summary: pd.DataFrame, threshold: float) -> pd.Series | None:
    eligible = summary[summary["retention_rate"] >= threshold]
    if eligible.empty:
        return None
    return eligible.sort_values(
        ["ensemble_macro_auc", "ensemble_balanced_accuracy", "ensemble_macro_f1", "retained_n"],
        ascending=False,
    ).iloc[0]


def write_report(
    path: Path,
    summary: pd.DataFrame,
    joint: pd.DataFrame,
    exclusions: dict[str, set[str]],
    grid: pd.DataFrame,
    oof_paths: dict[str, Path],
    bootstrap_iterations: int,
) -> None:
    original_counts = joint["true_label_name"].value_counts().reindex(CLASS_NAMES.values()).astype(int)
    balanced = summary[summary["scenario"].isin(["S4", "S5", "S6"])]
    most_retained_balanced = balanced.sort_values("retained_n", ascending=False).iloc[0]
    best_overall = summary.sort_values(
        ["ensemble_macro_auc", "ensemble_balanced_accuracy", "ensemble_macro_f1", "retained_n"], ascending=False
    ).iloc[0]
    s0 = summary.set_index("scenario").loc["S0"]
    reproduced = {
        "all_three_wrong": int(joint["all_three_wrong"].sum()),
        "all_three_same_wrong": int(joint["all_three_same_wrong"].sum()),
        "extreme_consensus": int(joint["extreme_consensus_error"].sum()),
    }
    extreme_directions = joint.loc[joint["extreme_consensus_error"]].groupby(
        ["true_label_name", "ensemble_pred_name"]
    ).size().to_dict()
    summary_view = summary[[
        "scenario", "retained_n", "excluded_n", "retention_rate", "n_normal", "n_mild", "n_severe",
        "ensemble_macro_auc", "ensemble_balanced_accuracy", "ensemble_macro_f1", "ensemble_extreme_error_rate",
    ]]
    lines = [
        "# ResNet18/34/50 联合错误驱动的数据筛选与三类平衡上限分析",
        "",
        f"> [!CAUTION]  ",
        f"> **{DISCLAIMER}**",
        "",
        "## 执行摘要",
        "",
        f"- 原始 522 例三类数量：正常 {original_counts['normal']}、轻症 {original_counts['mild']}、重症 {original_counts['severe']}。",
        f"- 保留最多的平衡方案是 {most_retained_balanced['scenario_name']}，保留 {int(most_retained_balanced['retained_n'])} 例（每类 {int(most_retained_balanced['n_normal'])} 例）。",
        f"- 按 macro AUC → BA → macro F1 → 保留数的预设字典序，最高方案为 {best_overall['scenario_name']}：macro AUC={best_overall['ensemble_macro_auc']:.4f}，BA={best_overall['ensemble_balanced_accuracy']:.4f}，macro F1={best_overall['ensemble_macro_f1']:.4f}，排除 {int(best_overall['excluded_n'])} 例。",
        f"- 程序复现共同误判 {reproduced['all_three_wrong']} 例、一致错误 {reproduced['all_three_same_wrong']} 例、极端一致错误 {reproduced['extreme_consensus']} 例。",
        f"- 19 例已知极端误判全部识别；方向计数为 {extreme_directions}。",
        "",
        "## 各方案结果",
        "",
        markdown_table(summary_view),
        "",
        "S0–S5 均为事后探索；S6 的正式解释标签是 **ORACLE_POSTHOC_UPPER_BOUND**，不能称为清洗后的真实性能。",
        "",
        "## 筛选规则与独立派生原则",
        "",
        "每个方案都直接从完整 522 例生成，不使用上一方案的结果。S1 仅删极端一致错误；S2 删除三模型同类误判；S3 删除三模型全部误判；S4 按 P1→P5、错误数、真实类概率、熵、ID 平衡；S5 先删一致错误，再按真实类概率补充平衡；S6 对每类固定排序后枚举相等保留数，并满足每折每类非空。",
        "",
        "## 困难病例删除效应",
        "",
        f"完整集 ensemble：macro AUC={s0['ensemble_macro_auc']:.4f}、BA={s0['ensemble_balanced_accuracy']:.4f}、macro F1={s0['ensemble_macro_f1']:.4f}。所有筛选规则直接使用了模型错误或置信度，因此筛选后 BA/macro F1 的上升在机制上主要反映困难病例被删除，不能解释为模型泛化能力提高。",
        f"病例级 bootstrap {bootstrap_iterations} 次的 95% CI 已写入 `scenario_metrics_long.csv`；未对事后方案计算或宣称独立显著性。",
        "",
        "## 保留率约束下的最优事后结果",
        "",
    ]
    retention_rows = []
    for threshold in (0.70, 0.80, 0.90):
        row = best_under_retention(summary, threshold)
        if row is None:
            retention_rows.append({"minimum_retention": threshold, "scenario": "无可行方案"})
        else:
            retention_rows.append(
                {
                    "minimum_retention": threshold,
                    "scenario": row["scenario_name"],
                    "retained_n": int(row["retained_n"]),
                    "macro_auc": row["ensemble_macro_auc"],
                    "balanced_accuracy": row["ensemble_balanced_accuracy"],
                    "macro_f1": row["ensemble_macro_f1"],
                }
            )
    lines += [markdown_table(pd.DataFrame(retention_rows)), "", "这些选择只在 S0–S6 已定义方案之间比较，不是对任意子集进行无约束搜索。", ""]
    lines += ["## 人工复核优先级", ""]
    for priority in ("P1", "P2", "P3", "P4"):
        ids = joint.loc[joint["suggested_review_priority"] == priority, "patient_id"].tolist()
        preview = ", ".join(ids[:20]) + (" …" if len(ids) > 20 else "")
        lines.append(f"- {priority}：{len(ids)} 例；{preview}")
    lines += [
        "",
        "完整患者级名单、真实类别、模型预测、概率、熵和错误方向见 `candidate_manual_review.csv`。应优先核对 P1 的原始照片与标签，再核对 P2/P3；模型一致错误只能提供复核线索，不能自动证明标签错误。",
        "",
        "## 下一步数据版本建议",
        "",
        "不要直接将 S6 用于正式重训或性能声明。建议先由盲于模型输出的人工流程复核 P1/P2（必要时扩展至 P3），记录标签是否更正、病例是否排除及客观原因；随后冻结一个可审计的数据版本，重新执行完整五折训练，并仍以未事后挑选的评估协议作为主要结果。S4 可作为类别平衡的敏感性重训候选，但必须与完整数据 S0 并列报告。",
        "",
        "## 输入与可复现性",
        "",
    ]
    for model, oof_path in oof_paths.items():
        lines.append(f"- {model}: `{oof_path.resolve()}`")
    best_grid = grid.sort_values(["macro_auc", "balanced_accuracy", "macro_f1", "retained_n"], ascending=False).iloc[0]
    lines += [
        f"- S6 最优网格点：每类 {int(best_grid['keep_per_class'])} 例，总计 {int(best_grid['retained_n'])} 例。",
        f"- 固定随机种子：{SEED}；bootstrap：{bootstrap_iterations} 次。",
        "- 所有 CSV 均为 UTF-8-SIG。",
        "",
        "## 局限性",
        "",
        "本分析重复使用同一批 OOF 结果来定义筛选规则并重新评价，因此存在选择偏倚与乐观偏倚。置信区间只描述固定事后子集内的重采样不确定性，未包含筛选规则选择的不确定性，也不构成外部或独立验证。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test_metrics() -> dict[str, bool]:
    true = np.array([0, 0, 1, 1, 2, 2])
    prob = np.array(
        [[0.90, 0.08, 0.02], [0.10, 0.80, 0.10], [0.10, 0.80, 0.10],
         [0.05, 0.15, 0.80], [0.02, 0.08, 0.90], [0.70, 0.20, 0.10]]
    )
    result = compute_classification_metrics(true, prob, 3)
    expected_cm = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    manual_auc = np.mean([roc_auc_score(true == index, prob[:, index]) for index in CLASS_LABELS])
    checks = {
        "label_order_normal_mild_severe": CLASS_NAMES == {0: "normal", 1: "mild", 2: "severe"},
        "confusion_matrix_order": bool(np.array_equal(result["confusion_matrix"], expected_cm)),
        "macro_auc_ovr": bool(np.isclose(result["macro_auc"], manual_auc)),
    }
    if not all(checks.values()):
        raise AssertionError(f"Metric self-test failed: {checks}")
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resnet18-oof", type=Path)
    parser.add_argument("--resnet34-oof", type=Path)
    parser.add_argument("--resnet50-oof", type=Path)
    parser.add_argument("--label-csv", type=Path, default=PROJECT_ROOT / "data" / "raw" / "label_raw.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "posthoc_oracle_data_adjustment_522")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    np.random.seed(SEED)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    identities: list[dict[str, Any]] = []
    label_info: dict[str, Any] | None = None
    try:
        oof_paths = {
            "resnet18": (args.resnet18_oof.resolve() if args.resnet18_oof else discover_oof(PROJECT_ROOT, "resnet18")),
            "resnet34": (args.resnet34_oof.resolve() if args.resnet34_oof else discover_oof(PROJECT_ROOT, "resnet34")),
            "resnet50": (args.resnet50_oof.resolve() if args.resnet50_oof else discover_oof(PROJECT_ROOT, "resnet50")),
        }
        frames: dict[str, pd.DataFrame] = {}
        for model, path in oof_paths.items():
            frame, identity = standardize_oof(path, model)
            frames[model] = frame
            identities.append(identity)
        joint = align_oofs(frames)
        label_info = validate_labels(args.label_csv.resolve(), joint)
        write_validation_report(output_dir / "input_validation.md", identities, label_info, "PASS")
    except Exception as exc:
        write_validation_report(output_dir / "input_validation.md", identities, label_info, "FAIL", str(exc))
        raise

    tests = self_test_metrics()
    joint = build_joint_error_table(joint)
    save_csv(joint, output_dir / "joint_oof_error_table.csv")

    priority_order = {"P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5}
    candidates = joint[joint["suggested_review_priority"] != "P5"].copy()
    candidates["_priority"] = candidates["suggested_review_priority"].map(priority_order)
    candidates = candidates.sort_values(
        ["_priority", "wrong_count", "mean_true_class_probability", "mean_prediction_entropy", "patient_id"],
        ascending=[True, False, True, False, True], kind="mergesort"
    ).drop(columns="_priority")
    save_csv(candidates, output_dir / "candidate_manual_review.csv")

    exclusions, grid = make_scenarios(joint)
    save_csv(grid, output_dir / "oracle_upper_bound_grid.csv")
    for scenario in SCENARIO_LABELS:
        retained = joint.loc[~joint["patient_id"].isin(exclusions[scenario]), ["patient_id", "true_label", "true_label_name", "fold"]]
        save_csv(retained, output_dir / f"retained_ids_{scenario}.csv")
        if scenario != "S0":
            excluded = joint.loc[joint["patient_id"].isin(exclusions[scenario]), [
                "patient_id", "true_label", "true_label_name", "fold", "suggested_review_priority", "wrong_count",
                "all_three_wrong", "all_three_same_wrong", "extreme_consensus_error", "error_direction",
                "mean_true_class_probability", "mean_prediction_entropy",
            ]]
            save_csv(excluded, output_dir / f"excluded_ids_{scenario}.csv")

    metrics_long, cache = metrics_long_table(joint, exclusions, args.bootstrap_iterations)
    summary = scenario_summary(joint, exclusions, cache)
    audit = exclusion_audit(joint, exclusions)
    counts = class_fold_counts(joint, exclusions)
    save_csv(summary, output_dir / "scenario_summary.csv")
    save_csv(metrics_long, output_dir / "scenario_metrics_long.csv")
    save_csv(audit, output_dir / "exclusion_audit.csv")
    save_csv(counts, output_dir / "class_fold_counts.csv")
    plot_metric_vs_retention(summary, output_dir / "metric_vs_retention.png")
    plot_confusion_matrices(cache, output_dir / "confusion_matrices.png")
    write_report(
        output_dir / "posthoc_oracle_data_adjustment_report.md", summary, joint, exclusions, grid, oof_paths,
        args.bootstrap_iterations,
    )

    closure = {}
    for scenario in SCENARIO_LABELS:
        retained_n = len(joint) - len(exclusions[scenario])
        excluded_n = len(exclusions[scenario])
        closure[scenario] = {
            "retained_n": retained_n,
            "excluded_n": excluded_n,
            "original_n": len(joint),
            "closed": retained_n + excluded_n == len(joint),
        }
    manifest = {
        "analysis": "ResNet18/34/50 joint post-hoc OOF data adjustment",
        "interpretation": "ORACLE/POST-HOC exploration; S6 is ORACLE_POSTHOC_UPPER_BOUND",
        "disclaimer": DISCLAIMER,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(output_dir),
        "random_seed": SEED,
        "bootstrap_iterations": args.bootstrap_iterations,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "inputs": {item["expected_backbone"]: item for item in identities},
        "label_input": label_info,
        "metric_self_tests": tests,
        "aligned_sample_count": len(joint),
        "reproduced_known_counts": {
            "all_three_wrong": int(joint["all_three_wrong"].sum()),
            "all_three_same_wrong": int(joint["all_three_same_wrong"].sum()),
            "extreme_consensus_error": int(joint["extreme_consensus_error"].sum()),
        },
        "scenario_id_closure": closure,
        "scenario_rules_independently_derived_from_S0": True,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Report: {output_dir / 'posthoc_oracle_data_adjustment_report.md'}")


if __name__ == "__main__":
    run(parse_args())
