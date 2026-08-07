from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import fold_z_features
from .metrics_utils import cluster_bootstrap_metrics


MODEL_SPECS: dict[str, dict[str, list[str]]] = {
    "META-D": {"numeric": [], "categorical": ["camera_model"]},
    "META-L": {"numeric": ["brightness_device_z", "log2_iso_device_z", "log2_exposure_device_z", "exposure_bias_ev"], "categorical": ["flash"]},
    "META-DL": {"numeric": ["brightness_device_z", "log2_iso_device_z", "log2_exposure_device_z", "exposure_bias_ev"], "categorical": ["camera_model", "flash"]},
    "META-T": {"numeric": [], "categorical": ["capture_year", "capture_month", "capture_year_month", "capture_hour_bin"]},
    "META-ALL": {
        "numeric": ["image_width", "image_height", "megapixels", "brightness_device_z", "log2_iso_device_z", "log2_exposure_device_z", "exposure_bias_ev"],
        "categorical": ["camera_make", "camera_model", "software", "flash", "metering_mode", "white_balance", "exposure_mode", "capture_year", "capture_month", "capture_year_month", "capture_hour_bin"],
    },
}


def _onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _available_features(frame: pd.DataFrame, spec: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    numeric = [c for c in spec["numeric"] if c in frame.columns and pd.to_numeric(frame[c], errors="coerce").notna().sum() > 0]
    categorical = [c for c in spec["categorical"] if c in frame.columns and frame[c].notna().sum() > 0 and frame[c].nunique(dropna=True) > 1]
    return numeric, categorical


def _make_pipeline(numeric: list[str], categorical: list[str], C: float, seed: int) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("onehot", _onehot())]), categorical))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        [
            ("preprocess", pre),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=float(C),
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def _choose_c(
    train: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    c_grid: tuple[float, ...],
    seed: int,
    splits: int,
) -> float:
    if len(c_grid) == 1:
        return float(c_grid[0])
    group_count = len(np.unique(groups))
    if group_count < splits or len(np.unique(y)) < 2:
        return float(c_grid[0])
    cv = StratifiedGroupKFold(n_splits=min(splits, group_count), shuffle=True, random_state=seed)
    scores: dict[float, list[float]] = {float(c): [] for c in c_grid}
    for tr_idx, va_idx in cv.split(train, y, groups):
        if len(np.unique(y[va_idx])) < 2 or len(np.unique(y[tr_idx])) < 2:
            continue
        for c in c_grid:
            pipe = _make_pipeline(numeric, categorical, c, seed)
            pipe.fit(train.iloc[tr_idx][numeric + categorical], y[tr_idx])
            prob = pipe.predict_proba(train.iloc[va_idx][numeric + categorical])[:, 1]
            scores[float(c)].append(float(roc_auc_score(y[va_idx], prob)))
    means = {c: (np.mean(v) if v else -np.inf) for c, v in scores.items()}
    return float(max(means, key=means.get))


def run_one_model_oof(
    master: pd.DataFrame,
    model_name: str,
    *,
    y_override: pd.Series | None = None,
    seed: int,
    c_grid: tuple[float, ...],
    inner_cv_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coef_rows = []
    labels = master["binary_label"].astype(int) if y_override is None else y_override.astype(int)
    for fold in sorted(master["fold"].astype(int).unique()):
        fold_frame, _ = fold_z_features(master, int(fold))
        numeric, categorical = _available_features(fold_frame, MODEL_SPECS[model_name])
        features = numeric + categorical
        if not features:
            raise ValueError(f"{model_name} has no available features")
        train_idx = fold_frame.index[fold_frame["fold"].astype(int) != int(fold)]
        test_idx = fold_frame.index[fold_frame["fold"].astype(int) == int(fold)]
        y_train = labels.loc[train_idx].to_numpy(dtype=int)
        groups = fold_frame.loc[train_idx, "patient_group_id"].astype(str).to_numpy()
        chosen_c = _choose_c(fold_frame.loc[train_idx], y_train, groups, numeric, categorical, c_grid, seed + int(fold), inner_cv_splits)
        pipe = _make_pipeline(numeric, categorical, chosen_c, seed + int(fold))
        pipe.fit(fold_frame.loc[train_idx, features], y_train)
        prob = pipe.predict_proba(fold_frame.loc[test_idx, features])[:, 1]
        pred = (prob >= 0.5).astype(int)
        for idx, p, pred_label in zip(test_idx, prob, pred):
            rows.append(
                {
                    "sample_id": master.loc[idx, "sample_id"],
                    "patient_group_id": master.loc[idx, "patient_group_id"],
                    "fold": int(fold),
                    "binary_label": int(labels.loc[idx]),
                    "model_name": model_name,
                    "probability_patient": float(p),
                    "predicted_label": int(pred_label),
                    "correct": int(pred_label == int(labels.loc[idx])),
                    "selected_C": float(chosen_c),
                    "features_numeric": ";".join(numeric),
                    "features_categorical": ";".join(categorical),
                }
            )
        try:
            names = pipe.named_steps["preprocess"].get_feature_names_out()
            coefs = pipe.named_steps["model"].coef_[0]
            for name, coef in zip(names, coefs):
                coef_rows.append({"model_name": model_name, "fold": int(fold), "selected_C": float(chosen_c), "feature": str(name), "coefficient": float(coef)})
        except Exception as exc:
            coef_rows.append({"model_name": model_name, "fold": int(fold), "selected_C": float(chosen_c), "feature": "coefficient_export_failed", "coefficient": np.nan, "error": str(exc)})
    oof = pd.DataFrame(rows).sort_values(["model_name", "fold", "sample_id"]).reset_index(drop=True)
    coefs = pd.DataFrame(coef_rows)
    return oof, coefs


def _prepare_permutation_designs(master: pd.DataFrame, model_name: str, *, C: float, seed: int) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for fold in sorted(master["fold"].astype(int).unique()):
        fold_frame, _ = fold_z_features(master, int(fold))
        numeric, categorical = _available_features(fold_frame, MODEL_SPECS[model_name])
        features = numeric + categorical
        if not features:
            raise ValueError(f"{model_name} has no available features")
        train_idx = fold_frame.index[fold_frame["fold"].astype(int) != int(fold)]
        test_idx = fold_frame.index[fold_frame["fold"].astype(int) == int(fold)]
        pipe = _make_pipeline(numeric, categorical, C, seed + int(fold))
        pre = pipe.named_steps["preprocess"]
        pre.fit(fold_frame.loc[train_idx, features])
        prepared.append(
            {
                "fold": int(fold),
                "train_idx": train_idx.to_numpy(),
                "test_idx": test_idx.to_numpy(),
                "x_train": pre.transform(fold_frame.loc[train_idx, features]),
                "x_test": pre.transform(fold_frame.loc[test_idx, features]),
                "C": float(C),
            }
        )
    return prepared


def _permutation_auc_fast(master: pd.DataFrame, y_perm: pd.Series, prepared: list[dict[str, Any]], *, seed: int) -> float:
    probs = np.zeros(len(master), dtype=float)
    labels = y_perm.to_numpy(dtype=int)
    for item in prepared:
        train_idx = item["train_idx"]
        test_idx = item["test_idx"]
        model = LogisticRegression(
            penalty="l2",
            C=float(item["C"]),
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=seed + int(item["fold"]),
        )
        model.fit(item["x_train"], labels[train_idx])
        probs[test_idx] = model.predict_proba(item["x_test"])[:, 1]
    if np.unique(labels).size < 2:
        raise ValueError("permuted labels contain one class")
    return float(roc_auc_score(labels, probs))


def _permuted_labels(master: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    group_labels = master.groupby("patient_group_id")["binary_label"].nunique()
    if group_labels.gt(1).any():
        raise ValueError("patient_group contains mixed binary labels; group-level permutation refused")
    group_to_label = master.groupby("patient_group_id")["binary_label"].first()
    shuffled = pd.Series(rng.permutation(group_to_label.to_numpy()), index=group_to_label.index)
    return master["patient_group_id"].map(shuffled).astype(int)


def run_metadata_models(
    master: pd.DataFrame,
    out_dir: Path,
    *,
    bootstrap_repeats: int,
    permutation_repeats: int,
    seed: int,
    c_grid: tuple[float, ...],
    inner_cv_splits: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_oof = []
    all_coef = []
    metric_rows = []
    metrics_json: dict[str, Any] = {}
    for model_name in MODEL_SPECS:
        oof, coefs = run_one_model_oof(master, model_name, seed=seed, c_grid=c_grid, inner_cv_splits=inner_cv_splits)
        all_oof.append(oof)
        all_coef.append(coefs)
        boot = cluster_bootstrap_metrics(oof, repeats=bootstrap_repeats, seed=seed)
        row = {"model_name": model_name, **boot["point_estimates"]}
        for metric, ci in boot["ci95"].items():
            row[f"{metric}_ci95_low"] = ci[0]
            row[f"{metric}_ci95_high"] = ci[1]
        row["bootstrap_valid_iterations"] = boot["valid_iterations"]
        metric_rows.append(row)
        metrics_json[model_name] = boot

    oof_all = pd.concat(all_oof, ignore_index=True)
    oof_all.to_csv(out_dir / "metadata_only_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "metadata_only_metrics.csv", index=False, encoding="utf-8-sig")
    (out_dir / "metadata_only_metrics.json").write_text(__import__("json").dumps(metrics_json, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.concat(all_coef, ignore_index=True).to_csv(out_dir / "metadata_model_coefficients.csv", index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(seed + 9000)
    perm_rows = []
    observed_auc = dict(zip(metrics_df["model_name"], metrics_df["roc_auc"]))
    for model_name in MODEL_SPECS:
        prepared = _prepare_permutation_designs(master, model_name, C=float(c_grid[0]), seed=seed)
        aucs: list[float] = []
        failed = 0
        for _ in range(int(permutation_repeats)):
            y_perm = _permuted_labels(master, rng)
            try:
                aucs.append(_permutation_auc_fast(master, y_perm, prepared, seed=seed))
            except Exception:
                failed += 1
        obs = float(observed_auc[model_name])
        arr = np.asarray(aucs, dtype=float)
        p_value = float((1 + np.sum(arr >= obs)) / (len(arr) + 1)) if len(arr) else np.nan
        perm_rows.append(
            {
                "model_name": model_name,
                "observed_auc": obs,
                "permutation_auc_mean": float(np.mean(arr)) if len(arr) else np.nan,
                "permutation_auc_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
                "empirical_p_value": p_value,
                "permutation_repeats": int(permutation_repeats),
                "valid_permutations": int(len(arr)),
                "failed_permutations": int(failed),
                "permutation_unit": "patient_group_id",
            }
        )
    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(out_dir / "metadata_permutation_results.csv", index=False, encoding="utf-8-sig")
    return {"oof": oof_all, "metrics": metrics_df, "permutation": perm_df}
