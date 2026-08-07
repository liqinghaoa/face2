from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .metrics_utils import cluster_bootstrap_metrics, compute_metrics


RGB_TARGETS = ["rgb_oof_probability_patient", "rgb_oof_logit", "rgb_oof_residual", "rgb_oof_error"]
META_CONT = ["brightness_device_z", "log2_iso_device_z", "log2_exposure_device_z"]


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None, int]:
    df = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(df) < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return None, None, int(len(df))
    r, p = stats.spearmanr(df["x"], df["y"])
    return float(r), float(p), int(len(df))


def run_rgb_dependence(master: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_rows = []
    strata = [("overall", "all", master)]
    for label, group in master.groupby("binary_label"):
        strata.append(("label", f"binary_label={int(label)}", group))
    for device, group in master.groupby("camera_model"):
        strata.append(("camera_model", str(device), group))
    for scope, level, frame in strata:
        for target in RGB_TARGETS:
            for var in META_CONT:
                if target in frame.columns and var in frame.columns:
                    r, p, n = _spearman(frame[target], frame[var])
                    corr_rows.append({"scope": scope, "level": level, "target": target, "metadata_variable": var, "spearman_r": r, "p_value": p, "n": n})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(out_dir / "rgb_metadata_correlations.csv", index=False, encoding="utf-8-sig")

    adjusted = _adjusted_regression(master)
    adjusted.to_csv(out_dir / "rgb_metadata_adjusted_regression.csv", index=False, encoding="utf-8-sig")
    error = _error_risk_regression(master)
    error.to_csv(out_dir / "rgb_error_risk_regression.csv", index=False, encoding="utf-8-sig")
    rgb_metrics = compute_metrics(master["binary_label"], master["rgb_oof_probability_patient"])
    return {"correlations": corr, "adjusted": adjusted, "error": error, "rgb_metrics": rgb_metrics}


def _design(master: pd.DataFrame, include_label: bool) -> tuple[pd.DataFrame, list[str]]:
    cols_num = [c for c in META_CONT if c in master.columns]
    x_parts = []
    if include_label:
        x_parts.append(master[["binary_label"]].astype(float))
    if cols_num:
        num = master[cols_num].apply(pd.to_numeric, errors="coerce")
        x_parts.append(pd.DataFrame(StandardScaler().fit_transform(num.fillna(num.median())), columns=cols_num, index=master.index))
    try:
        enc = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)
    arr = enc.fit_transform(master[["camera_model"]].fillna("missing").astype(str))
    names = [f"camera_model={n.split('_', 1)[-1]}" for n in enc.get_feature_names_out(["camera_model"])]
    x_parts.append(pd.DataFrame(arr, columns=names, index=master.index))
    if "flash" in master.columns and master["flash"].nunique(dropna=True) > 1:
        try:
            enc2 = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        except TypeError:
            enc2 = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)
        arr2 = enc2.fit_transform(master[["flash"]].fillna("missing").astype(str))
        names2 = [f"flash={n.split('_', 1)[-1]}" for n in enc2.get_feature_names_out(["flash"])]
        x_parts.append(pd.DataFrame(arr2, columns=names2, index=master.index))
    if "exposure_bias_ev" in master.columns:
        x_parts.append(master[["exposure_bias_ev"]].apply(pd.to_numeric, errors="coerce").fillna(master["exposure_bias_ev"].median()))
    x = pd.concat(x_parts, axis=1)
    return x, list(x.columns)


def _adjusted_regression(master: pd.DataFrame) -> pd.DataFrame:
    try:
        import statsmodels.api as sm

        x, _ = _design(master, include_label=True)
        x = sm.add_constant(x, has_constant="add")
        y = master["rgb_oof_logit"].astype(float)
        model = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": master["patient_group_id"].astype(str)})
        ci = model.conf_int()
        rows = []
        for term in model.params.index:
            rows.append({"term": term, "coefficient": float(model.params[term]), "ci95_low": float(ci.loc[term, 0]), "ci95_high": float(ci.loc[term, 1]), "p_value": float(model.pvalues[term]), "standardized_effect": float(model.params[term])})
        return pd.DataFrame(rows)
    except Exception as exc:
        return pd.DataFrame([{"term": "model_failed", "error": f"{type(exc).__name__}: {exc}"}])


def _error_risk_regression(master: pd.DataFrame) -> pd.DataFrame:
    try:
        import statsmodels.api as sm

        x, _ = _design(master, include_label=False)
        x = sm.add_constant(x, has_constant="add")
        y = master["rgb_oof_error"].astype(int)
        model = sm.Logit(y, x).fit(disp=False, maxiter=200)
        ci = model.conf_int()
        rows = []
        for term in model.params.index:
            rows.append({"term": term, "coefficient": float(model.params[term]), "or": float(np.exp(model.params[term])), "ci95_low": float(np.exp(ci.loc[term, 0])), "ci95_high": float(np.exp(ci.loc[term, 1])), "p_value": float(model.pvalues[term])})
        return pd.DataFrame(rows)
    except Exception as exc:
        return pd.DataFrame([{"term": "model_failed", "error": f"{type(exc).__name__}: {exc}"}])
