from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from lighting_confounding_stage1.metrics_utils import cluster_bootstrap_smd, smd

from .full500_pipeline import _fit_model, _z
from .r2_bootstrap import cluster_bootstrap_spearman


TARGET_METRICS = ["skin_y_median", "skin_y_p05", "dark_contrast_score", "cheek_relative_difference"]
EXIF_FEATURES = ["log2_exposure_time", "log2_iso", "brightness_value"]


def _as_seconds(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, str):
        text = value.strip()
        if "/" in text:
            a, b = text.split("/", 1)
            return float(a) / float(b)
        if text == "":
            return np.nan
        return float(text)
    return float(value)


def build_exif_frame(stage1: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    available = set(stage1.columns)
    mapping = {
        "sample_id": "sample_id",
        "exposure_time_seconds": "exposure_time_seconds" if "exposure_time_seconds" in available else "曝光时间(s)",
        "iso": "iso" if "iso" in available else "ISO",
        "brightness_value": "brightness_value_apex" if "brightness_value_apex" in available else "亮度值(APEX)",
    }
    cols = [mapping["sample_id"], mapping["exposure_time_seconds"], mapping["iso"], mapping["brightness_value"]]
    exif = stage1[cols].copy()
    exif.columns = ["sample_id", "exposure_time_seconds_raw", "iso_raw", "brightness_value_raw"]
    exif["sample_id"] = exif["sample_id"].astype(str)
    exif["exposure_time_seconds"] = exif["exposure_time_seconds_raw"].map(_as_seconds)
    exif["iso_value"] = pd.to_numeric(exif["iso_raw"], errors="coerce")
    exif["brightness_value"] = pd.to_numeric(exif["brightness_value_raw"], errors="coerce")
    exif["log2_exposure_time"] = np.log2(exif["exposure_time_seconds"].where(exif["exposure_time_seconds"] > 0))
    exif["log2_iso"] = np.log2(exif["iso_value"].where(exif["iso_value"] > 0))
    exif = features[["sample_id", "patient_group_id", "fold", "binary_label", "rgb_oof_probability_patient", "rgb_oof_logit", "rgb_oof_error", "rgb_oof_brier_contribution", "rgb_oof_residual", *TARGET_METRICS]].merge(exif, on="sample_id", how="left")
    audit_rows = []
    for col in ["exposure_time_seconds", "iso_value", "brightness_value", "log2_exposure_time", "log2_iso"]:
        x = pd.to_numeric(exif[col], errors="coerce")
        audit_rows.append({"field": col, "non_missing": int(x.notna().sum()), "missing": int(x.isna().sum()), "finite": int(np.isfinite(x.dropna()).sum()), "unique_non_missing": int(x.nunique(dropna=True))})
    field_mapping = {
        "source_table": "stage1_master_500.csv",
        "sample_id": mapping["sample_id"],
        "ExposureTime": mapping["exposure_time_seconds"],
        "ISOSpeedRatings": mapping["iso"],
        "BrightnessValue": mapping["brightness_value"],
        "derived_features": EXIF_FEATURES,
        "forbidden_fields_not_used": ["binary_label", "rgb_oof_probability_patient", "camera_model", "capture_time", "capture_year_month"],
        "ridge_alpha": 1.0,
    }
    return exif, field_mapping, pd.DataFrame(audit_rows)


def _prepare_fold_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float]]:
    medians: dict[str, float] = {}
    x_train = train[EXIF_FEATURES].copy()
    x_test = test[EXIF_FEATURES].copy()
    names = EXIF_FEATURES.copy()
    for col in EXIF_FEATURES:
        non_missing = pd.to_numeric(x_train[col], errors="coerce").dropna()
        if len(non_missing) < 20 or non_missing.nunique() < 2:
            raise ValueError(f"EXIF field {col} has insufficient train-fold data")
        med = float(non_missing.median())
        medians[col] = med
        x_train[f"{col}_missing"] = x_train[col].isna().astype(float)
        x_test[f"{col}_missing"] = x_test[col].isna().astype(float)
        x_train[col] = pd.to_numeric(x_train[col], errors="coerce").fillna(med)
        x_test[col] = pd.to_numeric(x_test[col], errors="coerce").fillna(med)
        names.append(f"{col}_missing")
    scaler = StandardScaler()
    train_arr = scaler.fit_transform(x_train[names])
    test_arr = scaler.transform(x_test[names])
    return train_arr, test_arr, names, medians


def crossfit_decomposition(exif: pd.DataFrame, out_dir: Path, *, alpha: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exif_output_cols = ["exposure_time_seconds", "iso_value", *EXIF_FEATURES]
    exif_output_cols = list(dict.fromkeys(exif_output_cols))
    pred = exif[["sample_id", "patient_group_id", "fold", "binary_label", *exif_output_cols, "rgb_oof_probability_patient", "rgb_oof_logit", "rgb_oof_error", "rgb_oof_brier_contribution", "rgb_oof_residual"]].copy()
    for col in EXIF_FEATURES:
        pred[f"{col}_missing"] = exif[col].isna().astype(int)
    coef_rows = []
    fold_rows = []
    for target in TARGET_METRICS:
        pred[f"observed_{target}"] = exif[target].astype(float)
        pred[f"predicted_{target}"] = np.nan
        for fold in sorted(exif["fold"].astype(int).unique()):
            train = exif.loc[exif["fold"].astype(int) != fold].copy()
            test = exif.loc[exif["fold"].astype(int) == fold].copy()
            x_train, x_test, names, _ = _prepare_fold_features(train, test)
            y_train = train[target].astype(float).to_numpy()
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            pred.loc[test.index, f"predicted_{target}"] = y_pred
            for name, coef in zip(names, model.coef_):
                coef_rows.append({"target_metric": target, "fold": int(fold), "feature_name": name, "coefficient_standardized": float(coef), "intercept": float(model.intercept_), "train_n": int(len(train)), "test_n": int(len(test)), "ridge_alpha": alpha})
            fold_rows.append({"target_metric": target, "fold": int(fold), **_performance(test[target].astype(float).to_numpy(), y_pred)})
        pred[f"residual_{target}"] = pred[f"observed_{target}"].astype(float) - pred[f"predicted_{target}"].astype(float)
        max_err = float((pred[f"observed_{target}"] - pred[f"predicted_{target}"] - pred[f"residual_{target}"]).abs().max())
        if max_err > 1e-12:
            raise ValueError(f"decomposition identity failed for {target}: {max_err}")
    perf_rows = []
    for target in TARGET_METRICS:
        perf_rows.append({"target_metric": target, **_performance(pred[f"observed_{target}"].astype(float).to_numpy(), pred[f"predicted_{target}"].astype(float).to_numpy()), "residual_predicted_spearman": _spearman(pred[f"residual_{target}"], pred[f"predicted_{target}"])})
    return pred, pd.DataFrame(perf_rows), pd.DataFrame(fold_rows), pd.DataFrame(coef_rows)


def _spearman(a: Any, b: Any) -> float | None:
    x = pd.to_numeric(pd.Series(a), errors="coerce")
    y = pd.to_numeric(pd.Series(b), errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return None
    r, _ = stats.spearmanr(x[ok], y[ok])
    return float(r) if np.isfinite(r) else None


def _performance(y: np.ndarray, yhat: np.ndarray) -> dict[str, float | None]:
    resid = y - yhat
    pearson = stats.pearsonr(y, yhat).statistic if len(y) >= 3 and np.std(yhat) > 0 else np.nan
    spearman = stats.spearmanr(y, yhat).statistic if len(y) >= 3 and len(np.unique(yhat)) > 1 else np.nan
    return {
        "r2": float(r2_score(y, yhat)),
        "mae": float(mean_absolute_error(y, yhat)),
        "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
        "spearman_r": float(spearman) if np.isfinite(spearman) else None,
        "pearson_r": float(pearson) if np.isfinite(pearson) else None,
        "observed_mean": float(np.mean(y)),
        "observed_sd": float(np.std(y, ddof=1)),
        "predicted_mean": float(np.mean(yhat)),
        "predicted_sd": float(np.std(yhat, ddof=1)),
        "residual_mean": float(np.mean(resid)),
        "residual_sd": float(np.std(resid, ddof=1)),
    }


def component_label_analysis(pred: pd.DataFrame, out_dir: Path, *, iterations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    desc_rows = []
    assoc_rows = []
    for comp_type in ["predicted", "residual"]:
        for target in TARGET_METRICS:
            col = f"{comp_type}_{target}"
            x = pd.to_numeric(pred[col], errors="coerce")
            c = x[pred["binary_label"] == 0].dropna()
            p = x[pred["binary_label"] == 1].dropna()
            desc_rows.append({"component_type": comp_type, "target_metric": target, "component": col, "control_mean": float(c.mean()), "control_sd": float(c.std(ddof=1)), "control_median": float(c.median()), "control_iqr": float(c.quantile(0.75) - c.quantile(0.25)), "patient_mean": float(p.mean()), "patient_sd": float(p.std(ddof=1)), "patient_median": float(p.median()), "patient_iqr": float(p.quantile(0.75) - p.quantile(0.25))})
            try:
                auc = float(roc_auc_score(pred["binary_label"].astype(int), x))
            except Exception:
                auc = np.nan
            ci_low, ci_high, valid = cluster_bootstrap_smd(pred[["patient_group_id", "binary_label", col]].rename(columns={col: "value"}), value_col="value", repeats=iterations, seed=seed)
            try:
                mw_p = float(stats.mannwhitneyu(c, p, alternative="two-sided").pvalue)
            except Exception:
                mw_p = np.nan
            work = pred[["patient_group_id", "binary_label", col]].copy()
            work["z_component"] = _z(work[col])
            reg = _fit_model(work, "binary_label", ["z_component"], "logit")
            zrow = reg[reg["term"] == "z_component"].head(1)
            assoc_rows.append({"component_type": comp_type, "target_metric": target, "component": col, "smd_patient_minus_control": smd(c, p), "smd_ci95_low": ci_low, "smd_ci95_high": ci_high, "bootstrap_valid": valid, "mannwhitney_p": mw_p, "label_auc": auc, "label_or_per_1sd": zrow.iloc[0].get("or") if not zrow.empty else None, "label_or_ci95_low": zrow.iloc[0].get("or_ci95_low") if not zrow.empty else None, "label_or_ci95_high": zrow.iloc[0].get("or_ci95_high") if not zrow.empty else None, "label_regression_p": zrow.iloc[0].get("p_value") if not zrow.empty else None})
    assoc = pd.DataFrame(assoc_rows)
    for comp_type in ["predicted", "residual"]:
        mask = assoc["component_type"] == comp_type
        assoc.loc[mask, "q_value"] = _bh(assoc.loc[mask, "mannwhitney_p"].tolist())
    desc = pd.DataFrame(desc_rows)
    desc.to_csv(out_dir / "exif_component_label_descriptive.csv", index=False, encoding="utf-8-sig")
    assoc.to_csv(out_dir / "exif_component_label_association.csv", index=False, encoding="utf-8-sig")
    fdr = _fdr_summary(assoc, "component_type")
    (out_dir / "exif_component_label_fdr.json").write_text(json.dumps(fdr, ensure_ascii=False, indent=2), encoding="utf-8")
    return desc, assoc, fdr


def component_rgb_models(pred: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    logit_rows = []
    error_rows = []
    for comp_type in ["predicted", "residual"]:
        for target in TARGET_METRICS:
            col = f"{comp_type}_{target}"
            work = pred[["patient_group_id", "binary_label", "rgb_oof_logit", "rgb_oof_error", col]].copy()
            work["z_component"] = _z(work[col])
            lg = _fit_model(work, "rgb_oof_logit", ["binary_label", "z_component"], "ols")
            er = _fit_model(work, "rgb_oof_error", ["binary_label", "z_component"], "logit")
            lg["component_type"] = comp_type
            lg["target_metric"] = target
            er["component_type"] = comp_type
            er["target_metric"] = target
            logit_rows.append(lg)
            error_rows.append(er)
    logit = pd.concat(logit_rows, ignore_index=True)
    error = pd.concat(error_rows, ignore_index=True)
    for df in [logit, error]:
        df["q_value"] = np.nan
        for comp_type in ["predicted", "residual"]:
            mask = (df["component_type"] == comp_type) & (df["term"] == "z_component")
            df.loc[mask, "q_value"] = _bh(df.loc[mask, "p_value"].tolist())
    logit.to_csv(out_dir / "exif_component_rgb_logit_regression.csv", index=False, encoding="utf-8-sig")
    error.to_csv(out_dir / "exif_component_error_risk_regression.csv", index=False, encoding="utf-8-sig")
    logit_fdr = _model_fdr_summary(logit)
    error_fdr = _model_fdr_summary(error)
    (out_dir / "exif_component_rgb_logit_fdr.json").write_text(json.dumps(logit_fdr, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "exif_component_error_risk_fdr.json").write_text(json.dumps(error_fdr, ensure_ascii=False, indent=2), encoding="utf-8")
    return logit, logit_fdr, error, error_fdr


def component_error_magnitude(pred: pd.DataFrame, out_dir: Path, *, iterations: int, seed: int) -> pd.DataFrame:
    rows = []
    pred = pred.copy()
    pred["abs_rgb_oof_residual"] = pred["rgb_oof_residual"].abs()
    for comp_type in ["predicted", "residual"]:
        for target in TARGET_METRICS:
            comp = f"{comp_type}_{target}"
            for y_col in ["rgb_oof_brier_contribution", "abs_rgb_oof_residual"]:
                for subset_name, sub in [("all", pred), ("control", pred[pred["binary_label"] == 0]), ("patient", pred[pred["binary_label"] == 1])]:
                    rho = _spearman(sub[comp], sub[y_col])
                    ci_low, ci_high, valid = cluster_bootstrap_spearman(sub[["patient_group_id", comp, y_col]].dropna(), x_col=comp, y_col=y_col, iterations=iterations, seed=seed)
                    rows.append({"component_type": comp_type, "target_metric": target, "component": comp, "target": y_col, "subset": subset_name, "spearman_r": rho, "spearman_ci95_low": ci_low, "spearman_ci95_high": ci_high, "bootstrap_valid": valid})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "exif_component_error_magnitude.csv", index=False, encoding="utf-8-sig")
    return out


def attenuation_summary(
    pred: pd.DataFrame,
    full_assoc: pd.DataFrame,
    full_logit: pd.DataFrame,
    full_error: pd.DataFrame,
    logit: pd.DataFrame,
    error: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    for target in TARGET_METRICS:
        orig_smd = float(full_assoc.loc[full_assoc["metric"] == target, "smd_patient_minus_control"].iloc[0])
        orig_auc = float(roc_auc_score(pred["binary_label"].astype(int), pred[f"observed_{target}"]))
        res = pred[f"residual_{target}"]
        res_smd = smd(res[pred["binary_label"] == 0], res[pred["binary_label"] == 1])
        res_auc = float(roc_auc_score(pred["binary_label"].astype(int), res))
        pred_smd = smd(pred.loc[pred["binary_label"] == 0, f"predicted_{target}"], pred.loc[pred["binary_label"] == 1, f"predicted_{target}"])
        orig_logit = _single_term(full_logit, f"z_{target}")
        orig_error = _single_term(full_error, f"z_{target}")
        resid_logit = _component_term(logit, target, "residual")
        resid_error = _component_term(error, target, "residual")
        eps = 1e-8
        rows.append(
            {
                "target_metric": target,
                "original_smd": orig_smd,
                "predicted_smd": pred_smd,
                "residual_smd": res_smd,
                "smd_attenuation": 1 - abs(res_smd) / (abs(orig_smd) + eps),
                "original_auc": orig_auc,
                "residual_auc": res_auc,
                "auc_distance_attenuation": 1 - abs(res_auc - 0.5) / (abs(orig_auc - 0.5) + eps),
                "original_rgb_logit_coef": orig_logit.get("coefficient"),
                "original_rgb_logit_ci95_low": orig_logit.get("ci95_low"),
                "original_rgb_logit_ci95_high": orig_logit.get("ci95_high"),
                "residual_rgb_logit_coef": resid_logit.get("coefficient"),
                "residual_rgb_logit_ci95_low": resid_logit.get("ci95_low"),
                "residual_rgb_logit_ci95_high": resid_logit.get("ci95_high"),
                "rgb_logit_coef_attenuation": _attenuate(resid_logit.get("coefficient"), orig_logit.get("coefficient")),
                "original_error_risk_or": orig_error.get("or"),
                "original_error_risk_or_ci95_low": orig_error.get("or_ci95_low"),
                "original_error_risk_or_ci95_high": orig_error.get("or_ci95_high"),
                "residual_error_risk_or": resid_error.get("or"),
                "residual_error_risk_or_ci95_low": resid_error.get("or_ci95_low"),
                "residual_error_risk_or_ci95_high": resid_error.get("or_ci95_high"),
                "error_log_or_attenuation": _attenuate(resid_error.get("coefficient"), orig_error.get("coefficient")),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "exif_decomposition_attenuation_summary.csv", index=False, encoding="utf-8-sig")
    return out


def _single_term(frame: pd.DataFrame, term: str) -> dict[str, Any]:
    rows = frame[frame["term"].astype(str) == term]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _component_term(frame: pd.DataFrame, target: str, component_type: str) -> dict[str, Any]:
    rows = frame[(frame["target_metric"] == target) & (frame["component_type"] == component_type) & (frame["term"] == "z_component")]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _attenuate(residual_value: Any, original_value: Any) -> float | None:
    if residual_value is None or original_value is None:
        return None
    residual = float(residual_value)
    original = float(original_value)
    if not np.isfinite(residual) or not np.isfinite(original):
        return None
    return 1 - abs(residual) / (abs(original) + 1e-8)


def interpretation(perf: pd.DataFrame, label_assoc: pd.DataFrame, error: pd.DataFrame, attenuation: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    result = {}
    for target in TARGET_METRICS:
        r2 = float(perf.loc[perf["target_metric"] == target, "r2"].iloc[0])
        pred_row = label_assoc[(label_assoc["target_metric"] == target) & (label_assoc["component_type"] == "predicted")].iloc[0]
        resid_row = label_assoc[(label_assoc["target_metric"] == target) & (label_assoc["component_type"] == "residual")].iloc[0]
        err_pred = error[(error["target_metric"] == target) & (error["component_type"] == "predicted") & (error["term"] == "z_component")]
        err_res = error[(error["target_metric"] == target) & (error["component_type"] == "residual") & (error["term"] == "z_component")]
        pred_sig = bool(float(pred_row.get("q_value", 1)) < 0.05)
        res_sig = bool(float(resid_row.get("q_value", 1)) < 0.05)
        err_pred_sig = (not err_pred.empty) and float(err_pred.iloc[0].get("q_value", 1)) < 0.05
        err_res_sig = (not err_res.empty) and float(err_res.iloc[0].get("q_value", 1)) < 0.05
        attenuation_row = attenuation[attenuation["target_metric"] == target].iloc[0]
        if r2 >= 0.20 and abs(float(attenuation_row["smd_attenuation"])) > 0.30 and not res_sig:
            category = "possible_exposure_related_contribution"
        elif r2 < 0.10 and res_sig:
            category = "limited_exif_explanation"
        elif (pred_sig or err_pred_sig) and (res_sig or err_res_sig):
            category = "mixed_recorded_exposure_and_unexplained_component"
        else:
            category = "limited_independent_error_association"
        result[target] = {"interpretation": category, "oof_r2": r2, "predicted_label_q": pred_row.get("q_value"), "residual_label_q": resid_row.get("q_value"), "predicted_error_q": err_pred.iloc[0].get("q_value") if not err_pred.empty else None, "residual_error_q": err_res.iloc[0].get("q_value") if not err_res.empty else None, "warning": "EXIF-residual component denotes image-metric variation unexplained by the recorded EXIF exposure fields."}
    (out_dir / "exif_decomposition_interpretation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _bh(p_values: list[Any]) -> list[float | None]:
    vals = np.asarray([np.nan if p is None else float(p) for p in p_values], dtype=float)
    out = np.full(vals.shape, np.nan)
    ok = np.isfinite(vals)
    if not ok.any():
        return [None for _ in vals]
    p = vals[ok]
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    temp = np.empty_like(q)
    temp[order] = np.clip(q, 0, 1)
    out[ok] = temp
    return [None if not np.isfinite(x) else float(x) for x in out]


def _fdr_summary(df: pd.DataFrame, family_col: str) -> dict[str, Any]:
    out = {}
    for family, sub in df.groupby(family_col, sort=False):
        out[str(family)] = {"tested": int(len(sub)), "q_lt_0.05": int(pd.to_numeric(sub["q_value"], errors="coerce").lt(0.05).sum()), "significant": sub.loc[pd.to_numeric(sub["q_value"], errors="coerce").lt(0.05), "target_metric"].tolist()}
    return out


def _model_fdr_summary(df: pd.DataFrame) -> dict[str, Any]:
    sub = df[df["term"] == "z_component"]
    return _fdr_summary(sub, "component_type")


def run_exif_decomposition(features: pd.DataFrame, stage1: pd.DataFrame, out_dir: Path, *, iterations: int, seed: int) -> dict[str, Any]:
    exif_dir = out_dir / "exif"
    exif_dir.mkdir(parents=True, exist_ok=True)
    exif, mapping, coverage = build_exif_frame(stage1, features)
    (exif_dir / "exif_field_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    coverage.to_csv(out_dir / "preflight" / "r2_exif_coverage_audit.csv", index=False, encoding="utf-8-sig")
    pred, perf, fold_perf, coefs = crossfit_decomposition(exif, exif_dir, alpha=1.0)
    pred.to_csv(exif_dir / "exif_crossfit_predictions_500.csv", index=False, encoding="utf-8-sig")
    perf.to_csv(exif_dir / "exif_decomposition_model_performance.csv", index=False, encoding="utf-8-sig")
    fold_perf.to_csv(exif_dir / "exif_decomposition_fold_performance.csv", index=False, encoding="utf-8-sig")
    coefs.to_csv(exif_dir / "exif_model_coefficients_by_fold.csv", index=False, encoding="utf-8-sig")
    desc, label_assoc, label_fdr = component_label_analysis(pred, exif_dir, iterations=iterations, seed=seed)
    logit, logit_fdr, error, error_fdr = component_rgb_models(pred, exif_dir)
    magnitude = component_error_magnitude(pred, exif_dir, iterations=iterations, seed=seed)
    full500_root = out_dir.parents[0] / "Lighting_QC_Stage2_Full500_v1"
    full_assoc = pd.read_csv(full500_root / "association" / "lighting_label_effect_sizes.csv")
    full_logit = pd.read_csv(full500_root / "rgb_dependence" / "rgb_logit_adjusted_regression.csv")
    full_error = pd.read_csv(full500_root / "rgb_dependence" / "rgb_error_risk_regression.csv")
    atten = attenuation_summary(pred, full_assoc, full_logit, full_error, logit, error, exif_dir)
    interp = interpretation(perf, label_assoc, error, atten, exif_dir)
    return {
        "predictions": pred,
        "performance": perf,
        "label_assoc": label_assoc,
        "rgb_logit": logit,
        "error": error,
        "attenuation": atten,
        "interpretation": interp,
        "field_mapping": mapping,
        "coverage": coverage,
        "label_fdr": label_fdr,
        "logit_fdr": logit_fdr,
        "error_fdr": error_fdr,
    }
