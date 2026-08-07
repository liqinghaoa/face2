from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.contingency import association
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .metrics_utils import cluster_bootstrap_smd, smd


CONTINUOUS_VARS = [
    "log2_exposure_time",
    "log2_iso",
    "brightness_value_apex",
    "brightness_device_z",
    "log2_iso_device_z",
    "log2_exposure_device_z",
    "f_number",
    "exposure_bias_ev",
]

CATEGORICAL_VARS = ["camera_model", "flash", "metering_mode", "white_balance", "exposure_mode", "capture_year", "capture_month", "capture_year_month", "capture_hour_bin"]


def _iqr(series: pd.Series) -> tuple[float | None, float | None, float | None]:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return None, None, None
    return float(x.median()), float(x.quantile(0.25)), float(x.quantile(0.75))


def _categorical_test(table: pd.DataFrame) -> tuple[str, float | None, float | None, float | None]:
    if table.shape == (2, 2):
        try:
            oddsratio, p = stats.fisher_exact(table.to_numpy())
            return "fisher_exact", float(p), float(oddsratio), None
        except Exception:
            pass
    try:
        chi2, p, _, _ = stats.chi2_contingency(table.to_numpy())
        v = association(table.to_numpy(), method="cramer")
        return "chi_square", float(p), None, float(v)
    except Exception:
        return "not_available", None, None, None


def run_association_analysis(master: pd.DataFrame, out_dir: Path, *, bootstrap_repeats: int, seed: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cont_rows = []
    for var in CONTINUOUS_VARS:
        if var not in master.columns:
            continue
        c = master.loc[master["binary_label"] == 0, var]
        p = master.loc[master["binary_label"] == 1, var]
        cm, cq1, cq3 = _iqr(c)
        pm, pq1, pq3 = _iqr(p)
        try:
            mw_p = float(stats.mannwhitneyu(pd.to_numeric(c, errors="coerce").dropna(), pd.to_numeric(p, errors="coerce").dropna(), alternative="two-sided").pvalue)
        except Exception:
            mw_p = None
        point_smd = smd(c, p)
        ci_low, ci_high, valid = cluster_bootstrap_smd(master, value_col=var, repeats=bootstrap_repeats, seed=seed)
        cont_rows.append(
            {
                "variable": var,
                "control_n": int(pd.to_numeric(c, errors="coerce").notna().sum()),
                "patient_n": int(pd.to_numeric(p, errors="coerce").notna().sum()),
                "control_median": cm,
                "control_iqr_q1": cq1,
                "control_iqr_q3": cq3,
                "patient_median": pm,
                "patient_iqr_q1": pq1,
                "patient_iqr_q3": pq3,
                "control_mean": float(pd.to_numeric(c, errors="coerce").mean()),
                "control_sd": float(pd.to_numeric(c, errors="coerce").std()),
                "patient_mean": float(pd.to_numeric(p, errors="coerce").mean()),
                "patient_sd": float(pd.to_numeric(p, errors="coerce").std()),
                "mannwhitney_p": mw_p,
                "smd_patient_minus_control": point_smd,
                "smd_ci95_low": ci_low,
                "smd_ci95_high": ci_high,
                "smd_bootstrap_valid": valid,
            }
        )
    cont = pd.DataFrame(cont_rows)
    cont.to_csv(out_dir / "metadata_label_association_continuous.csv", index=False, encoding="utf-8-sig")

    cat_rows = []
    by_device_rows = []
    for var in CATEGORICAL_VARS:
        if var not in master.columns:
            continue
        table = pd.crosstab(master[var].fillna("missing").astype(str), master["binary_label"].astype(int))
        test, p_value, oddsratio, cramer_v = _categorical_test(table)
        for level, row in table.iterrows():
            control = int(row.get(0, 0))
            patient = int(row.get(1, 0))
            total = control + patient
            payload = {
                "variable": var,
                "level": level,
                "n": total,
                "control_n": control,
                "patient_n": patient,
                "patient_proportion": patient / total if total else np.nan,
                "test": test,
                "p_value": p_value,
                "oddsratio_if_2x2": oddsratio,
                "cramers_v": cramer_v,
            }
            cat_rows.append(payload)
            if var == "camera_model":
                by_device_rows.append(payload.copy())
    cat = pd.DataFrame(cat_rows)
    cat.to_csv(out_dir / "metadata_label_association_categorical.csv", index=False, encoding="utf-8-sig")
    by_device = pd.DataFrame(by_device_rows)
    by_device.to_csv(out_dir / "metadata_label_association_by_device.csv", index=False, encoding="utf-8-sig")

    multivariable = _multivariable_association(master)
    multivariable.to_csv(out_dir / "metadata_multivariable_association.csv", index=False, encoding="utf-8-sig")
    return {"continuous": cont, "categorical": cat, "by_device": by_device, "multivariable": multivariable}


def _multivariable_association(master: pd.DataFrame) -> pd.DataFrame:
    try:
        import statsmodels.api as sm

        cols_num = [c for c in ["brightness_device_z", "log2_iso_device_z", "log2_exposure_device_z", "exposure_bias_ev"] if c in master.columns and master[c].notna().sum() > 10]
        cats = ["camera_model"] + (["flash"] if "flash" in master.columns and master["flash"].nunique(dropna=True) > 1 else [])
        x_parts = []
        names = []
        if cols_num:
            num = master[cols_num].apply(pd.to_numeric, errors="coerce")
            num = pd.DataFrame(StandardScaler().fit_transform(num.fillna(num.median())), columns=cols_num, index=master.index)
            x_parts.append(num)
            names.extend(cols_num)
        for cat in cats:
            enc = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
            arr = enc.fit_transform(master[[cat]].fillna("missing").astype(str))
            enc_names = [f"{cat}={name.split('_', 1)[-1]}" for name in enc.get_feature_names_out([cat])]
            x_parts.append(pd.DataFrame(arr, columns=enc_names, index=master.index))
            names.extend(enc_names)
        x = pd.concat(x_parts, axis=1)
        x = sm.add_constant(x, has_constant="add")
        y = master["binary_label"].astype(int)
        model = sm.Logit(y, x).fit(disp=False, maxiter=200)
        ci = model.conf_int()
        rows = []
        for term in model.params.index:
            rows.append(
                {
                    "term": term,
                    "coefficient": float(model.params[term]),
                    "or": float(np.exp(model.params[term])),
                    "ci95_low": float(np.exp(ci.loc[term, 0])),
                    "ci95_high": float(np.exp(ci.loc[term, 1])),
                    "p_value": float(model.pvalues[term]),
                    "direction": "positive" if model.params[term] > 0 else "negative",
                }
            )
        return pd.DataFrame(rows)
    except Exception as exc:
        return pd.DataFrame([{"term": "model_failed", "error": f"{type(exc).__name__}: {exc}"}])
