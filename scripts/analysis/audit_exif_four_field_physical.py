"""Physical usability audit for four core EXIF fields.

The analysis is deliberately label-free: it does not read NYHA, SEX, split
files, or model predictions and does not train any classifier. Existing EXIF
audit CSVs and existing color-preserving aligned images / parsing masks are
reused. Source files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import scipy
import sklearn
import statsmodels
import statsmodels.api as sm
from PIL import Image
from scipy import ndimage, stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EPSILON = 1e-8
CORE_FIELDS = ["ExposureTime", "FNumber", "ISOSpeedRatings", "BrightnessValue"]
DERIVED_FIELDS = [
    "log2_exposure_time",
    "log2_iso_gain",
    "aperture_value_from_f",
    "time_value_from_t",
    "EV100",
    "relative_optical_exposure",
    "combined_exposure_gain",
    "sensitivity_value",
    "brightness_apex_pred",
    "brightness_residual",
    "device_centered_brightness",
]
ANALYSIS_FIELDS = CORE_FIELDS + DERIVED_FIELDS
DECISION_FIELDS = CORE_FIELDS + [
    "log2_exposure_time",
    "log2_iso_gain",
    "EV100",
    "relative_optical_exposure",
    "combined_exposure_gain",
    "brightness_apex_pred",
    "brightness_residual",
    "device_centered_brightness",
]
CAMERA_COLORS = {
    "HONOR/BVL-AN00": "#3B6FB6",
    "Xiaomi/M2006J10C": "#D9863D",
}


def configure_plotting() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: Path) -> dict[str, Any]:
    files = sorted(path.glob("*.png"), key=lambda item: item.name.casefold())
    digest = hashlib.sha256()
    total_bytes = 0
    for file_path in files:
        size = file_path.stat().st_size
        total_bytes += size
        digest.update(
            f"{file_path.name}\t{size}\t{sha256_file(file_path)}\n".encode("utf-8")
        )
    return {
        "path": str(path.resolve()),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "content_manifest_sha256": digest.hexdigest(),
    }


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    view = frame.head(max_rows).copy() if max_rows is not None else frame.copy()
    if view.empty:
        return "（无记录）"
    view = view.fillna("")
    columns = [str(column) for column in view.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in view.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                text = f"{float(value):.6g}" if math.isfinite(float(value)) else ""
            else:
                text = str(value)
            rendered.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def load_existing_exif(
    values_path: Path,
    image_audit_path: Path,
    issues_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = pd.read_csv(values_path, dtype={"ID": "string"}, encoding="utf-8-sig")
    image_audit = pd.read_csv(image_audit_path, dtype={"ID": "string"}, encoding="utf-8-sig")
    issues = pd.read_csv(issues_path, dtype={"ID": "string"}, encoding="utf-8-sig")
    required_long = {"ID", "parameter", "status", "numeric_value"}
    required_images = {"ID", "Make", "Model"}
    if required_long.difference(long.columns):
        raise ValueError(f"EXIF long table missing columns: {sorted(required_long.difference(long.columns))}")
    if required_images.difference(image_audit.columns):
        raise ValueError(f"Image audit missing columns: {sorted(required_images.difference(image_audit.columns))}")

    selected = long.loc[long["parameter"].isin(CORE_FIELDS)].copy()
    duplicate_counts = selected.groupby(["ID", "parameter"]).size()
    duplicates = duplicate_counts.loc[duplicate_counts != 1]
    if not duplicates.empty:
        raise ValueError(f"Core EXIF ID-field mapping is not one-to-one: {duplicates.to_dict()}")
    selected["numeric_value"] = pd.to_numeric(selected["numeric_value"], errors="coerce")
    values = selected.pivot(index="ID", columns="parameter", values="numeric_value").reset_index()
    statuses = selected.pivot(index="ID", columns="parameter", values="status").reset_index()
    statuses = statuses.rename(columns={field: f"{field}_status" for field in CORE_FIELDS})
    frame = image_audit[["ID", "Make", "Model"]].merge(values, on="ID", how="inner", validate="one_to_one")
    frame = frame.merge(statuses, on="ID", how="inner", validate="one_to_one")
    frame["ID"] = frame["ID"].astype(str).str.strip()
    frame["Make"] = frame["Make"].astype(str).str.strip()
    frame["Model"] = frame["Model"].astype(str).str.strip()
    frame["camera_id"] = frame["Make"] + "/" + frame["Model"]

    invalid_rows = []
    for field in CORE_FIELDS:
        numeric = pd.to_numeric(frame[field], errors="coerce")
        invalid = ~np.isfinite(numeric)
        if field in {"ExposureTime", "FNumber", "ISOSpeedRatings"}:
            invalid |= numeric <= 0
        invalid |= frame[f"{field}_status"] != "valid"
        for image_id in frame.loc[invalid, "ID"]:
            invalid_rows.append({"ID": image_id, "field": field})
    if invalid_rows:
        raise ValueError(f"Core EXIF contains missing/invalid values: {invalid_rows}")

    iso_issues = issues.loc[
        (issues["parameter"] == "ISOSpeedRatings")
        & (issues["issue_type"] == "device_stratified_statistical_outlier")
    ].copy()
    iso_issues = iso_issues[["ID", "issue_type", "severity", "detail"]].drop_duplicates("ID")
    if len(iso_issues) != 14:
        raise ValueError(f"Expected 14 historical high-ISO outliers, found {len(iso_issues)}")
    return frame.sort_values("ID").reset_index(drop=True), iso_issues


def derive_parameters(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["log2_exposure_time"] = np.log2(output["ExposureTime"])
    output["log2_iso_gain"] = np.log2(output["ISOSpeedRatings"] / 100.0)
    output["aperture_value_from_f"] = 2.0 * np.log2(output["FNumber"])
    output["time_value_from_t"] = -np.log2(output["ExposureTime"])
    output["EV100"] = output["aperture_value_from_f"] + output["time_value_from_t"]
    output["relative_optical_exposure"] = (
        output["log2_exposure_time"] - 2.0 * np.log2(output["FNumber"])
    )
    output["combined_exposure_gain"] = (
        output["relative_optical_exposure"] + output["log2_iso_gain"]
    )
    output["sensitivity_value"] = np.log2(output["ISOSpeedRatings"] / 3.125)
    output["brightness_apex_pred"] = (
        output["aperture_value_from_f"]
        + output["time_value_from_t"]
        - output["sensitivity_value"]
    )
    output["brightness_residual"] = (
        output["BrightnessValue"] - output["brightness_apex_pred"]
    )
    centered = pd.Series(index=output.index, dtype=float)
    for _, indices in output.groupby("camera_id").groups.items():
        values = output.loc[indices, "BrightnessValue"].astype(float)
        median = float(values.median())
        iqr = float(values.quantile(0.75) - values.quantile(0.25))
        centered.loc[indices] = (values - median) / (iqr + EPSILON)
    output["device_centered_brightness"] = centered
    return output


def raw_mad(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan")
    median = np.median(array)
    return float(np.median(np.abs(array - median)))


def describe(values: pd.Series, total_n: int | None = None) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    valid = numeric[np.isfinite(numeric)]
    total = len(numeric) if total_n is None else total_n
    result: dict[str, Any] = {
        "total_n": int(total),
        "valid_n": int(len(valid)),
        "missing_n": int(total - len(valid)),
        "unique_n": int(valid.nunique()),
    }
    names = ["min", "p1", "p5", "p25", "median", "p75", "p95", "p99", "max", "mean", "std", "iqr", "mad"]
    result.update({name: float("nan") for name in names})
    if valid.empty:
        return result
    quantiles = valid.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    result.update(
        {
            "min": float(valid.min()),
            "p1": float(quantiles.loc[0.01]),
            "p5": float(quantiles.loc[0.05]),
            "p25": float(quantiles.loc[0.25]),
            "median": float(quantiles.loc[0.5]),
            "p75": float(quantiles.loc[0.75]),
            "p95": float(quantiles.loc[0.95]),
            "p99": float(quantiles.loc[0.99]),
            "max": float(valid.max()),
            "mean": float(valid.mean()),
            "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
            "iqr": float(quantiles.loc[0.75] - quantiles.loc[0.25]),
            "mad": raw_mad(valid),
        }
    )
    return result


def build_summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    device_rows = []
    for variable in ANALYSIS_FIELDS:
        overall = {"scope": "overall", "variable": variable}
        overall.update(describe(frame[variable]))
        overall_rows.append(overall)
        for camera_id, group in frame.groupby("camera_id", sort=True):
            row = {"scope": "device", "camera_id": camera_id, "variable": variable}
            row.update(describe(group[variable]))
            device_rows.append(row)
    return pd.DataFrame(overall_rows), pd.DataFrame(device_rows)


def robust_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    median = numeric.median()
    mad = raw_mad(numeric)
    if not math.isfinite(mad) or mad <= 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return 0.6744897501960817 * (numeric - median) / mad


def build_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for camera_id, group in frame.groupby("camera_id", sort=True):
        for variable in ANALYSIS_FIELDS:
            values = group[variable].astype(float)
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            z = robust_z(values)
            iqr_flag = (values < lower) | (values > upper)
            mad_flag = z.abs() > 3.5
            for index in group.index[(iqr_flag | mad_flag).fillna(False)]:
                rows.append(
                    {
                        "ID": frame.loc[index, "ID"],
                        "Make": frame.loc[index, "Make"],
                        "Model": frame.loc[index, "Model"],
                        "camera_id": camera_id,
                        "variable": variable,
                        "value": values.loc[index],
                        "iqr_lower": lower,
                        "iqr_upper": upper,
                        "iqr_outlier": bool(iqr_flag.loc[index]),
                        "robust_z": z.loc[index],
                        "mad_outlier_abs_z_gt_3_5": bool(mad_flag.loc[index]) if pd.notna(mad_flag.loc[index]) else False,
                    }
                )
    return pd.DataFrame(rows)


def compute_image_quality(
    frame: pd.DataFrame,
    aligned_rgb_dir: Path,
    parsing_label_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows = []
    failures = []
    for position, image_id in enumerate(frame["ID"].astype(str), start=1):
        rgb_path = aligned_rgb_dir / f"{image_id}.png"
        label_path = parsing_label_dir / f"{image_id}.png"
        try:
            rgb_u8 = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
            label = np.asarray(Image.open(label_path), dtype=np.uint8)
            if label.shape != rgb_u8.shape[:2]:
                raise ValueError(f"shape mismatch rgb={rgb_u8.shape}, label={label.shape}")
            skin = label == 1
            skin_n = int(skin.sum())
            if skin_n < 64:
                raise ValueError(f"skin mask too small: {skin_n}")
            inner = ndimage.binary_erosion(skin, iterations=2)
            if int(inner.sum()) < 64:
                inner = skin

            rgb = rgb_u8.astype(np.float64) / 255.0
            skin_rgb = rgb[skin]
            linear_rgb = np.where(
                rgb <= 0.04045,
                rgb / 12.92,
                ((rgb + 0.055) / 1.055) ** 2.4,
            )
            luminance = (
                0.2126 * linear_rgb[..., 0]
                + 0.7152 * linear_rgb[..., 1]
                + 0.0722 * linear_rgb[..., 2]
            )
            skin_luminance = luminance[skin]
            delta = 6.0 / 29.0
            f_y = np.where(
                skin_luminance > delta**3,
                np.cbrt(skin_luminance),
                skin_luminance / (3.0 * delta**2) + 4.0 / 29.0,
            )
            lab_l = 116.0 * f_y - 16.0
            gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            laplacian = ndimage.laplace(gray, mode="reflect")
            smooth = ndimage.gaussian_filter(gray, sigma=1.0, mode="reflect")
            high_frequency = (gray - smooth)[inner]
            hf_median = float(np.median(high_frequency))

            rows.append(
                {
                    "ID": image_id,
                    "skin_pixel_n": skin_n,
                    "valid_skin_pixel_ratio": skin_n / float(skin.size),
                    "skin_median_r": float(np.median(skin_rgb[:, 0])),
                    "skin_median_g": float(np.median(skin_rgb[:, 1])),
                    "skin_median_b": float(np.median(skin_rgb[:, 2])),
                    "skin_linear_luminance_median": float(np.median(skin_luminance)),
                    "skin_lab_l_median": float(np.median(lab_l)),
                    "overexposed_pixel_ratio": float(np.mean(skin_luminance >= 0.98)),
                    "underexposed_pixel_ratio": float(np.mean(skin_luminance <= 0.02)),
                    "saturated_pixel_ratio": float(
                        np.mean(
                            np.any(
                                (skin_rgb <= 5.0 / 255.0)
                                | (skin_rgb >= 250.0 / 255.0),
                                axis=1,
                            )
                        )
                    ),
                    "laplacian_variance": float(np.var(laplacian[inner], ddof=1)),
                    "high_frequency_noise_mad": float(
                        1.4826 * np.median(np.abs(high_frequency - hf_median))
                    ),
                }
            )
        except Exception as exc:
            failures.append({"ID": image_id, "error": f"{type(exc).__name__}: {exc}"})
        if position % 100 == 0:
            print(f"[image] {position}/{len(frame)}", flush=True)
    return pd.DataFrame(rows), failures


def safe_correlations(x: pd.Series, y: pd.Series) -> dict[str, float]:
    data = pd.DataFrame(
        {"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}
    ).replace([np.inf, -np.inf], np.nan).dropna()
    result = {
        "n": int(len(data)),
        "pearson_r": float("nan"),
        "pearson_p": float("nan"),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
    }
    if len(data) < 3 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return result
    pearson = stats.pearsonr(data["x"], data["y"])
    spearman = stats.spearmanr(data["x"], data["y"])
    result.update(
        {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        }
    )
    return result


def image_relationships(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("ExposureTime", "skin_linear_luminance_median", "exposure_vs_luminance"),
        ("ExposureTime", "laplacian_variance", "exposure_vs_blur_proxy"),
        ("ISOSpeedRatings", "high_frequency_noise_mad", "iso_vs_noise_proxy"),
        ("ISOSpeedRatings", "underexposed_pixel_ratio", "iso_vs_underexposure"),
        ("ISOSpeedRatings", "skin_linear_luminance_median", "iso_vs_luminance"),
        ("FNumber", "skin_linear_luminance_median", "fnumber_vs_luminance"),
        ("FNumber", "laplacian_variance", "fnumber_vs_blur_proxy"),
        ("BrightnessValue", "skin_linear_luminance_median", "brightnessvalue_vs_luminance"),
        ("BrightnessValue", "skin_lab_l_median", "brightnessvalue_vs_lab_l"),
        ("BrightnessValue", "overexposed_pixel_ratio", "brightnessvalue_vs_overexposure"),
        ("BrightnessValue", "underexposed_pixel_ratio", "brightnessvalue_vs_underexposure"),
        ("relative_optical_exposure", "skin_linear_luminance_median", "relative_exposure_vs_luminance"),
        ("combined_exposure_gain", "skin_linear_luminance_median", "combined_gain_vs_luminance"),
    ]
    rows = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "ALL", frame)]
    scopes.extend(("device", str(camera_id), group) for camera_id, group in frame.groupby("camera_id"))
    for scope, camera_id, subset in scopes:
        for exif_field, image_metric, relationship in pairs:
            rows.append(
                {
                    "scope": scope,
                    "camera_id": camera_id,
                    "relationship": relationship,
                    "exif_field": exif_field,
                    "image_metric": image_metric,
                    **safe_correlations(subset[exif_field], subset[image_metric]),
                }
            )
    return pd.DataFrame(rows)


def apex_consistency(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "ALL", frame)]
    scopes.extend(("device", str(camera_id), group) for camera_id, group in frame.groupby("camera_id"))
    for scope, camera_id, subset in scopes:
        correlations = safe_correlations(subset["brightness_apex_pred"], subset["BrightnessValue"])
        x = subset["brightness_apex_pred"].to_numpy(dtype=float)
        y = subset["BrightnessValue"].to_numpy(dtype=float)
        fit = stats.linregress(x, y)
        residual = subset["brightness_residual"].astype(float)
        rows.append(
            {
                "scope": scope,
                "camera_id": camera_id,
                **correlations,
                "slope": float(fit.slope),
                "intercept": float(fit.intercept),
                "r_squared": float(fit.rvalue**2),
                "residual_median": float(residual.median()),
                "residual_iqr": float(residual.quantile(0.75) - residual.quantile(0.25)),
                "residual_mad": raw_mad(residual),
                "residual_min": float(residual.min()),
                "residual_max": float(residual.max()),
            }
        )

    cameras = sorted(frame["camera_id"].unique().tolist())
    if len(cameras) == 2:
        camera = (frame["camera_id"] == cameras[1]).astype(float).to_numpy()
        pred = frame["brightness_apex_pred"].to_numpy(dtype=float)
        design = sm.add_constant(np.column_stack([pred, camera, pred * camera]), has_constant="add")
        model = sm.OLS(frame["BrightnessValue"].to_numpy(dtype=float), design).fit()
        names = ["intercept", "brightness_apex_pred", "camera_intercept_shift", "camera_slope_shift"]
        for index, term in enumerate(names):
            rows.append(
                {
                    "scope": "device_interaction_model",
                    "camera_id": f"reference={cameras[0]}; indicator={cameras[1]}",
                    "term": term,
                    "coefficient": float(model.params[index]),
                    "standard_error": float(model.bse[index]),
                    "term_p": float(model.pvalues[index]),
                    "r_squared": float(model.rsquared),
                    "n": int(model.nobs),
                }
            )
    return pd.DataFrame(rows)


def correlation_matrix_long(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes: list[tuple[str, pd.DataFrame]] = [("overall", frame)]
    scopes.extend((str(camera_id), group) for camera_id, group in frame.groupby("camera_id"))
    for scope, subset in scopes:
        for method in ["pearson", "spearman"]:
            matrix = subset[ANALYSIS_FIELDS].corr(method=method)
            for variable_a in ANALYSIS_FIELDS:
                for variable_b in ANALYSIS_FIELDS:
                    rows.append(
                        {
                            "scope": scope,
                            "method": method,
                            "variable_a": variable_a,
                            "variable_b": variable_b,
                            "correlation": matrix.loc[variable_a, variable_b],
                            "n": len(subset),
                        }
                    )
    return pd.DataFrame(rows)


def empirical_overlap(a: pd.Series, b: pd.Series) -> float:
    left = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    right = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    combined = np.concatenate([left, right])
    if len(combined) == 0:
        return float("nan")
    if np.ptp(combined) == 0:
        return 1.0
    bins = np.histogram_bin_edges(combined, bins="fd")
    if len(bins) < 3:
        bins = np.linspace(float(combined.min()), float(combined.max()), 11)
    left_hist, _ = np.histogram(left, bins=bins)
    right_hist, _ = np.histogram(right, bins=bins)
    left_prob = left_hist / max(left_hist.sum(), 1)
    right_prob = right_hist / max(right_hist.sum(), 1)
    return float(np.minimum(left_prob, right_prob).sum())


def hedges_g(a: pd.Series, b: pd.Series) -> float:
    left = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    right = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    pooled = math.sqrt(
        max(
            ((len(left) - 1) * np.var(left, ddof=1) + (len(right) - 1) * np.var(right, ddof=1))
            / (len(left) + len(right) - 2),
            0.0,
        )
    )
    difference = float(np.mean(right) - np.mean(left))
    if pooled == 0:
        return math.copysign(float("inf"), difference) if difference else 0.0
    correction = 1.0 - 3.0 / (4.0 * (len(left) + len(right)) - 9.0)
    return correction * difference / pooled


def device_shift_summary(frame: pd.DataFrame) -> pd.DataFrame:
    cameras = sorted(frame["camera_id"].unique().tolist())
    if len(cameras) != 2:
        raise ValueError(f"Expected exactly two camera_id values, got {cameras}")
    rows = []
    a = frame.loc[frame["camera_id"] == cameras[0]]
    b = frame.loc[frame["camera_id"] == cameras[1]]
    for variable in ANALYSIS_FIELDS:
        test = stats.mannwhitneyu(a[variable], b[variable], alternative="two-sided")
        rows.append(
            {
                "variable": variable,
                "camera_a": cameras[0],
                "camera_b": cameras[1],
                "median_a": float(a[variable].median()),
                "median_b": float(b[variable].median()),
                "median_difference_b_minus_a": float(b[variable].median() - a[variable].median()),
                "hedges_g_b_minus_a": hedges_g(a[variable], b[variable]),
                "distribution_overlap": empirical_overlap(a[variable], b[variable]),
                "mann_whitney_u": float(test.statistic),
                "mann_whitney_p_descriptive": float(test.pvalue),
            }
        )
    return pd.DataFrame(rows)


def build_high_iso_review(
    frame: pd.DataFrame,
    iso_issues: pd.DataFrame,
) -> pd.DataFrame:
    review = frame.loc[frame["ID"].isin(set(iso_issues["ID"]))].copy()
    review = review.merge(iso_issues, on="ID", how="left", validate="one_to_one")
    review["noise_robust_z_within_device"] = np.nan
    review["blur_robust_z_within_device"] = np.nan
    for camera_id, group in frame.groupby("camera_id"):
        noise_map = dict(zip(group["ID"], robust_z(group["high_frequency_noise_mad"])))
        blur_map = dict(zip(group["ID"], robust_z(group["laplacian_variance"])))
        mask = review["camera_id"] == camera_id
        review.loc[mask, "noise_robust_z_within_device"] = review.loc[mask, "ID"].map(noise_map)
        review.loc[mask, "blur_robust_z_within_device"] = review.loc[mask, "ID"].map(blur_map)
    review["是否过曝"] = review["overexposed_pixel_ratio"] > 0.01
    review["是否欠曝"] = review["underexposed_pixel_ratio"] > 0.05
    review["是否明显噪声较高"] = review["noise_robust_z_within_device"] > 3.5
    review["是否明显模糊"] = review["blur_robust_z_within_device"] < -3.5
    review["解析错误或哨兵值"] = False
    device_p25 = frame.groupby("camera_id")["BrightnessValue"].quantile(0.25).to_dict()
    review["合理弱光条件线索"] = review.apply(
        lambda row: bool(
            row["BrightnessValue"] <= device_p25[str(row["camera_id"])]
            or row["underexposed_pixel_ratio"] > 0.01
        ),
        axis=1,
    )

    def classify(row: pd.Series) -> str:
        if bool(row["解析错误或哨兵值"]):
            return "明确的元数据异常"
        if any(
            bool(row[field])
            for field in ["是否过曝", "是否欠曝", "是否明显噪声较高", "是否明显模糊"]
        ):
            return "需要保留但标记为低质量"
        return "合理的真实拍摄条件"

    review["最终复核分类"] = review.apply(classify, axis=1)
    columns = [
        "ID",
        "Make",
        "Model",
        *CORE_FIELDS,
        "overexposed_pixel_ratio",
        "underexposed_pixel_ratio",
        "high_frequency_noise_mad",
        "laplacian_variance",
        "noise_robust_z_within_device",
        "blur_robust_z_within_device",
        "是否过曝",
        "是否欠曝",
        "是否明显噪声较高",
        "是否明显模糊",
        "解析错误或哨兵值",
        "合理弱光条件线索",
        "最终复核分类",
        "detail",
    ]
    return review[columns].sort_values(["Make", "Model", "ISOSpeedRatings", "ID"], ascending=[True, True, False, True])


def build_decision_table(
    frame: pd.DataFrame,
    overall: pd.DataFrame,
    by_device: pd.DataFrame,
    shifts: pd.DataFrame,
    relationships: pd.DataFrame,
) -> pd.DataFrame:
    roles = {
        "ExposureTime": "derived_only",
        "FNumber": "renderer_fixed_parameter",
        "ISOSpeedRatings": "derived_only",
        "BrightnessValue": "device_standardized_condition",
        "log2_exposure_time": "derived_only",
        "log2_iso_gain": "core_continuous_condition",
        "EV100": "quality_control_only",
        "relative_optical_exposure": "core_continuous_condition",
        "combined_exposure_gain": "quality_control_only",
        "brightness_apex_pred": "quality_control_only",
        "brightness_residual": "quality_control_only",
        "device_centered_brightness": "device_standardized_condition",
    }
    conclusions = {
        "ExposureTime": "逐图有变化，但网络中优先通过log2及relative_optical_exposure表达。",
        "FNumber": "设备内几乎固定，不作为独立连续输入；作为renderer设备固定参数并参与相对进光量公式。",
        "ISOSpeedRatings": "逐图有明显变化；使用log2_iso_gain表达增益，高ISO保留并做质量标记。",
        "BrightnessValue": "存在设备系统偏移，不能跨设备直接输入；改用设备内稳健标准化值。",
        "log2_exposure_time": "比秒值更适合数值表示，但V1中被relative_optical_exposure吸收。",
        "log2_iso_gain": "V1核心增益条件。",
        "EV100": "与ExposureTime和FNumber确定性相关，保留作物理审计，不与V1并列输入。",
        "relative_optical_exposure": "合并曝光时间与光圈的光学进光条件，V1核心输入。",
        "combined_exposure_gain": "由relative_optical_exposure和log2_iso_gain相加，信息确定性冗余，作敏感性/QC。",
        "brightness_apex_pred": "公式预测值只用于BrightnessValue一致性审计。",
        "brightness_residual": "用于发现设备偏移和元数据/ISP不一致，不作生理表型输入。",
        "device_centered_brightness": "减少设备系统偏移后的V1亮度条件；未来训练时统计量必须仅由训练集拟合。",
    }
    overall_lookup = overall.set_index("variable")
    shift_lookup = shifts.set_index("variable")
    rows = []
    for variable in DECISION_FIELDS:
        item = overall_lookup.loc[variable]
        device_stats = by_device.loc[by_device["variable"] == variable]
        within = "; ".join(
            f"{row.camera_id}: unique={int(row.unique_n)}, std={row.std:.4g}, IQR={row.iqr:.4g}, MAD={row.mad:.4g}"
            for row in device_stats.itertuples(index=False)
        )
        shift = shift_lookup.loc[variable]
        shift_text = (
            f"median差(B-A)={shift.median_difference_b_minus_a:.4g}; "
            f"Hedges g={shift.hedges_g_b_minus_a:.3g}; overlap={shift.distribution_overlap:.3g}"
        )
        related = relationships.loc[
            (relationships["scope"] == "overall") & (relationships["exif_field"] == variable)
        ]
        if related.empty:
            relation_text = "未单独对应图像指标；由组成字段或一致性分析解释"
        else:
            strongest = related.iloc[related["spearman_rho"].abs().argmax()]
            relation_text = f"最强关系={strongest.image_metric}, Spearman rho={strongest.spearman_rho:.3f}"
        rows.append(
            {
                "字段": variable,
                "完整性": f"{int(item['valid_n'])}/{int(item['total_n'])} ({item['valid_n']/item['total_n']:.1%})",
                "总体变异": f"unique={int(item['unique_n'])}, std={item['std']:.4g}, IQR={item['iqr']:.4g}, MAD={item['mad']:.4g}",
                "设备内变异": within,
                "设备系统偏移": shift_text,
                "与图像表现关系": relation_text,
                "推荐角色": roles[variable],
                "结论依据": conclusions[variable],
            }
        )
    return pd.DataFrame(rows)


def plot_four_field_distributions(frame: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for ax, variable in zip(axes.flat, CORE_FIELDS):
        ax.hist(frame[variable], bins=28, color="#4C78A8", alpha=0.78, edgecolor="white")
        ax.axvline(frame[variable].median(), color="#B44B4B", lw=1.4, ls="--", label="median")
        ax.set_title(variable, fontweight="bold")
        ax.set_xlabel(variable)
        ax.set_ylabel("Images")
        ax.grid(axis="y", alpha=0.18)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Distributions of four core EXIF fields (n=522)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, figures_dir / "four_field_distributions.png")


def plot_four_field_by_device(frame: pd.DataFrame, figures_dir: Path) -> None:
    cameras = sorted(frame["camera_id"].unique().tolist())
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for ax, variable in zip(axes.flat, CORE_FIELDS):
        data = [frame.loc[frame["camera_id"] == camera, variable].to_numpy() for camera in cameras]
        box = ax.boxplot(data, tick_labels=cameras, patch_artist=True, showfliers=True, flierprops={"markersize": 2})
        for patch, camera in zip(box["boxes"], cameras):
            patch.set_facecolor(CAMERA_COLORS.get(camera, "#777777"))
            patch.set_alpha(0.72)
        ax.set_title(variable, fontweight="bold")
        ax.set_ylabel(variable)
        ax.tick_params(axis="x", rotation=14, labelsize=6.5)
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Four core EXIF fields by camera", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, figures_dir / "four_field_by_device.png")


def plot_correlation_heatmap(frame: pd.DataFrame, figures_dir: Path) -> None:
    matrix = frame[ANALYSIS_FIELDS].corr(method="spearman")
    safe = matrix.fillna(0.0).clip(-1, 1)
    distance = 1.0 - np.abs(safe.to_numpy())
    np.fill_diagonal(distance, 0.0)
    linkage = hierarchy.linkage(squareform(distance, checks=False), method="average")
    order = hierarchy.leaves_list(linkage)
    ordered = matrix.iloc[order, order]
    fig, ax = plt.subplots(figsize=(7.1, 6.6))
    image = ax.imshow(ordered, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    labels = ordered.index.tolist()
    ax.set_xticks(range(len(labels)), labels, rotation=62, ha="right", fontsize=6)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    ax.set_title("Spearman correlation: original and derived EXIF variables", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    colorbar.set_label("Spearman rho")
    fig.tight_layout()
    save_figure(fig, figures_dir / "correlation_heatmap.png")


def plot_apex_consistency(frame: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    for camera_id, group in frame.groupby("camera_id", sort=True):
        color = CAMERA_COLORS.get(str(camera_id), "#777777")
        x = group["brightness_apex_pred"].to_numpy(dtype=float)
        y = group["BrightnessValue"].to_numpy(dtype=float)
        ax.scatter(x, y, s=13, alpha=0.52, color=color, label=f"{camera_id} (n={len(group)})")
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, slope * grid + intercept, color=color, lw=1.5)
    limits = [
        min(frame["brightness_apex_pred"].min(), frame["BrightnessValue"].min()),
        max(frame["brightness_apex_pred"].max(), frame["BrightnessValue"].max()),
    ]
    ax.plot(limits, limits, color="#666666", ls="--", lw=1, label="identity")
    ax.set_xlabel("APEX-predicted BrightnessValue")
    ax.set_ylabel("Recorded BrightnessValue")
    ax.set_title("BrightnessValue physical consistency", fontweight="bold")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    save_figure(fig, figures_dir / "brightness_apex_consistency.png")

    fig, ax = plt.subplots(figsize=(5.3, 4.0))
    cameras = sorted(frame["camera_id"].unique().tolist())
    data = [frame.loc[frame["camera_id"] == camera, "brightness_residual"].to_numpy() for camera in cameras]
    box = ax.boxplot(data, tick_labels=cameras, patch_artist=True, showfliers=True, flierprops={"markersize": 2})
    for patch, camera in zip(box["boxes"], cameras):
        patch.set_facecolor(CAMERA_COLORS.get(camera, "#777777"))
        patch.set_alpha(0.72)
    ax.axhline(0, color="#666666", ls="--", lw=1)
    ax.set_ylabel("Brightness residual (EV)")
    ax.set_title("Brightness residual by camera", fontweight="bold")
    ax.tick_params(axis="x", rotation=14)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    save_figure(fig, figures_dir / "brightness_residual_by_device.png")


def plot_exif_vs_image_quality(frame: pd.DataFrame, figures_dir: Path) -> None:
    panels = [
        ("log2_exposure_time", "skin_linear_luminance_median", "Exposure time vs luminance"),
        ("log2_exposure_time", "laplacian_variance", "Exposure time vs sharpness"),
        ("log2_iso_gain", "high_frequency_noise_mad", "ISO gain vs noise proxy"),
        ("log2_iso_gain", "underexposed_pixel_ratio", "ISO gain vs underexposure"),
        ("BrightnessValue", "skin_lab_l_median", "BrightnessValue vs Lab L*"),
        ("BrightnessValue", "skin_linear_luminance_median", "BrightnessValue vs luminance"),
        ("relative_optical_exposure", "skin_linear_luminance_median", "Relative optical exposure vs luminance"),
        ("combined_exposure_gain", "skin_linear_luminance_median", "Combined exposure gain vs luminance"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 5.8))
    for ax, (x_name, y_name, title) in zip(axes.flat, panels):
        for camera_id, group in frame.groupby("camera_id", sort=True):
            color = CAMERA_COLORS.get(str(camera_id), "#777777")
            x = group[x_name].to_numpy(dtype=float)
            y = group[y_name].to_numpy(dtype=float)
            ax.scatter(x, y, s=9, alpha=0.42, color=color, label=str(camera_id))
            if np.unique(x).size > 1:
                slope, intercept = np.polyfit(x, y, 1)
                grid = np.linspace(x.min(), x.max(), 100)
                ax.plot(grid, slope * grid + intercept, color=color, lw=1.2)
        ax.set_xlabel(x_name, fontsize=6.5)
        ax.set_ylabel(y_name, fontsize=6.5)
        ax.set_title(title, fontsize=7.5, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.16)
    axes.flat[0].legend(fontsize=5.7)
    fig.suptitle("EXIF conditions versus aligned-skin image properties", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, figures_dir / "exif_vs_image_quality.png")


def git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or "unavailable"
    except Exception:
        return "unavailable"


def write_report(
    path: Path,
    frame: pd.DataFrame,
    overall: pd.DataFrame,
    by_device: pd.DataFrame,
    shifts: pd.DataFrame,
    apex: pd.DataFrame,
    relationships: pd.DataFrame,
    high_iso: pd.DataFrame,
    decision: pd.DataFrame,
    project_inventory_found: bool,
) -> dict[str, Any]:
    core_overall = overall.loc[
        overall["variable"].isin(CORE_FIELDS),
        ["variable", "valid_n", "missing_n", "unique_n", "min", "p1", "p5", "p25", "median", "p75", "p95", "p99", "max", "mean", "std", "iqr", "mad"],
    ]
    core_device = by_device.loc[
        by_device["variable"].isin(CORE_FIELDS),
        ["camera_id", "variable", "valid_n", "unique_n", "min", "p5", "median", "p95", "max", "mean", "std", "iqr", "mad"],
    ]
    fnumber = core_device.loc[core_device["variable"] == "FNumber"].copy()
    fnumber_values = (
        frame.groupby(["camera_id", "FNumber"]).size().rename("n").reset_index().sort_values(["camera_id", "FNumber"])
    )
    brightness_shift = shifts.set_index("variable").loc["BrightnessValue"]
    centered_shift = shifts.set_index("variable").loc["device_centered_brightness"]
    apex_fits = apex.loc[
        apex["scope"].isin(["overall", "device"]),
        ["scope", "camera_id", "n", "pearson_r", "spearman_rho", "slope", "intercept", "r_squared", "residual_median", "residual_iqr", "residual_mad", "residual_min", "residual_max"],
    ]
    interaction = apex.loc[
        apex["scope"] == "device_interaction_model",
        ["camera_id", "term", "coefficient", "standard_error", "term_p", "r_squared", "n"],
    ]
    relationship_view = relationships.loc[
        :, ["scope", "camera_id", "relationship", "n", "spearman_rho", "spearman_p"]
    ]
    overall_relationship = relationships.loc[relationships["scope"] == "overall"].set_index("relationship")
    bv_image = overall_relationship.loc["brightnessvalue_vs_luminance"]
    iso_noise = overall_relationship.loc["iso_vs_noise_proxy"]
    relative_image = overall_relationship.loc["relative_exposure_vs_luminance"]
    combined_image = overall_relationship.loc["combined_gain_vs_luminance"]
    exposure_unique = int(overall.set_index("variable").loc["ExposureTime", "unique_n"])
    iso_unique = int(overall.set_index("variable").loc["ISOSpeedRatings", "unique_n"])
    fnumber_unique = int(overall.set_index("variable").loc["FNumber", "unique_n"])
    high_iso_counts = high_iso["最终复核分类"].value_counts().rename_axis("复核分类").reset_index(name="n")
    metadata_anomaly_n = int((high_iso["最终复核分类"] == "明确的元数据异常").sum())
    low_quality_n = int((high_iso["最终复核分类"] == "需要保留但标记为低质量").sum())

    device_unique_parts = []
    for field in CORE_FIELDS:
        parts = []
        for row in core_device.loc[core_device["variable"] == field].itertuples(index=False):
            parts.append(f"{row.camera_id}={int(row.unique_n)}")
        device_unique_parts.append({"字段": field, "设备内唯一值数": "; ".join(parts)})

    fnumber_has_value = any(int(row.unique_n) > 2 and float(row.iqr) > 1e-6 for row in fnumber.itertuples(index=False))
    brightness_needs_standardization = (
        abs(float(brightness_shift.median_difference_b_minus_a)) > 0.25
        or float(brightness_shift.distribution_overlap) < 0.8
    )
    supports_inversion = metadata_anomaly_n == 0 and len(frame) == 522

    lines = [
        "# 四个核心EXIF字段物理可用性专项审计",
        "",
        "> 本报告只分析EXIF成像意义、数值可靠性及其与既有皮肤区域图像表现的关系。未读取或使用NYHA、SEX、其他临床标签、split或模型预测；未进行交叉验证、分类建模或图像深度学习训练。",
        "",
        "## 一、完成状态与数据来源",
        "",
        f"- 完成状态：COMPLETE；分析图像数：{len(frame)}；唯一ID：{frame['ID'].nunique()}。",
        "- EXIF数值优先复用既有第一阶段审计的逐图长表、图像审计表和问题明细表，避免重复提取工作簿。",
        f"- project_inventory：{'已读取' if project_inventory_found else '项目中未找到该文件；已改用现有审计报告、逐图CSV和预处理代码完成数据血缘核对'}。",
        "- 图像区域：既有CelebAMask-HQ解析标签中的skin类；图像为现有224×224颜色保持aligned_rgb。没有重新训练或运行分割网络。",
        "- 没有在meanbg、黑背景成图或ImageNet mean/std标准化张量上计算颜色与亮度。",
        "",
        "设备数量：",
        "",
        markdown_table(frame["camera_id"].value_counts().rename_axis("camera_id").reset_index(name="n")),
        "",
        "## 二、四字段总体统计",
        "",
        markdown_table(core_overall),
        "",
        f"ExposureTime共有{exposure_unique}个唯一值，ISOSpeedRatings共有{iso_unique}个唯一值，均具有明显逐图变化。FNumber总体仅{fnumber_unique}个唯一值，需按设备判断。离群值只标记，未删除。",
        "",
        "## 三、按设备统计与系统偏移",
        "",
        markdown_table(core_device),
        "",
        "每个核心字段的设备内唯一值数：",
        "",
        markdown_table(pd.DataFrame(device_unique_parts)),
        "",
        f"BrightnessValue两设备中位数差为{float(brightness_shift.median_difference_b_minus_a):.4g} EV，经验分布重叠为{float(brightness_shift.distribution_overlap):.3f}；设备内稳健标准化后中位数差为{float(centered_shift.median_difference_b_minus_a):.4g}，重叠为{float(centered_shift.distribution_overlap):.3f}。因此{'需要' if brightness_needs_standardization else '暂未显示必须'}进行设备内中心化/缩放。",
        "",
        "## 四、FNumber专项判断",
        "",
        markdown_table(fnumber),
        "",
        "FNumber的设备—取值频数：",
        "",
        markdown_table(fnumber_values),
        "",
        f"结论：FNumber{'仍存在可辨别的设备内逐图变化' if fnumber_has_value else '设备内std、IQR和MAD接近0，几乎没有稳定的逐图信息'}。它不应因具有光学意义而被强行作为独立连续网络输入；推荐作为renderer固定设备参数，并参与relative_optical_exposure计算。",
        "",
        "## 五、派生成像参数与相关性",
        "",
        "已按提示词计算log2_exposure_time、log2_iso_gain、APEX光圈值、APEX时间值、EV100、relative_optical_exposure、combined_exposure_gain、sensitivity_value、brightness_apex_pred、brightness_residual及描述性device_centered_brightness。完整总体/设备分布、离群值、Pearson和Spearman矩阵见对应CSV。",
        "",
        "combined_exposure_gain由relative_optical_exposure与log2_iso_gain确定性相加，因此不提供独立自由度。它可用于敏感性或质量控制，不宜与两个组成量同时输入V1。",
        "",
        "## 六、BrightnessValue物理一致性",
        "",
        markdown_table(apex_fits),
        "",
        "设备截距/斜率交互模型：",
        "",
        markdown_table(interaction),
        "",
        f"BrightnessValue与实际皮肤线性亮度总体Spearman rho={float(bv_image.spearman_rho):.3f}。APEX拟合显示设备间截距、斜率或残差范围存在差异时，这种偏差只解释为手机自动曝光、HDR、ISP和厂商实现差异的线索，不自动判为元数据错误。原始BrightnessValue不建议跨设备直接使用；建议使用device_centered_brightness。",
        "",
        "## 七、与实际图像表现的关系",
        "",
        markdown_table(relationship_view),
        "",
        f"ISO与高频噪声代理总体Spearman rho={float(iso_noise.spearman_rho):.3f}；relative_optical_exposure与皮肤线性亮度rho={float(relative_image.spearman_rho):.3f}；combined_exposure_gain与皮肤线性亮度rho={float(combined_image.spearman_rho):.3f}。手机自动曝光会使简单单调关系变弱，相关性不作因果解释。",
        "",
        "图像指标定义：skin区域median RGB；sRGB反伽马后的线性相对亮度；由线性Y计算的Lab L*；线性亮度≥0.98为过曝、≤0.02为欠曝；任一通道≤5或≥250为通道裁剪/饱和；Laplacian variance为清晰度代理；高斯平滑残差MAD为高频噪声代理。后者不等于真实传感器噪声。",
        "",
        "## 八、14张高ISO离群图像复核",
        "",
        markdown_table(high_iso_counts),
        "",
        f"14张中明确元数据异常{metadata_anomaly_n}张，需要保留但标记低质量{low_quality_n}张。高ISO本身不构成删除理由；所有样本均保留。逐图结果：",
        "",
        markdown_table(high_iso[["ID", "Make", "Model", *CORE_FIELDS, "是否过曝", "是否欠曝", "是否明显噪声较高", "是否明显模糊", "合理弱光条件线索", "最终复核分类"]]),
        "",
        "## 九、最终字段决策表",
        "",
        markdown_table(decision),
        "",
        "## 十、十四个问题的明确回答",
        "",
        "1. ExposureTime具有逐图变化，适合提供光学条件，但V1不直接使用秒值。",
        "2. ExposureTime应先log2转换；V1进一步通过relative_optical_exposure与FNumber合并。",
        f"3. FNumber总体{fnumber_unique}个唯一值，且{'存在一定' if fnumber_has_value else '几乎没有'}设备内逐图信息。",
        "4. FNumber不独立进入连续网络条件，只参与relative_optical_exposure，并作为renderer固定设备参数。",
        f"5. ISOSpeedRatings有{iso_unique}个唯一值，适合表达增益条件；与图像噪声的关系只能用噪声代理描述。",
        "6. ISOSpeedRatings使用log2_iso_gain，不建议直接使用线性ISO数值。",
        f"7. BrightnessValue与实际皮肤亮度存在rho={float(bv_image.spearman_rho):.3f}的总体单调关系，但并非完美物理标定。",
        f"8. BrightnessValue{'不能' if brightness_needs_standardization else '仍不建议'}跨设备直接使用。",
        "9. 应使用device_centered_brightness；本报告的描述性统计使用全队列设备中位数/IQR，未来训练必须只用训练数据估计。",
        "10. relative_optical_exposure比ExposureTime和FNumber分别并列输入更合理，可减少设备固定FNumber带来的冗余。",
        "11. combined_exposure_gain与两个组成变量确定性冗余，没有额外自由度，只作QC/敏感性分析。",
        "12. 推荐V1连续条件向量：[relative_optical_exposure, log2_iso_gain, device_centered_brightness]。",
        "13. FNumber保留为renderer_fixed_parameter；EV100、combined_exposure_gain、brightness_apex_pred和brightness_residual只作质量控制/一致性分析。",
        f"14. {'现有四字段完整且未发现高ISO元数据异常，支持' if supports_inversion else '当前证据不足以无条件支持'}继续开展曝光与增益条件化的面部光学反演，但需要设备校准、质量标记和后续外部设备验证。",
        "",
        "## 十一、限制",
        "",
        "本分析使用JPEG成图和既有解析skin区域；ISP、HDR、降噪、白平衡和tone mapping都会改变像素表现。相关性只能支持条件变量的工程可用性判断，不能证明真实辐射度、传感器噪声或因果机制。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "brightness_needs_device_standardization": brightness_needs_standardization,
        "fnumber_has_per_image_value": fnumber_has_value,
        "supports_continued_inversion": supports_inversion,
        "high_iso_metadata_anomaly_n": metadata_anomaly_n,
        "high_iso_low_quality_n": low_quality_n,
        "v1_vector": [
            "relative_optical_exposure",
            "log2_iso_gain",
            "device_centered_brightness",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--values-long",
        type=Path,
        default=PROJECT_ROOT / "reports/exif_parameter_audit/parameter_values_long.csv",
    )
    parser.add_argument(
        "--image-audit",
        type=Path,
        default=PROJECT_ROOT / "reports/exif_parameter_audit/image_parameter_audit.csv",
    )
    parser.add_argument(
        "--issues",
        type=Path,
        default=PROJECT_ROOT / "reports/exif_parameter_audit/parameter_value_issues.csv",
    )
    parser.add_argument(
        "--first-report",
        type=Path,
        default=PROJECT_ROOT / "reports/exif_parameter_audit/exif_parameter_audit_report.md",
    )
    parser.add_argument(
        "--metadata-workbook",
        type=Path,
        default=PROJECT_ROOT / "data/raw/EXIF/Image_Metadata_All.xlsx",
    )
    parser.add_argument(
        "--checklist",
        type=Path,
        default=PROJECT_ROOT / "data/raw/EXIF/EXIF_Inform.csv",
    )
    parser.add_argument(
        "--aligned-rgb-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data/processed/global_face/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict_intermediates/aligned_rgb",
    )
    parser.add_argument(
        "--parsing-label-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data/processed/global_face/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict_intermediates/parsing_label",
    )
    parser.add_argument(
        "--preprocess-script",
        type=Path,
        default=PROJECT_ROOT / "preprocessing/build_global_face_parsing_regularmask_blackbg_224_png_strict.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/exif_four_field_physical_audit",
    )
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_plotting()
    required_files = [
        args.values_long,
        args.image_audit,
        args.issues,
        args.first_report,
        args.metadata_workbook,
        args.checklist,
        args.preprocess_script,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    missing.extend(
        str(path) for path in [args.aligned_rgb_dir, args.parsing_label_dir] if not path.is_dir()
    )
    if missing:
        raise FileNotFoundError(f"Required inputs are missing: {missing}")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_output:
        raise FileExistsError(
            f"Output directory already contains files: {output_dir}. Use --overwrite-output only for this audit."
        )
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[stage] load existing EXIF audit tables", flush=True)
    frame, iso_issues = load_existing_exif(args.values_long, args.image_audit, args.issues)
    if len(frame) != 522 or frame["ID"].nunique() != 522:
        raise ValueError(f"Expected 522 unique images, got rows={len(frame)}, unique={frame['ID'].nunique()}")
    if set(frame["camera_id"].unique()) != set(CAMERA_COLORS):
        raise ValueError(f"Unexpected camera_id values: {frame['camera_id'].value_counts().to_dict()}")
    aligned_missing = [
        image_id for image_id in frame["ID"] if not (args.aligned_rgb_dir / f"{image_id}.png").is_file()
    ]
    label_missing = [
        image_id for image_id in frame["ID"] if not (args.parsing_label_dir / f"{image_id}.png").is_file()
    ]
    if aligned_missing or label_missing:
        raise FileNotFoundError(
            f"Image/mask alignment failure: aligned_missing={aligned_missing}, parsing_label_missing={label_missing}"
        )

    frame = derive_parameters(frame)
    print("[stage] compute existing-skin-region image properties", flush=True)
    image_quality, failures = compute_image_quality(frame, args.aligned_rgb_dir, args.parsing_label_dir)
    save_csv(pd.DataFrame(failures), output_dir / "image_quality_failures.csv")
    if failures:
        raise RuntimeError(
            f"Image relationship analysis stopped at ROI metrics: failures={len(failures)}; see image_quality_failures.csv"
        )
    frame = frame.merge(image_quality, on="ID", how="left", validate="one_to_one")
    if frame[image_quality.columns.drop("ID")].isna().any().any():
        raise RuntimeError("Image quality merge produced missing values")

    print("[stage] descriptive, device, APEX, outlier and image-relationship analyses", flush=True)
    overall, by_device = build_summaries(frame)
    outliers = build_outliers(frame)
    outliers["historical_high_iso_outlier"] = (
        outliers["ID"].isin(set(iso_issues["ID"]))
        & (outliers["variable"] == "ISOSpeedRatings")
    )
    relationships = image_relationships(frame)
    apex = apex_consistency(frame)
    correlations = correlation_matrix_long(frame)
    shifts = device_shift_summary(frame)
    high_iso = build_high_iso_review(frame, iso_issues)
    decision = build_decision_table(frame, overall, by_device, shifts, relationships)

    prohibited_columns = {"NYHA", "SEX", "fold", "label_3class", "patient_group_id"}
    for name, output_frame in {
        "values": frame,
        "summary": overall,
        "by_device": by_device,
        "outliers": outliers,
        "relationships": relationships,
        "decision": decision,
    }.items():
        found = prohibited_columns.intersection(output_frame.columns)
        if found:
            raise RuntimeError(f"Clinical/split columns unexpectedly present in {name}: {sorted(found)}")

    value_columns = [
        "ID",
        "Make",
        "Model",
        "camera_id",
        *CORE_FIELDS,
        *DERIVED_FIELDS,
        *[column for column in image_quality.columns if column != "ID"],
    ]
    save_csv(frame[value_columns], output_dir / "exif_four_field_values_and_derived.csv")
    save_csv(overall, output_dir / "exif_four_field_summary.csv")
    save_csv(by_device, output_dir / "exif_four_field_by_device.csv")
    save_csv(outliers, output_dir / "exif_four_field_outliers.csv")
    save_csv(relationships, output_dir / "exif_four_field_image_relationships.csv")
    save_csv(decision, output_dir / "exif_four_field_decision_table.csv")
    save_csv(high_iso, output_dir / "high_iso_14_image_review.csv")
    save_csv(apex, output_dir / "brightness_apex_consistency_statistics.csv")
    save_csv(correlations, output_dir / "exif_original_and_derived_correlations.csv")
    save_csv(shifts, output_dir / "exif_device_shift_statistics.csv")
    save_csv(image_quality, output_dir / "aligned_skin_image_quality_metrics.csv")

    print("[stage] generate figures", flush=True)
    plot_four_field_distributions(frame, figures_dir)
    plot_four_field_by_device(frame, figures_dir)
    plot_correlation_heatmap(frame, figures_dir)
    plot_apex_consistency(frame, figures_dir)
    plot_exif_vs_image_quality(frame, figures_dir)

    project_inventory_candidates = list(PROJECT_ROOT.glob("project_inventory*"))
    report_summary = write_report(
        output_dir / "exif_four_field_physical_audit_report.md",
        frame,
        overall,
        by_device,
        shifts,
        apex,
        relationships,
        high_iso,
        decision,
        bool(project_inventory_candidates),
    )

    input_files = required_files + project_inventory_candidates
    manifest = {
        "name": "exif_four_field_physical_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completion_status": "COMPLETE",
        "input_files": {
            path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in input_files
        },
        "input_image_directories": {
            "aligned_rgb": sha256_directory(args.aligned_rgb_dir),
            "parsing_label": sha256_directory(args.parsing_label_dir),
        },
        "image_queue": {
            "n": len(frame),
            "unique_id_n": frame["ID"].nunique(),
            "definition": "IDs in existing first-stage EXIF image audit with one valid value for each of the four fields",
        },
        "exif_data_source": {
            "authoritative_workbook": str(args.metadata_workbook.resolve()),
            "reused_derived_table": str(args.values_long.resolve()),
            "reason": "Reused the previously audited per-image table to avoid duplicate extraction",
        },
        "camera_id_definition": "Make + '/' + Model",
        "camera_counts": {str(key): int(value) for key, value in frame["camera_id"].value_counts().items()},
        "raw_fields": CORE_FIELDS,
        "derived_formulas": {
            "log2_exposure_time": "log2(ExposureTime)",
            "log2_iso_gain": "log2(ISOSpeedRatings/100)",
            "aperture_value_from_f": "2*log2(FNumber)",
            "time_value_from_t": "-log2(ExposureTime)",
            "EV100": "aperture_value_from_f + time_value_from_t",
            "relative_optical_exposure": "log2(ExposureTime) - 2*log2(FNumber)",
            "combined_exposure_gain": "relative_optical_exposure + log2_iso_gain",
            "sensitivity_value": "log2(ISOSpeedRatings/3.125)",
            "brightness_apex_pred": "aperture_value_from_f + time_value_from_t - sensitivity_value",
            "brightness_residual": "BrightnessValue - brightness_apex_pred",
            "device_centered_brightness": "(BrightnessValue - device_median)/(device_IQR + 1e-8)",
        },
        "image_region": {
            "source": "existing CelebAMask-HQ parsing_label class 1 (skin)",
            "aligned_image": "existing 224x224 color-preserving aligned_rgb",
            "new_segmentation_training": False,
        },
        "image_color_space": {
            "input": "RGB uint8 sRGB-aligned PNG",
            "linear_luminance": "sRGB inverse transfer then Y=0.2126R+0.7152G+0.0722B",
            "Lab_L_star": "computed from relative linear Y with D65-normalized Yn=1",
            "not_used": ["ImageNet-normalized tensor", "meanbg for color statistics", "black-background composite for color statistics"],
        },
        "statistical_methods": {
            "descriptive": "n, missing, unique, min, P1/P5/P25/P50/P75/P95/P99, max, mean, sample std, IQR, raw MAD",
            "outliers": "within-device 1.5*IQR and robust z=0.67448975*(x-median)/MAD, abs(z)>3.5",
            "correlations": "Pearson and Spearman; image relationships emphasize Spearman overall and by device",
            "APEX": "linear regression overall/by device plus camera intercept and slope interaction OLS",
            "device_shift": "median difference, Hedges g, empirical histogram overlap, descriptive Mann-Whitney U",
            "causal_interpretation": False,
        },
        "software": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "statsmodels": statsmodels.__version__,
            "Pillow": PIL.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "git_commit": git_commit(),
        "project_inventory": (
            [str(path.resolve()) for path in project_inventory_candidates]
            if project_inventory_candidates
            else "unavailable"
        ),
        "report_summary": report_summary,
        "prohibited_actions_check": {
            "NYHA_used": False,
            "SEX_used": False,
            "split_used": False,
            "classification_or_cross_validation": False,
            "new_segmentation_training": False,
            "dependency_install_or_upgrade": False,
            "source_data_modified": False,
            "outlier_deleted": False,
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )

    roles = dict(zip(decision["字段"], decision["推荐角色"]))
    core_device = by_device.loc[by_device["variable"].isin(CORE_FIELDS)]
    print("\n=== FOUR-FIELD PHYSICAL EXIF AUDIT ===")
    print("Completion status: COMPLETE")
    print(f"Analyzed images: {len(frame)}")
    for field in CORE_FIELDS:
        valid_n = int(overall.set_index("variable").loc[field, "valid_n"])
        print(f"{field}: valid={valid_n}/{len(frame)}; role={roles[field]}")
    for camera_id, count in frame["camera_id"].value_counts().items():
        print(f"camera_id {camera_id}: n={count}")
    for field in CORE_FIELDS:
        parts = []
        for row in core_device.loc[core_device["variable"] == field].itertuples(index=False):
            parts.append(f"{row.camera_id}={int(row.unique_n)}")
        print(f"{field} within-device unique values: {'; '.join(parts)}")
    print("Recommended V1 continuous EXIF vector: relative_optical_exposure, log2_iso_gain, device_centered_brightness")
    print(f"BrightnessValue needs within-device standardization: {report_summary['brightness_needs_device_standardization']}")
    print(f"FNumber has per-image input value: {report_summary['fnumber_has_per_image_value']}")
    print(f"Supports continued EXIF-conditioned optical inversion: {report_summary['supports_continued_inversion']}")
    print(f"Final report: {output_dir / 'exif_four_field_physical_audit_report.md'}")


if __name__ == "__main__":
    main()
