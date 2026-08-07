from __future__ import annotations

import numpy as np
import pandas as pd


STANDARD_COLUMNS = {
    "camera_make": "camera_make",
    "camera_model": "camera_model",
    "image_width": "image_width",
    "image_height": "image_height",
    "software": "software",
    "exposure_time_seconds": "exposure_time_seconds",
    "iso": "iso",
    "brightness_value_apex": "brightness_value_apex",
    "f_number": "f_number",
    "exposure_bias_ev": "exposure_bias_ev",
    "flash": "flash",
    "metering_mode": "metering_mode",
    "white_balance": "white_balance",
    "exposure_mode": "exposure_mode",
    "capture_time": "capture_time",
}

BASE_NUMERIC = ("brightness_value_apex", "log2_iso", "log2_exposure_time")
Z_MAP = {
    "brightness_value_apex": "brightness_device_z",
    "log2_iso": "log2_iso_device_z",
    "log2_exposure_time": "log2_exposure_device_z",
}


def derive_metadata_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    out = frame.copy()
    invalid: dict[str, int] = {}
    for col in ("exposure_time_seconds", "iso", "brightness_value_apex", "f_number", "exposure_bias_ev", "image_width", "image_height"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if {"image_width", "image_height"}.issubset(out.columns):
        out["megapixels"] = out["image_width"] * out["image_height"] / 1_000_000.0
    if "exposure_time_seconds" in out.columns:
        valid = out["exposure_time_seconds"] > 0
        invalid["nonpositive_exposure_time_seconds"] = int((out["exposure_time_seconds"].notna() & ~valid).sum())
        out["log2_exposure_time"] = np.where(valid, np.log2(out["exposure_time_seconds"]), np.nan)
    if "iso" in out.columns:
        valid = out["iso"] > 0
        invalid["nonpositive_iso"] = int((out["iso"].notna() & ~valid).sum())
        out["log2_iso"] = np.where(valid, np.log2(out["iso"]), np.nan)
    if "capture_time" in out.columns:
        dt = pd.to_datetime(out["capture_time"], errors="coerce", format="%Y:%m:%d %H:%M:%S")
        fallback = pd.to_datetime(out["capture_time"], errors="coerce")
        dt = dt.fillna(fallback)
        out["capture_year"] = dt.dt.year.astype("Int64").astype(str).replace("<NA>", np.nan)
        out["capture_month"] = dt.dt.month.astype("Int64").astype(str).replace("<NA>", np.nan)
        out["capture_year_month_raw"] = dt.dt.strftime("%Y-%m")
        out["capture_hour"] = dt.dt.hour.astype("Int64")
        out["capture_hour_bin"] = pd.cut(
            out["capture_hour"].astype(float),
            bins=[-1, 5, 11, 17, 23],
            labels=["night", "morning", "afternoon", "evening"],
        ).astype(object)
        counts = out["capture_year_month_raw"].value_counts(dropna=True)
        rare = set(counts[counts < 10].index)
        out["capture_year_month"] = out["capture_year_month_raw"].where(~out["capture_year_month_raw"].isin(rare), "rare_or_sparse")
    return out, invalid


def add_oof_device_z(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    out = frame.copy()
    diagnostics: dict[str, object] = {"unknown_test_devices": {}, "zero_std_counts": {}}
    for raw, zcol in Z_MAP.items():
        out[zcol] = np.nan
    for fold in sorted(out["fold"].dropna().astype(int).unique()):
        train = out[out["fold"].astype(int) != fold]
        test_idx = out.index[out["fold"].astype(int) == fold]
        for raw, zcol in Z_MAP.items():
            stats = train.groupby("camera_model")[raw].agg(["mean", "std"])
            unknown = sorted(set(out.loc[test_idx, "camera_model"].dropna().astype(str)) - set(stats.index.astype(str)))
            if unknown:
                diagnostics["unknown_test_devices"][str(fold)] = unknown
            zero_std = int((stats["std"].fillna(0) == 0).sum())
            diagnostics["zero_std_counts"][f"fold_{fold}_{raw}"] = zero_std
            means = out.loc[test_idx, "camera_model"].map(stats["mean"])
            stds = out.loc[test_idx, "camera_model"].map(stats["std"]).replace(0, np.nan)
            out.loc[test_idx, zcol] = (out.loc[test_idx, raw] - means) / stds
    for raw, zcol in Z_MAP.items():
        desc_col = "descriptive_only_" + zcol
        stats = out.groupby("camera_model")[raw].agg(["mean", "std"])
        out[desc_col] = (out[raw] - out["camera_model"].map(stats["mean"])) / out["camera_model"].map(stats["std"]).replace(0, np.nan)
    return out, diagnostics


def fold_z_features(frame: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, dict[str, object]]:
    out = frame.copy()
    train = out[out["fold"].astype(int) != int(fold)]
    test = out[out["fold"].astype(int) == int(fold)]
    diagnostics: dict[str, object] = {"fold": int(fold), "unknown_test_devices": [], "zero_std": {}}
    for raw, zcol in Z_MAP.items():
        stats = train.groupby("camera_model")[raw].agg(["mean", "std"])
        unknown = sorted(set(test["camera_model"].dropna().astype(str)) - set(stats.index.astype(str)))
        if unknown:
            diagnostics["unknown_test_devices"] = unknown
            raise ValueError(f"fold {fold} has test camera_model not present in training fold: {unknown}")
        diagnostics["zero_std"][raw] = int((stats["std"].fillna(0) == 0).sum())
        means = out["camera_model"].map(stats["mean"])
        stds = out["camera_model"].map(stats["std"]).replace(0, np.nan)
        out[zcol] = (out[raw] - means) / stds
    return out, diagnostics


def feature_inventory(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_cols = {
        "META-D": {"camera_model"},
        "META-L": {"brightness_device_z", "log2_iso_device_z", "log2_exposure_device_z", "exposure_bias_ev", "flash"},
        "META-DL": {"camera_model", "brightness_device_z", "log2_iso_device_z", "log2_exposure_device_z", "exposure_bias_ev", "flash"},
        "META-T": {"capture_year", "capture_month", "capture_year_month", "capture_hour_bin"},
        "META-ALL": {
            "camera_make",
            "camera_model",
            "image_width",
            "image_height",
            "megapixels",
            "software",
            "brightness_device_z",
            "log2_iso_device_z",
            "log2_exposure_device_z",
            "exposure_bias_ev",
            "flash",
            "metering_mode",
            "white_balance",
            "exposure_mode",
            "capture_year",
            "capture_month",
            "capture_year_month",
            "capture_hour_bin",
        },
    }
    metadata_cols = [c for c in master.columns if c.startswith("metadata__") or c in set().union(*model_cols.values())]
    seen = set()
    for col in metadata_cols:
        std = col.replace("metadata__", "")
        if std in seen:
            continue
        seen.add(std)
        series = master[col] if col in master.columns else master[std]
        non_missing = int(series.notna().sum())
        unique = int(series.nunique(dropna=True))
        top_prop = float(series.value_counts(normalize=True, dropna=True).iloc[0]) if non_missing and unique else 0.0
        reasons = []
        if unique <= 1:
            reasons.append("constant_or_single_valid_value")
        if top_prop >= 0.98:
            reasons.append("near_constant")
        if any(token in std.lower() for token in ("path", "sha", "文件名", "绝对路径", "相对路径")):
            reasons.append("path_or_nonbiological_id")
        rows.append(
            {
                "original_column": col,
                "standardized_column": std,
                "dtype": str(series.dtype),
                "missing_rate": float(series.isna().mean()),
                "unique_values": unique,
                "top_value_proportion": top_prop,
                "enter_descriptive_statistics": bool(not reasons or "path_or_nonbiological_id" not in reasons),
                "enter_META_D": bool(std in model_cols["META-D"] and not reasons),
                "enter_META_L": bool(std in model_cols["META-L"] and not reasons),
                "enter_META_DL": bool(std in model_cols["META-DL"] and not reasons),
                "enter_META_T": bool(std in model_cols["META-T"] and not reasons),
                "enter_META_ALL": bool(std in model_cols["META-ALL"] and not reasons),
                "exclusion_reason": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows)
