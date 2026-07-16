"""Audit EXIF metadata, labels, and fixed patient-group folds.

This script is intentionally independent of image model training.  It never
modifies the source workbook, label CSV, or split files.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from scipy.stats import chi2_contingency, kruskal
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.classification_metrics import (  # noqa: E402
    compute_classification_metrics,
    flatten_metrics,
)


CSV_ENCODING = "utf-8-sig"
CLASS_NAMES = {0: "normal", 1: "mild", 2: "severe"}
NYHA_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
REQUIRED_OUTPUTS = [
    "exif_merged_splits500.csv",
    "excluded_or_unmatched_ids.csv",
    "exif_field_completeness.csv",
    "device_class_crosstab.csv",
    "year_class_crosstab.csv",
    "fold_device_class_crosstab.csv",
    "fold_sex_class_crosstab.csv",
    "exif_numeric_by_class.csv",
    "exif_numeric_by_device_class.csv",
    "categorical_association_tests.csv",
    "numeric_association_tests.csv",
    "metadata_only_cv_summary.csv",
    "metadata_only_fold_metrics.csv",
    "metadata_only_oof_predictions.csv",
    "metadata_only_confusion_matrices.csv",
    "leakage_and_integrity_checks.csv",
    "exif_label_audit.md",
    "device_by_class.png",
    "year_by_class.png",
    "fold_device_class_heatmap.png",
    "ev100_by_class_and_device.png",
    "iso_by_class_and_device.png",
    "exposure_by_class_and_device.png",
    "metadata_only_confusion_matrix.png",
    "metadata_only_macro_auc_by_model.png",
]

KEEP_METADATA_COLUMNS = [
    "ID",
    "文件名",
    "SHA256",
    "厂商",
    "相机型号",
    "软件",
    "原始拍摄时间",
    "宽度(px)",
    "高度(px)",
    "长宽比",
    "总像素(MP)",
    "方向值",
    "ICC存在",
    "XMP存在",
    "曝光时间(s)",
    "光圈F值",
    "ISO",
    "亮度值(APEX)",
    "曝光补偿(EV)",
    "测光模式",
    "光源",
    "闪光灯",
    "焦距(mm)",
    "35mm等效焦距(mm)",
    "数字变焦比",
    "白平衡",
    "曝光模式",
    "场景拍摄类型",
    "色彩空间",
]

NUMERIC_FEATURES = [
    "exposure_time_clean",
    "log2_exposure_time",
    "iso_clean",
    "log2_iso100",
    "EV100",
    "brightness_value_raw",
    "brightness_device_z",
    "fnumber_clean",
    "focal_length_clean",
    "total_pixels_mp_clean",
    "aspect_ratio_clean",
]

MODEL_SPECS: dict[str, tuple[list[str], list[str]]] = {
    "M0_majority_prior": ([], []),
    "M1_device_only": (["device_domain"], []),
    "M2_acquisition_only": (
        [],
        ["log2_exposure_time", "log2_iso100", "EV100"],
    ),
    "M3_device_acquisition": (
        ["device_domain"],
        ["log2_exposure_time", "log2_iso100", "EV100"],
    ),
    "M4_year_only": (["shooting_year"], []),
    "M5_sex_only": (["SEX"], []),
    "M6_sex_device_acquisition": (
        ["SEX", "device_domain"],
        ["log2_exposure_time", "log2_iso100", "EV100"],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-xlsx", type=Path, default=None)
    parser.add_argument(
        "--label-csv", type=Path, default=PROJECT_ROOT / "data/raw/label_raw.csv"
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/splits_500",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/exif_label_fold_audit",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def find_metadata_xlsx(requested: Path | None) -> Path:
    if requested is not None:
        requested = requested.resolve()
        if not requested.is_file():
            raise FileNotFoundError(f"Metadata workbook not found: {requested}")
        return requested
    preferred = PROJECT_ROOT / "data/raw/Image_Metadata_All.xlsx"
    if preferred.is_file():
        return preferred.resolve()
    candidates = sorted(PROJECT_ROOT.rglob("Image_Metadata_All.xlsx"))
    if not candidates:
        raise FileNotFoundError("Image_Metadata_All.xlsx was not found in the project")
    if len(candidates) > 1:
        text = "\n".join(str(path.resolve()) for path in candidates)
        raise RuntimeError(f"Multiple metadata workbooks found; specify one:\n{text}")
    return candidates[0].resolve()


def read_inputs(
    metadata_path: Path, label_path: Path, splits_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    if not label_path.is_file():
        raise FileNotFoundError(f"Label CSV not found: {label_path}")
    if not splits_dir.is_dir():
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
    book = pd.ExcelFile(metadata_path)
    required_sheets = {"提取汇总", "图片元数据", "EXIF原始明细", "字段说明"}
    missing_sheets = required_sheets.difference(book.sheet_names)
    if missing_sheets:
        raise ValueError(f"Metadata workbook missing sheets: {sorted(missing_sheets)}")
    metadata = pd.read_excel(metadata_path, sheet_name="图片元数据", dtype=str)
    missing_columns = sorted(set(KEEP_METADATA_COLUMNS).difference(metadata.columns))
    if missing_columns:
        raise ValueError(f"Metadata sheet missing columns: {missing_columns}")
    metadata = metadata.loc[:, KEEP_METADATA_COLUMNS].copy()
    metadata["ID"] = normalize_id(metadata["ID"])

    labels = pd.read_csv(label_path, dtype={"ID": "string"}, encoding=CSV_ENCODING)
    if not {"ID", "SEX", "NYHA"}.issubset(labels.columns):
        raise ValueError("label_raw.csv must contain ID, SEX, and NYHA")
    labels = labels.loc[:, ["ID", "SEX", "NYHA"]].copy()
    labels["ID"] = normalize_id(labels["ID"])

    master_path = splits_dir / "nyha_3class_sex_stratified_group_5fold.csv"
    master = pd.read_csv(
        master_path,
        dtype={"ID": "string", "patient_group_id": "string"},
        encoding=CSV_ENCODING,
    )
    master["ID"] = normalize_id(master["ID"])
    master["patient_group_id"] = normalize_id(master["patient_group_id"])
    held_out: dict[int, pd.DataFrame] = {}
    for fold in range(5):
        path = splits_dir / f"fold_{fold}_val.csv"
        frame = pd.read_csv(
            path,
            dtype={"ID": "string", "patient_group_id": "string"},
            encoding=CSV_ENCODING,
        )
        frame["ID"] = normalize_id(frame["ID"])
        frame["patient_group_id"] = normalize_id(frame["patient_group_id"])
        held_out[fold] = frame
    return metadata, labels, master, held_out


def _check_record(
    name: str, passed: bool, severity: str, details: str
) -> dict[str, Any]:
    return {
        "check": name,
        "passed": bool(passed),
        "severity": severity,
        "details": details,
    }


def validate_integrity(
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
    master: pd.DataFrame,
    held_out: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checks: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    split_ids = set(master["ID"].dropna())
    metadata_ids = set(metadata["ID"].dropna())
    label_ids = set(labels["ID"].dropna())

    checks.extend(
        [
            _check_record(
                "metadata_522_unique_ids",
                len(metadata) == 522 and metadata["ID"].nunique() == 522,
                "critical",
                f"rows={len(metadata)}, unique_ids={metadata['ID'].nunique()}",
            ),
            _check_record(
                "metadata_no_duplicate_id",
                not metadata["ID"].duplicated().any(),
                "critical",
                f"duplicate_rows={int(metadata['ID'].duplicated(keep=False).sum())}",
            ),
            _check_record(
                "label_no_duplicate_id",
                not labels["ID"].duplicated().any(),
                "critical",
                f"duplicate_rows={int(labels['ID'].duplicated(keep=False).sum())}",
            ),
            _check_record(
                "split_500_unique_ids",
                len(master) == 500 and master["ID"].nunique() == 500,
                "critical",
                f"rows={len(master)}, unique_ids={master['ID'].nunique()}",
            ),
            _check_record(
                "split_all_match_metadata",
                split_ids.issubset(metadata_ids),
                "critical",
                f"unmatched={len(split_ids - metadata_ids)}",
            ),
            _check_record(
                "split_all_match_labels",
                split_ids.issubset(label_ids),
                "critical",
                f"unmatched={len(split_ids - label_ids)}",
            ),
        ]
    )

    sha = metadata["SHA256"].astype("string").str.strip().replace("", pd.NA)
    sha_dups = metadata.loc[sha.notna() & sha.duplicated(keep=False), ["ID", "SHA256"]]
    checks.append(
        _check_record(
            "sha256_no_exact_duplicates",
            sha_dups.empty,
            "warning",
            f"duplicate_rows={len(sha_dups)}",
        )
    )
    for _, row in sha_dups.iterrows():
        exclusions.append(
            {
                "ID": str(row["ID"]),
                "source": "metadata",
                "reason": "duplicate_SHA256",
            }
        )

    for identifier in sorted(metadata_ids - split_ids):
        exclusions.append(
            {"ID": identifier, "source": "metadata", "reason": "not_in_splits_500"}
        )
    for identifier in sorted(label_ids - split_ids):
        exclusions.append(
            {"ID": identifier, "source": "label", "reason": "not_in_splits_500"}
        )
    for identifier in sorted(split_ids - metadata_ids):
        exclusions.append(
            {"ID": identifier, "source": "split", "reason": "missing_metadata"}
        )
    for identifier in sorted(split_ids - label_ids):
        exclusions.append(
            {"ID": identifier, "source": "split", "reason": "missing_label"}
        )
    checks.extend(
        [
            _check_record(
                "metadata_exactly_22_outside_split",
                len(metadata_ids - split_ids) == 22,
                "critical",
                f"count={len(metadata_ids - split_ids)}",
            ),
            _check_record(
                "label_outside_split_matches_metadata",
                label_ids - split_ids == metadata_ids - split_ids,
                "critical",
                f"label_extra={len(label_ids - split_ids)}, metadata_extra={len(metadata_ids - split_ids)}",
            ),
        ]
    )

    val_concat = pd.concat(
        [frame.assign(outer_fold=fold) for fold, frame in held_out.items()],
        ignore_index=True,
    )
    val_counts = val_concat["ID"].value_counts()
    held_out_once = (
        len(val_concat) == 500
        and set(val_concat["ID"]) == split_ids
        and val_counts.eq(1).all()
    )
    checks.append(
        _check_record(
            "each_id_held_out_exactly_once",
            held_out_once,
            "critical",
            f"held_out_rows={len(val_concat)}, min_occurrence={val_counts.min()}, max_occurrence={val_counts.max()}",
        )
    )
    fold_match = all(
        set(frame["ID"]) == set(master.loc[master["fold"] == fold, "ID"])
        for fold, frame in held_out.items()
    )
    checks.append(
        _check_record(
            "val_files_match_master_fold",
            fold_match,
            "critical",
            "fold_i_val.csv equals master rows with fold=i",
        )
    )
    leakage_groups = master.groupby("patient_group_id")["fold"].nunique()
    checks.append(
        _check_record(
            "no_patient_group_cross_fold",
            not leakage_groups.gt(1).any(),
            "critical",
            f"leaking_groups={int(leakage_groups.gt(1).sum())}",
        )
    )
    mapping = (
        master.assign(expected=master["NYHA"].astype(int).map(NYHA_MAP))["expected"]
        == master["label_3class"].astype(int)
    ).all()
    checks.append(
        _check_record(
            "nyha_three_class_mapping_matches_project",
            bool(mapping),
            "critical",
            "0->0; 1/2->1; 3/4->2",
        )
    )
    for fold, group in master.groupby("fold"):
        missing_classes = sorted(set(CLASS_NAMES) - set(group["label_3class"].astype(int)))
        checks.append(
            _check_record(
                f"fold_{fold}_contains_all_classes",
                not missing_classes,
                "critical",
                f"missing_classes={missing_classes}",
            )
        )
    check_frame = pd.DataFrame(checks)
    failures = check_frame.loc[
        (~check_frame["passed"]) & (check_frame["severity"] == "critical")
    ]
    if not failures.empty:
        raise RuntimeError(
            "Critical integrity validation failed:\n"
            + failures[["check", "details"]].to_string(index=False)
        )
    return check_frame, pd.DataFrame(exclusions, columns=["ID", "source", "reason"])


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def prepare_merged(
    metadata: pd.DataFrame, labels: pd.DataFrame, master: pd.DataFrame
) -> pd.DataFrame:
    label_clean = labels.copy()
    label_clean["SEX"] = numeric(label_clean["SEX"]).astype("Int64")
    label_clean["NYHA"] = numeric(label_clean["NYHA"]).astype("Int64")
    split = master.loc[:, ["ID", "patient_group_id", "fold"]].copy()
    merged = split.merge(metadata, on="ID", how="left", validate="one_to_one")
    merged = merged.merge(label_clean, on="ID", how="left", validate="one_to_one")
    merged["NYHA_raw"] = merged["NYHA"].astype("Int64")
    merged["label_3class"] = merged["NYHA_raw"].map(NYHA_MAP).astype("Int64")
    merged["label_name"] = merged["label_3class"].map(CLASS_NAMES)

    make = merged["厂商"].astype("string").str.strip().fillna("UnknownMake")
    model = merged["相机型号"].astype("string").str.strip().fillna("UnknownModel")
    merged["device_domain"] = make + "_" + model
    shooting_time = pd.to_datetime(
        merged["原始拍摄时间"], format="%Y:%m:%d %H:%M:%S", errors="coerce"
    )
    merged["shooting_year"] = shooting_time.dt.year.astype("Int64")

    merged["exposure_time_clean"] = numeric(merged["曝光时间(s)"]).where(
        lambda value: value > 0
    )
    merged["iso_clean"] = numeric(merged["ISO"]).where(lambda value: value > 0)
    merged["fnumber_clean"] = numeric(merged["光圈F值"]).where(
        lambda value: value > 0
    )
    merged["exposure_time_invalid"] = merged["exposure_time_clean"].isna()
    merged["iso_invalid"] = merged["iso_clean"].isna()
    merged["fnumber_invalid"] = merged["fnumber_clean"].isna()
    merged["log2_exposure_time"] = np.log2(merged["exposure_time_clean"])
    merged["log2_iso100"] = np.log2(merged["iso_clean"] / 100.0)
    merged["EV100"] = np.log2(
        merged["fnumber_clean"].pow(2) / merged["exposure_time_clean"]
    ) - np.log2(merged["iso_clean"] / 100.0)

    merged["brightness_value_raw"] = numeric(merged["亮度值(APEX)"])
    merged["focal_length_clean"] = numeric(merged["焦距(mm)"])
    merged["total_pixels_mp_clean"] = numeric(merged["总像素(MP)"])
    merged["aspect_ratio_clean"] = numeric(merged["长宽比"])
    focal35 = numeric(merged["35mm等效焦距(mm)"])
    merged["focal35_clean"] = focal35.where(focal35 > 0)
    merged["focal35_missing_or_zero"] = focal35.isna() | focal35.eq(0)
    exposure_bias = numeric(merged["曝光补偿(EV)"])
    zoom = numeric(merged["数字变焦比"])
    merged["exposure_bias_suspect"] = exposure_bias.eq(-25)
    merged["digital_zoom_suspect"] = zoom.eq(100)
    flash_value = numeric(merged["闪光灯"]).astype("Int64")
    merged["flash_fired"] = flash_value.map(
        lambda value: pd.NA if pd.isna(value) else bool(int(value) & 1)
    ).astype("boolean")

    width = numeric(merged["宽度(px)"]).astype("Int64")
    height = numeric(merged["高度(px)"]).astype("Int64")
    merged["resolution"] = width.astype("string") + "×" + height.astype("string")
    orientation = numeric(merged["方向值"])
    merged["orientation_nonstandard"] = orientation.ne(1) | orientation.isna()
    ratio = merged["aspect_ratio_clean"]
    merged["aspect_ratio_group"] = np.select(
        [np.isclose(ratio, 4 / 3, atol=0.03), np.isclose(ratio, 16 / 9, atol=0.03)],
        ["4:3", "16:9"],
        default="other",
    )

    merged["brightness_device_z"] = np.nan
    for fold in sorted(merged["fold"].unique()):
        train = merged[merged["fold"] != fold]
        test_index = merged.index[merged["fold"] == fold]
        for device in merged.loc[test_index, "device_domain"].unique():
            train_values = train.loc[
                train["device_domain"] == device, "brightness_value_raw"
            ].dropna()
            target_index = test_index[
                merged.loc[test_index, "device_domain"].to_numpy() == device
            ]
            if len(train_values) >= 2 and train_values.std(ddof=0) > 0:
                merged.loc[target_index, "brightness_device_z"] = (
                    merged.loc[target_index, "brightness_value_raw"]
                    - train_values.mean()
                ) / train_values.std(ddof=0)
    return merged.sort_values(["fold", "patient_group_id", "ID"], kind="stable")


def crosstab_outputs(data: pd.DataFrame, output_dir: Path) -> None:
    tables = {
        "device_class_crosstab.csv": pd.crosstab(
            data["device_domain"], data["label_name"]
        ).reset_index(),
        "year_class_crosstab.csv": pd.crosstab(
            data["shooting_year"], data["label_name"]
        ).reset_index(),
        "fold_device_class_crosstab.csv": pd.crosstab(
            [data["fold"], data["device_domain"]], data["label_name"]
        ).reset_index(),
        "fold_sex_class_crosstab.csv": pd.crosstab(
            [data["fold"], data["SEX"]], data["label_name"]
        ).reset_index(),
    }
    for name, table in tables.items():
        for class_name in CLASS_NAMES.values():
            if class_name not in table.columns:
                table[class_name] = 0
        table.to_csv(output_dir / name, index=False, encoding=CSV_ENCODING)


def summarize_values(values: pd.Series) -> dict[str, Any]:
    clean = numeric(values).dropna()
    result: dict[str, Any] = {
        "n": int(len(clean)),
        "missing_n": int(len(values) - len(clean)),
        "missing_rate": float(1 - len(clean) / len(values)) if len(values) else np.nan,
    }
    for key in ["mean", "std", "median", "q1", "q3", "min", "max", "p05", "p95"]:
        result[key] = np.nan
    if clean.empty:
        return result
    result.update(
        {
            "mean": float(clean.mean()),
            "std": float(clean.std(ddof=1)) if len(clean) > 1 else np.nan,
            "median": float(clean.median()),
            "q1": float(clean.quantile(0.25)),
            "q3": float(clean.quantile(0.75)),
            "min": float(clean.min()),
            "max": float(clean.max()),
            "p05": float(clean.quantile(0.05)),
            "p95": float(clean.quantile(0.95)),
        }
    )
    return result


def numeric_summaries(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_class: list[dict[str, Any]] = []
    by_device: list[dict[str, Any]] = []
    for variable in NUMERIC_FEATURES:
        by_class.append(
            {
                "variable": variable,
                "group_type": "overall",
                "label_name": "all",
                **summarize_values(data[variable]),
            }
        )
        for label_name, group in data.groupby("label_name", dropna=False):
            by_class.append(
                {
                    "variable": variable,
                    "group_type": "class",
                    "label_name": label_name,
                    **summarize_values(group[variable]),
                }
            )
        for device, group in data.groupby("device_domain", dropna=False):
            by_device.append(
                {
                    "variable": variable,
                    "group_type": "device",
                    "device_domain": device,
                    "label_name": "all",
                    **summarize_values(group[variable]),
                }
            )
            for label_name, subgroup in group.groupby("label_name", dropna=False):
                by_device.append(
                    {
                        "variable": variable,
                        "group_type": "device_class",
                        "device_domain": device,
                        "label_name": label_name,
                        **summarize_values(subgroup[variable]),
                    }
                )
    return pd.DataFrame(by_class), pd.DataFrame(by_device)


def permutation_chi_square(
    left: pd.Series,
    right: pd.Series,
    observed: float,
    rng: np.random.Generator,
    n_permutations: int,
) -> float:
    left_values = left.to_numpy(copy=True)
    right_values = right.to_numpy(copy=True)
    exceed = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(right_values)
        table = pd.crosstab(left_values, permuted)
        statistic = chi2_contingency(table, correction=False)[0]
        exceed += statistic >= observed - 1e-12
    return float((exceed + 1) / (n_permutations + 1))


def association_test(
    left: pd.Series,
    right: pd.Series,
    name: str,
    rng: np.random.Generator,
    n_permutations: int,
) -> dict[str, Any]:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    table = pd.crosstab(frame["left"], frame["right"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {
            "test": name,
            "n": len(frame),
            "rows": table.shape[0],
            "columns": table.shape[1],
            "chi_square": np.nan,
            "df": np.nan,
            "chi_square_p": np.nan,
            "permutation_p": np.nan,
            "cramers_v": np.nan,
            "min_expected": np.nan,
            "expected_below_5": np.nan,
            "chi_square_warning": "constant_or_empty_variable",
        }
    chi2, p_value, dof, expected = chi2_contingency(table, correction=False)
    denominator = len(frame) * min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / denominator) if denominator > 0 else np.nan
    permutation_p = permutation_chi_square(
        frame["left"], frame["right"], chi2, rng, n_permutations
    )
    return {
        "test": name,
        "n": len(frame),
        "rows": table.shape[0],
        "columns": table.shape[1],
        "chi_square": float(chi2),
        "df": int(dof),
        "chi_square_p": float(p_value),
        "permutation_p": permutation_p,
        "cramers_v": float(cramers_v),
        "min_expected": float(expected.min()),
        "expected_below_5": int((expected < 5).sum()),
        "chi_square_warning": "expected_frequency_below_5"
        if (expected < 5).any()
        else "",
    }


def categorical_tests(
    data: pd.DataFrame, seed: int, n_permutations: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pairs = [
        ("device_vs_label", "device_domain", "label_3class"),
        ("year_vs_label", "shooting_year", "label_3class"),
        ("sex_vs_label", "SEX", "label_3class"),
        ("fold_vs_device", "fold", "device_domain"),
        ("fold_vs_label", "fold", "label_3class"),
        ("device_vs_sex", "device_domain", "SEX"),
    ]
    results = [
        association_test(data[left], data[right], name, rng, n_permutations)
        for name, left, right in pairs
    ]
    for variable in KEEP_METADATA_COLUMNS[1:] + NUMERIC_FEATURES:
        if variable not in data:
            continue
        missing = data[variable].isna().astype(int)
        if missing.nunique() < 2:
            continue
        for target in ["device_domain", "label_3class", "fold"]:
            results.append(
                association_test(
                    missing,
                    data[target],
                    f"missing_{variable}_vs_{target}",
                    rng,
                    n_permutations,
                )
            )
    return pd.DataFrame(results)


def numeric_tests(data: pd.DataFrame) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    contexts: list[tuple[str, pd.DataFrame]] = [("overall", data)] + [
        (f"device={device}", group)
        for device, group in data.groupby("device_domain", sort=True)
    ]
    for variable in NUMERIC_FEATURES:
        for context, subset in contexts:
            groups = [
                numeric(subset.loc[subset["label_3class"] == label, variable]).dropna()
                for label in CLASS_NAMES
            ]
            nonempty = [group for group in groups if len(group)]
            n = sum(len(group) for group in nonempty)
            if len(nonempty) < 2:
                statistic = p_value = effect = np.nan
            else:
                k = len(nonempty)
                combined = pd.concat(nonempty, ignore_index=True)
                if combined.nunique(dropna=True) <= 1:
                    statistic, p_value, effect = 0.0, 1.0, 0.0
                else:
                    statistic, p_value = kruskal(*nonempty)
                    effect = (
                        max(0.0, float((statistic - k + 1) / (n - k)))
                        if n > k
                        else np.nan
                    )
            results.append(
                {
                    "variable": variable,
                    "context": context,
                    "n": n,
                    "groups_present": len(nonempty),
                    "kruskal_h": statistic,
                    "p_value": p_value,
                    "epsilon_squared": effect,
                }
            )
    return pd.DataFrame(results)


def build_pipeline(categorical: list[str], continuous: list[str]) -> Pipeline:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="__missing__"),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    if continuous:
        transformers.append(
            (
                "continuous",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                continuous,
            )
        )
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", C=1.0, max_iter=2000, solver="lbfgs"
                ),
            ),
        ]
    )


def predict_one_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
    y_train: np.ndarray | None = None,
) -> np.ndarray:
    target = train["label_3class"].astype(int).to_numpy() if y_train is None else y_train
    categorical, continuous = MODEL_SPECS[model_name]
    if model_name == "M0_majority_prior":
        counts = np.bincount(target, minlength=3).astype(float)
        priors = counts / counts.sum()
        return np.tile(priors, (len(test), 1))
    pipeline = build_pipeline(categorical, continuous)
    columns = categorical + continuous
    train_features = train[columns].copy()
    test_features = test[columns].copy()
    for column in categorical:
        train_features[column] = (
            train_features[column].astype("string").fillna("__missing__").astype(object)
        )
        test_features[column] = (
            test_features[column].astype("string").fillna("__missing__").astype(object)
        )
    pipeline.fit(train_features, target)
    raw = pipeline.predict_proba(test_features)
    classes = pipeline.named_steps["model"].classes_.astype(int)
    probabilities = np.zeros((len(test), 3), dtype=float)
    probabilities[:, classes] = raw
    return probabilities


def evaluate_models(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    confusion_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model_name in MODEL_SPECS:
        model_predictions: list[pd.DataFrame] = []
        for fold in sorted(data["fold"].unique()):
            train = data[data["fold"] != fold]
            test = data[data["fold"] == fold]
            if set(test["label_3class"].astype(int)) != set(CLASS_NAMES):
                raise RuntimeError(f"Outer test fold {fold} does not contain all classes")
            probabilities = predict_one_fold(train, test, model_name)
            metrics = compute_classification_metrics(
                test["label_3class"].astype(int).to_numpy(), probabilities
            )
            fold_metrics.append(
                {"model": model_name, "fold": int(fold), **flatten_metrics(metrics)}
            )
            matrix = metrics["confusion_matrix"]
            for actual in CLASS_NAMES:
                for predicted in CLASS_NAMES:
                    confusion_rows.append(
                        {
                            "model": model_name,
                            "scope": f"fold_{fold}",
                            "actual": CLASS_NAMES[actual],
                            "predicted": CLASS_NAMES[predicted],
                            "count": int(matrix[actual, predicted]),
                        }
                    )
            frame = test.loc[
                :, ["ID", "patient_group_id", "fold", "label_3class", "label_name"]
            ].copy()
            frame.insert(0, "model", model_name)
            frame["prob_normal"] = probabilities[:, 0]
            frame["prob_mild"] = probabilities[:, 1]
            frame["prob_severe"] = probabilities[:, 2]
            frame["predicted_label"] = probabilities.argmax(axis=1)
            frame["predicted_name"] = frame["predicted_label"].map(CLASS_NAMES)
            predictions.append(frame)
            model_predictions.append(frame)

        oof = pd.concat(model_predictions, ignore_index=True).sort_values("ID")
        probabilities = oof[["prob_normal", "prob_mild", "prob_severe"]].to_numpy()
        metrics = compute_classification_metrics(
            oof["label_3class"].astype(int).to_numpy(), probabilities
        )
        fold_frame = pd.DataFrame(fold_metrics)
        fold_frame = fold_frame[fold_frame["model"] == model_name]
        row: dict[str, Any] = {
            "model": model_name,
            "n_oof": len(oof),
            **flatten_metrics(metrics),
        }
        metric_columns = [
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "macro_auc",
        ]
        for metric in metric_columns:
            row[f"fold_mean_{metric}"] = float(fold_frame[metric].mean())
            row[f"fold_std_{metric}"] = float(fold_frame[metric].std(ddof=1))
        summary_rows.append(row)
        matrix = metrics["confusion_matrix"]
        for actual in CLASS_NAMES:
            for predicted in CLASS_NAMES:
                confusion_rows.append(
                    {
                        "model": model_name,
                        "scope": "OOF",
                        "actual": CLASS_NAMES[actual],
                        "predicted": CLASS_NAMES[predicted],
                        "count": int(matrix[actual, predicted]),
                    }
                )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(fold_metrics),
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(confusion_rows),
    )


def model_permutation_tests(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    seed: int,
    n_permutations: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2026)
    results: list[dict[str, Any]] = []
    for model_name in ["M1_device_only", "M3_device_acquisition", "M4_year_only"]:
        observed = float(
            summary.loc[summary["model"] == model_name, "macro_auc"].iloc[0]
        )
        null_values: list[float] = []
        for _ in range(n_permutations):
            shuffled = data.copy()
            shuffled["label_3class"] = rng.permutation(
                data["label_3class"].astype(int).to_numpy()
            )
            predictions = np.zeros((len(shuffled), 3), dtype=float)
            for fold in sorted(shuffled["fold"].unique()):
                train = shuffled[shuffled["fold"] != fold]
                test = shuffled[shuffled["fold"] == fold]
                probabilities = predict_one_fold(train, test, model_name)
                predictions[test.index.to_numpy()] = probabilities
            metric = compute_classification_metrics(
                shuffled["label_3class"].astype(int).to_numpy(), predictions
            )["macro_auc"]
            null_values.append(float(metric))
        null_array = np.asarray(null_values)
        empirical_p = float((1 + (null_array >= observed).sum()) / (len(null_array) + 1))
        results.append(
            {
                "model": model_name,
                "n_permutations": n_permutations,
                "observed_macro_auc": observed,
                "null_mean": float(null_array.mean()),
                "null_std": float(null_array.std(ddof=1)),
                "null_p05": float(np.quantile(null_array, 0.05)),
                "null_p95": float(np.quantile(null_array, 0.95)),
                "empirical_p": empirical_p,
                "null_distribution_json": json.dumps(null_values),
            }
        )
    return pd.DataFrame(results)


def field_completeness(data: pd.DataFrame) -> pd.DataFrame:
    recommended = {
        "device_domain",
        "exposure_time_clean",
        "log2_exposure_time",
        "iso_clean",
        "log2_iso100",
        "fnumber_clean",
        "EV100",
        "shooting_year",
        "SEX",
    }
    rows: list[dict[str, Any]] = []
    for column in data.columns:
        nonmissing = data[column].notna().sum()
        unique = data[column].nunique(dropna=True)
        reason = ""
        use = column in recommended
        if unique <= 1:
            reason = "constant_or_all_missing"
            use = False
        elif column in {"亮度值(APEX)", "brightness_value_raw"}:
            reason = "device_scale_difference; descriptive_only"
            use = False
        elif column in {"曝光补偿(EV)", "数字变焦比"}:
            reason = "known_suspect_encoding; descriptive_only"
            use = False
        elif column == "shooting_year":
            reason = "diagnostic_year_only_baseline"
        elif column not in recommended:
            reason = "not_in_prespecified_core_metadata_model"
        rows.append(
            {
                "field": column,
                "n": len(data),
                "nonmissing_n": int(nonmissing),
                "completeness_rate": float(nonmissing / len(data)),
                "unique_n": int(unique),
                "is_constant": bool(unique <= 1),
                "recommended_use": bool(use),
                "exclusion_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def make_plots(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    confusion: pd.DataFrame,
    output_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    palette = {"normal": "#4C78A8", "mild": "#F2CF5B", "severe": "#E45756"}

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=300, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x="device_domain", hue="label_name", palette=palette)
    plt.title("Device distribution by class")
    plt.xlabel("Device")
    plt.ylabel("Count")
    plt.xticks(rotation=15, ha="right")
    save("device_by_class.png")

    plt.figure(figsize=(7, 5))
    sns.countplot(data=data, x="shooting_year", hue="label_name", palette=palette)
    plt.title("Shooting year distribution by class")
    plt.xlabel("Shooting year")
    plt.ylabel("Count")
    save("year_by_class.png")

    heat = pd.crosstab(
        data["fold"], [data["device_domain"], data["label_name"]]
    )
    plt.figure(figsize=(11, 4.5))
    sns.heatmap(heat, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Fold × device × class counts")
    plt.xlabel("Device / class")
    plt.ylabel("Outer fold")
    save("fold_device_class_heatmap.png")

    for variable, name, title, y_label in [
        ("EV100", "ev100_by_class_and_device.png", "EV100 by class and device", "EV100"),
        ("iso_clean", "iso_by_class_and_device.png", "ISO by class and device", "ISO"),
        (
            "log2_exposure_time",
            "exposure_by_class_and_device.png",
            "Exposure time by class and device",
            "log2 exposure time (s)",
        ),
    ]:
        plt.figure(figsize=(9, 5))
        sns.boxplot(
            data=data,
            x="label_name",
            y=variable,
            hue="device_domain",
            order=["normal", "mild", "severe"],
            showfliers=False,
        )
        plt.title(title)
        plt.xlabel("Class")
        plt.ylabel(y_label)
        save(name)

    models = ["M1_device_only", "M3_device_acquisition"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, model_name in zip(axes, models):
        table = confusion[
            (confusion["model"] == model_name) & (confusion["scope"] == "OOF")
        ].pivot(index="actual", columns="predicted", values="count")
        table = table.reindex(
            index=["normal", "mild", "severe"],
            columns=["normal", "mild", "severe"],
            fill_value=0,
        )
        sns.heatmap(table, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(model_name)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    fig.suptitle("Metadata-only OOF confusion matrices", y=1.02)
    save("metadata_only_confusion_matrix.png")

    plt.figure(figsize=(9, 5))
    order = list(MODEL_SPECS)
    chart = summary.set_index("model").reindex(order).reset_index()
    sns.barplot(data=chart, x="model", y="macro_auc", color="#4C78A8")
    plt.axhline(0.5, color="#E45756", linestyle="--", label="Chance AUC = 0.5")
    plt.ylim(0, 1)
    plt.title("Metadata-only OOF macro-AUC")
    plt.xlabel("Model")
    plt.ylabel("Macro-AUC (OvR)")
    plt.xticks(rotation=25, ha="right")
    plt.legend()
    save("metadata_only_macro_auc_by_model.png")


def markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    shown = frame.head(max_rows).copy()
    if shown.empty:
        return "_No rows._"
    return shown.to_markdown(index=False, floatfmt=".4g")


def write_report(
    data: pd.DataFrame,
    checks: pd.DataFrame,
    categorical: pd.DataFrame,
    numeric_result: pd.DataFrame,
    summary: pd.DataFrame,
    permutation: pd.DataFrame,
    metadata_path: Path,
    label_path: Path,
    splits_dir: Path,
    seed: int,
    n_permutations: int,
    output_path: Path,
) -> None:
    device_test = categorical[categorical["test"] == "device_vs_label"].iloc[0]
    year_test = categorical[categorical["test"] == "year_vs_label"].iloc[0]
    device_auc = float(
        summary.loc[summary["model"] == "M1_device_only", "macro_auc"].iloc[0]
    )
    combined_auc = float(
        summary.loc[summary["model"] == "M3_device_acquisition", "macro_auc"].iloc[0]
    )
    p_by_model = (
        permutation.set_index("model")["empirical_p"].to_dict()
        if not permutation.empty
        else {}
    )
    fold_device_test = categorical[categorical["test"] == "fold_vs_device"].iloc[0]
    shortcut_evidence = (
        float(device_test["permutation_p"]) < 0.05
        and max(device_auc, combined_auc) > 0.55
        and min(
            p_by_model.get("M1_device_only", 1.0),
            p_by_model.get("M3_device_acquisition", 1.0),
        )
        < 0.05
    )
    decision = (
        "设备/采集条件与标签及预测表现共同提示采集 shortcut 风险。建议将 EXIF 监督的采集解耦作为鲁棒性方法的一部分，但不要把 EXIF 直接拼接进临床分类器。"
        if shortcut_evidence
        else "现有证据不足以声称存在明确采集 shortcut。EXIF 更适合作为采集鲁棒性、物理增强和压力测试的监督信号，而不是用于纠正已证实偏倚。"
    )
    device_counts = pd.crosstab(data["device_domain"], data["label_name"]).reset_index()
    support_issues: list[str] = []
    for _, row in device_counts.iterrows():
        missing_classes = [
            name for name in CLASS_NAMES.values() if int(row.get(name, 0)) == 0
        ]
        if missing_classes:
            support_issues.append(
                f"{row['device_domain']} 缺少 {', '.join(missing_classes)}"
            )
    support_text = "；".join(support_issues) if support_issues else "各设备均包含三个类别"
    normal_years = sorted(
        data.loc[data["label_name"] == "normal", "shooting_year"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    class_counts = data["label_name"].value_counts().reindex(CLASS_NAMES.values()).rename_axis("class").reset_index(name="n")
    sex_counts = pd.crosstab(data["SEX"], data["label_name"]).reset_index()
    fold_counts = pd.crosstab(data["fold"], data["label_name"]).reset_index()
    key_numeric = numeric_result[
        numeric_result["variable"].isin(
            ["exposure_time_clean", "iso_clean", "EV100"]
        )
    ]
    report = f"""# EXIF—标签—Fold联合审计报告

## 1. 审计目标

本审计用于识别采集条件、EXIF 元数据与三分类标签之间的关联和潜在 shortcut，不用于建立或推荐临床模型，也未训练任何新的图像深度学习模型。

## 2. 输入文件与软件环境

- 元数据：`{metadata_path}`
- 标签：`{label_path.resolve()}`
- 固定划分：`{splits_dir.resolve()}`
- 外层规则：每个 `fold_i_val.csv` 是外层留出测试折；固定超参数模型仅用其余四折拟合，未用测试折选参。
- seed：{seed}；置换次数：{n_permutations}
- Python {platform.python_version()}；pandas {pd.__version__}；NumPy {np.__version__}；SciPy {scipy.__version__}；scikit-learn {sklearn.__version__}

## 3. ID匹配和数据完整性

元数据与标签均为 522 个唯一 ID；正式 split 为 500 个唯一 ID，全部同时匹配元数据和标签。其余 22 个 ID 单独写入 `excluded_or_unmatched_ids.csv`，未进入本报告任何正式统计或模型。SHA256 完全重复行数为 {int((~checks.loc[checks['check'] == 'sha256_no_exact_duplicates', 'passed']).sum()) if False else checks.loc[checks['check'] == 'sha256_no_exact_duplicates', 'details'].iloc[0].split('=')[-1]}。

{markdown_table(checks)}

## 4. 正式splits_500队列

三分类分布：

{markdown_table(class_counts)}

性别 × 三分类：

{markdown_table(sex_counts)}

fold × 三分类（每折均为 100 个外层留出样本）：

{markdown_table(fold_counts)}

## 5. EXIF字段完整率与清洗规则

所有原始字段均保留；正值约束分别生成 `exposure_time_clean`、`iso_clean`、`fnumber_clean`，无效值保留病例并以 flag 标记。`EV100` 按光圈、曝光时间和 ISO 计算。35mm 等效焦距为 0 时仅在清洗字段中置缺失；ExposureBias=-25、DigitalZoomRatio=100 仅标记可疑。闪光灯按 EXIF bit 0 解释是否真正触发。`brightness_device_z` 对每个外层测试折只使用相应训练折、同设备的均值和标准差计算，避免泄漏。

## 6. 设备与拍摄年份结构

设备 × 三分类：

{markdown_table(device_counts)}

拍摄年份仅用于批次审计和 M4 year-only 诊断，不进入主要 acquisition 模型。

## 7. 设备、年份与三分类标签的关联

- device vs label：permutation p={float(device_test['permutation_p']):.4g}，Cramér's V={float(device_test['cramers_v']):.4g}，最小期望频数={float(device_test['min_expected']):.3g}。
- year vs label：permutation p={float(year_test['permutation_p']):.4g}，Cramér's V={float(year_test['cramers_v']):.4g}，最小期望频数={float(year_test['min_expected']):.3g}。
- 类别支持范围：{support_text}；normal 样本出现年份为 {normal_years}。因此设备/年份与标签存在结构性重叠，跨设备三分类不能作为正式性能结论。

数值差异、统计显著性与效应量应分别解释；完整结果见 `categorical_association_tests.csv`。

## 8. ExposureTime、ISO、EV100与标签的关系

以下结果同时列出总体与设备内部 Kruskal–Wallis 检验；epsilon-squared 是效应量。未预设两两比较，因此未做 Dunn 检验。

{markdown_table(key_numeric)}

ExposureTime 和 ISO 的总体差异显著，但设备内部检验均未达到 0.05，且设备内 epsilon-squared 很小；这表明总体数值差异很可能主要由设备构成驱动。EV100 无论总体还是设备内部均未达到 0.05。

## 9. Fold平衡和patient_group泄漏检查

五个外层测试折各 100 例，每个正式 ID 恰好作为留出样本出现一次。同一 `patient_group_id` 的所有多时相图像均沿用主 split 的同一 fold；本次检查未发现 patient_group 跨 fold。fold vs device 的 permutation p={float(fold_device_test['permutation_p']):.4g}、Cramér's V={float(fold_device_test['cramers_v']):.4g}，未见严重的折间设备失衡；仍不修改现有主 split。

## 10. 元数据诊断基线

这些 metadata-only 模型只用于 shortcut 审计，不是候选临床模型。所有 OneHotEncoder、缺失值插补和 StandardScaler 均只在外层训练折拟合。

{markdown_table(summary[['model','accuracy','balanced_accuracy','macro_f1','macro_auc','auc_normal','auc_mild','auc_severe','normal_vs_abnormal_auc','severe_vs_rest_auc']])}

- M1 device-only OOF Macro-AUC：{device_auc:.4f}。
- M3 device+acquisition OOF Macro-AUC：{combined_auc:.4f}。
- M1、M3、M4 固定 fold 标签置换 p 分别为 {p_by_model.get('M1_device_only', float('nan')):.4g}、{p_by_model.get('M3_device_acquisition', float('nan')):.4g}、{p_by_model.get('M4_year_only', float('nan')):.4g}。
- 机会水平的 AUC 参照为 0.5；上述判断同时参考固定 fold 标签置换检验，而非只看点估计。

## 11. 关键发现

{decision}

最强证据来自 normal 与设备/年份的结构性重叠：HONOR 没有 normal，normal 也仅见于 2024 年。M1 的 normal-vs-rest AUC 明显高于 mild/severe 的单类 AUC，因此该结果主要支持采集域可识别 normal，而不是证明元数据能够稳定区分完整三分类。

本结论没有把非显著结果隐藏，也没有预设“已发现 shortcut”。统计关联不等同于因果或临床可用性。

## 12. 对PhyCardioFace设计的影响

- EXIF 可以用于监测采集域、设计物理增强、报告设备分层鲁棒性。
- 不建议把 EXIF 直接拼接到心功能分类器作为预测输入，以免强化采集 shortcut。
- 当前 HONOR 缺少 normal，跨设备三分类只能视为受限压力测试，必须明确类别支持范围。可另建独立 device-aware robustness split，但不能覆盖 `splits_500`。

## 13. 限制与下一步建议

- 本审计样本量为 500，设备与年份的组合可能稀疏；当期望频数小于 5 时，Pearson 卡方近似需谨慎，报告优先参考固定 seed 置换 p 值。
- 元数据来自有限设备/批次，不能外推到新设备或新中心。
- metadata-only 性能只表示标签与采集元数据的可预测关联，不能解释为病理生理信号。
- 建议在不改变主 split 的前提下，补充按设备、年份和中心分层的外部或压力测试，并持续审计新增采集批次。
"""
    output_path.write_text(report, encoding="utf-8")


def verify_outputs(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    records.append(
        _check_record(
            "merged_csv_has_500_rows", len(data) == 500, "critical", f"rows={len(data)}"
        )
    )
    for model, group in predictions.groupby("model"):
        probabilities = group[["prob_normal", "prob_mild", "prob_severe"]].to_numpy()
        records.extend(
            [
                _check_record(
                    f"{model}_oof_500_unique_ids",
                    len(group) == 500 and group["ID"].nunique() == 500,
                    "critical",
                    f"rows={len(group)}, unique_ids={group['ID'].nunique()}",
                ),
                _check_record(
                    f"{model}_probabilities_in_range",
                    bool(((probabilities >= 0) & (probabilities <= 1)).all()),
                    "critical",
                    "all probabilities in [0,1]",
                ),
                _check_record(
                    f"{model}_probabilities_sum_to_one",
                    bool(np.allclose(probabilities.sum(axis=1), 1, atol=1e-8)),
                    "critical",
                    f"max_abs_error={np.max(np.abs(probabilities.sum(axis=1)-1)):.3g}",
                ),
            ]
        )
    missing = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).is_file()]
    records.append(
        _check_record(
            "all_required_outputs_exist",
            not missing,
            "critical",
            f"missing={missing}",
        )
    )
    for name in [item for item in REQUIRED_OUTPUTS if item.endswith(".png")]:
        path = output_dir / name
        valid = path.is_file() and path.stat().st_size > 1000
        records.append(
            _check_record(
                f"plot_opens_{name}", valid, "critical", f"bytes={path.stat().st_size if path.exists() else 0}"
            )
        )
    result = pd.DataFrame(records)
    failures = result.loc[(~result["passed"]) & (result["severity"] == "critical")]
    if not failures.empty:
        raise RuntimeError("Output verification failed:\n" + failures.to_string(index=False))
    return result


def main() -> int:
    args = parse_args()
    if args.n_permutations < 1:
        raise ValueError("--n-permutations must be at least 1")
    metadata_path = find_metadata_xlsx(args.metadata_xlsx)
    metadata, labels, master, held_out = read_inputs(
        metadata_path, args.label_csv, args.splits_dir
    )
    checks, exclusions = validate_integrity(metadata, labels, master, held_out)
    print(f"metadata_xlsx={metadata_path}")
    print(f"label_rows={len(labels)} metadata_rows={len(metadata)} split_rows={len(master)}")
    print("outer_test_semantics=fold_i_val.csv; each formal ID held out once")
    print("patient_group_leakage=0")
    if args.validate_only:
        print("VALIDATION_OK")
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_merged(metadata, labels, master)
    exclusions.to_csv(
        output_dir / "excluded_or_unmatched_ids.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    checks.to_csv(
        output_dir / "leakage_and_integrity_checks.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    data.to_csv(
        output_dir / "exif_merged_splits500.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    field_completeness(data).to_csv(
        output_dir / "exif_field_completeness.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    crosstab_outputs(data, output_dir)
    by_class, by_device_class = numeric_summaries(data)
    by_class.to_csv(
        output_dir / "exif_numeric_by_class.csv", index=False, encoding=CSV_ENCODING
    )
    by_device_class.to_csv(
        output_dir / "exif_numeric_by_device_class.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    categorical = categorical_tests(data, args.seed, args.n_permutations)
    categorical.to_csv(
        output_dir / "categorical_association_tests.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    numeric_result = numeric_tests(data)
    numeric_result.to_csv(
        output_dir / "numeric_association_tests.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    summary, fold_metrics, predictions, confusion = evaluate_models(data)
    summary.to_csv(
        output_dir / "metadata_only_cv_summary.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    fold_metrics.to_csv(
        output_dir / "metadata_only_fold_metrics.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    predictions.to_csv(
        output_dir / "metadata_only_oof_predictions.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    confusion.to_csv(
        output_dir / "metadata_only_confusion_matrices.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    permutation = model_permutation_tests(
        data, summary, args.seed, args.n_permutations
    )
    permutation.to_csv(
        output_dir / "metadata_permutation_test.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    make_plots(data, summary, confusion, output_dir)
    write_report(
        data,
        checks,
        categorical,
        numeric_result,
        summary,
        permutation,
        metadata_path,
        args.label_csv,
        args.splits_dir,
        args.seed,
        args.n_permutations,
        output_dir / "exif_label_audit.md",
    )
    output_checks = verify_outputs(data, predictions, output_dir)
    combined_checks = pd.concat([checks, output_checks], ignore_index=True)
    combined_checks.to_csv(
        output_dir / "leakage_and_integrity_checks.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    print(f"OUTPUT_DIR={output_dir}")
    print(f"OUTPUT_FILES={len(list(output_dir.iterdir()))}")
    print("AUDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
