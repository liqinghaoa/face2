"""Build a fixed patient-grouped, sex-stratified 5-fold split for NYHA 3-class experiments."""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"E:\projects\face2")
LABEL_CSV = PROJECT_ROOT / "data" / "raw" / "label_raw.csv"
PROCESSED_IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "global_face"
    / "global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict"
    / "images"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
N_SPLITS = 5
RANDOM_SEED = 2026
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
CSV_ENCODING = "utf-8-sig"
STRATA_ORDER = ("0_0", "0_1", "1_0", "1_1", "2_0", "2_1")
NYHA_ORDER = (0, 1, 2, 3, 4)
MAX_REFINEMENT_PASSES = 50
BALANCE_RAW_NYHA = False
RAW_NYHA_BALANCE_WEIGHT = 20.0
MAX_SWAP_REFINEMENT_PASSES = 100

OUTPUT_COLUMNS = [
    "ID",
    "patient_group_id",
    "SEX",
    "sex_name",
    "NYHA",
    "label_3class",
    "label_3class_name",
    "image_path",
    "stratum",
    "fold",
]


def _normalize_identifier(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def load_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"输入标签文件不存在：{path}")
    try:
        frame = pd.read_csv(path, dtype={"ID": "string"}, encoding=CSV_ENCODING)
    except Exception as exc:
        raise RuntimeError(f"读取 CSV 失败：{path}") from exc

    required = {"ID", "SEX", "NYHA"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"标签文件缺少必要列：{', '.join(missing)}")
    return frame.loc[:, ["ID", "SEX", "NYHA"]].copy()


def clean_labels(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = frame.copy()
    clean["ID"] = clean["ID"].map(_normalize_identifier)
    clean["SEX"] = pd.to_numeric(clean["SEX"], errors="coerce")
    clean["NYHA"] = pd.to_numeric(clean["NYHA"], errors="coerce")

    issues: list[dict[str, object]] = []
    for row_number, row in clean.iterrows():
        row_issues: list[str] = []
        if row["ID"] is None:
            row_issues.append("missing_ID")
        if pd.isna(row["SEX"]):
            row_issues.append("missing_SEX")
        elif row["SEX"] not in (0, 1):
            row_issues.append("invalid_SEX")
        if pd.isna(row["NYHA"]):
            row_issues.append("missing_NYHA")
        elif row["NYHA"] not in (0, 1, 2, 3, 4):
            row_issues.append("invalid_NYHA")
        if row_issues:
            issues.append(
                {
                    "excel_row": row_number + 2,
                    "ID": row["ID"],
                    "SEX": row["SEX"],
                    "NYHA": row["NYHA"],
                    "issue": ";".join(row_issues),
                }
            )

    valid_ids = clean["ID"].dropna()
    duplicated_ids = set(valid_ids[valid_ids.duplicated(keep=False)].tolist())
    for row_number, row in clean[clean["ID"].isin(duplicated_ids)].iterrows():
        issues.append(
            {
                "excel_row": row_number + 2,
                "ID": row["ID"],
                "SEX": row["SEX"],
                "NYHA": row["NYHA"],
                "issue": "duplicated_ID",
            }
        )

    invalid_columns = ["excel_row", "ID", "SEX", "NYHA", "issue"]
    invalid = pd.DataFrame(issues, columns=invalid_columns)
    invalid.to_csv(
        output_dir / "invalid_records.csv", index=False, encoding=CSV_ENCODING
    )
    if not invalid.empty:
        raise ValueError(
            f"发现 {len(invalid)} 条无效记录，详情已保存至："
            f"{output_dir / 'invalid_records.csv'}"
        )

    clean["SEX"] = clean["SEX"].astype("int64")
    clean["NYHA"] = clean["NYHA"].astype("int64")
    return clean.reset_index(drop=True)


def build_patient_group_id(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["patient_group_id"] = result["ID"].str.replace(
        r"-\d+$", "", regex=True
    )
    inconsistent_sex = (
        result.groupby("patient_group_id", sort=True)["SEX"].nunique().loc[lambda x: x > 1]
    )
    if not inconsistent_sex.empty:
        groups = ", ".join(inconsistent_sex.index.astype(str).tolist())
        raise ValueError(f"同一 patient_group_id 内 SEX 不一致：{groups}")
    return result


def map_nyha_to_3class(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["label_3class"] = result["NYHA"].map(
        {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
    )
    result["label_3class_name"] = result["label_3class"].map(
        {0: "normal", 1: "mild", 2: "severe"}
    )
    result["sex_name"] = result["SEX"].map({0: "female", 1: "male"})
    return result


def find_image_path(identifier: str, image_root: Path) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_root / f"{identifier}{extension}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def match_images(frame: pd.DataFrame, image_root: Path, output_dir: Path) -> pd.DataFrame:
    if not image_root.is_dir():
        raise FileNotFoundError(f"处理后图像目录不存在：{image_root}")

    result = frame.copy()
    paths = [find_image_path(identifier, image_root) for identifier in result["ID"]]
    result["image_path"] = [str(path) if path else None for path in paths]

    missing_mask = result["image_path"].isna()
    missing_columns = [
        "ID",
        "patient_group_id",
        "SEX",
        "NYHA",
        "expected_patterns",
    ]
    if missing_mask.any():
        missing = result.loc[
            missing_mask, ["ID", "patient_group_id", "SEX", "NYHA"]
        ].copy()
        missing["expected_patterns"] = missing["ID"].map(
            lambda identifier: ";".join(
                str(image_root / f"{identifier}{extension}")
                for extension in IMAGE_EXTENSIONS
            )
        )
    else:
        missing = pd.DataFrame(columns=missing_columns)
    missing.to_csv(
        output_dir / "missing_images.csv", index=False, encoding=CSV_ENCODING
    )
    if not missing.empty:
        raise FileNotFoundError(
            f"发现 {len(missing)} 个标签缺少处理后图像，详情已保存至："
            f"{output_dir / 'missing_images.csv'}"
        )
    return result


def build_stratum(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["stratum"] = (
        result["label_3class"].astype(str) + "_" + result["SEX"].astype(str)
    )
    small_strata = result["stratum"].value_counts().loc[lambda x: x < N_SPLITS]
    if not small_strata.empty:
        warnings.warn(
            "以下 stratum 样本数小于 5，分层结果可能不稳定："
            + ", ".join(f"{name}={count}" for name, count in small_strata.items()),
            stacklevel=2,
        )
    return result


def run_split(frame: pd.DataFrame) -> pd.DataFrame:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED
    )
    result = frame.copy()
    result["fold"] = -1
    x = result.index.to_numpy()
    for fold, (_, validation_indices) in enumerate(
        splitter.split(
            X=x,
            y=result["stratum"],
            groups=result["patient_group_id"],
        )
    ):
        result.loc[validation_indices, "fold"] = fold
    if (result["fold"] < 0).any():
        raise RuntimeError("部分样本未被分配 fold。")
    result["fold"] = result["fold"].astype("int64")
    return refine_split_balance(result)


def refine_split_balance_legacy(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministically improve the SGKF result by moving whole patient groups.

    The main objective is the six label_3class × SEX strata. Fold size and
    NYHA 4 dispersion are soft quality-control terms, not hard constraints.
    """
    result = frame.copy()
    grouped: list[tuple[str, np.ndarray, float, float]] = []
    for patient_group_id, group in result.groupby("patient_group_id", sort=True):
        stratum_counts = np.array(
            [(group["stratum"] == stratum).sum() for stratum in STRATA_ORDER],
            dtype=float,
        )
        grouped.append(
            (
                str(patient_group_id),
                stratum_counts,
                float(len(group)),
                float((group["NYHA"] == 4).sum()),
            )
        )

    assignments = np.array(
        [
            int(
                result.loc[
                    result["patient_group_id"] == patient_group_id, "fold"
                ].iloc[0]
            )
            for patient_group_id, _, _, _ in grouped
        ],
        dtype=int,
    )
    stratum_targets = np.array(
        [
            float((result["stratum"] == stratum).sum()) / N_SPLITS
            for stratum in STRATA_ORDER
        ]
    )
    size_target = len(result) / N_SPLITS
    nyha4_target = float((result["NYHA"] == 4).sum()) / N_SPLITS

    fold_strata = np.zeros((N_SPLITS, len(STRATA_ORDER)), dtype=float)
    fold_sizes = np.zeros(N_SPLITS, dtype=float)
    fold_nyha4 = np.zeros(N_SPLITS, dtype=float)
    for assignment, (_, vector, size, nyha4_count) in zip(assignments, grouped):
        fold_strata[assignment] += vector
        fold_sizes[assignment] += size
        fold_nyha4[assignment] += nyha4_count

    def fold_score(fold: int) -> float:
        stratum_score = np.sum(
            (fold_strata[fold] - stratum_targets) ** 2
            / np.maximum(stratum_targets, 1.0)
        )
        size_score = 0.5 * (fold_sizes[fold] - size_target) ** 2 / max(
            size_target, 1.0
        )
        nyha4_score = 0.75 * (fold_nyha4[fold] - nyha4_target) ** 2 / max(
            nyha4_target, 1.0
        )
        return float(stratum_score + size_score + nyha4_score)

    for pass_number in range(MAX_REFINEMENT_PASSES):
        improved = False
        order = np.random.default_rng(RANDOM_SEED + pass_number).permutation(
            len(grouped)
        )
        for group_index in order:
            _, vector, size, nyha4_count = grouped[group_index]
            source_fold = int(assignments[group_index])
            best_fold = source_fold
            best_delta = 0.0
            source_score = fold_score(source_fold)

            for target_fold in range(N_SPLITS):
                if target_fold == source_fold:
                    continue
                old_score = source_score + fold_score(target_fold)
                fold_strata[source_fold] -= vector
                fold_strata[target_fold] += vector
                fold_sizes[source_fold] -= size
                fold_sizes[target_fold] += size
                fold_nyha4[source_fold] -= nyha4_count
                fold_nyha4[target_fold] += nyha4_count
                delta = (
                    fold_score(source_fold) + fold_score(target_fold) - old_score
                )
                fold_strata[source_fold] += vector
                fold_strata[target_fold] -= vector
                fold_sizes[source_fold] += size
                fold_sizes[target_fold] -= size
                fold_nyha4[source_fold] += nyha4_count
                fold_nyha4[target_fold] -= nyha4_count

                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_fold = target_fold

            if best_fold != source_fold:
                fold_strata[source_fold] -= vector
                fold_strata[best_fold] += vector
                fold_sizes[source_fold] -= size
                fold_sizes[best_fold] += size
                fold_nyha4[source_fold] -= nyha4_count
                fold_nyha4[best_fold] += nyha4_count
                assignments[group_index] = best_fold
                improved = True

        if not improved:
            break

    group_to_fold = {
        patient_group_id: int(fold)
        for (patient_group_id, _, _, _), fold in zip(grouped, assignments)
    }
    result["fold"] = result["patient_group_id"].map(group_to_fold).astype("int64")
    return result


def refine_split_balance(frame: pd.DataFrame) -> pd.DataFrame:
    """Improve fold balance while preserving whole patient groups.

    BALANCE_RAW_NYHA=False keeps the original objective: label_3class x SEX
    strata, fold size, and NYHA 4 dispersion.

    BALANCE_RAW_NYHA=True adds raw NYHA 0-4 balance and then performs
    deterministic equal-size group swaps. This is useful for the 500-sample
    table where NYHA 1 and NYHA 2 should remain balanced separately even
    though both map to the same 3-class label.
    """
    result = frame.copy()
    grouped: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    for patient_group_id, group in result.groupby("patient_group_id", sort=True):
        stratum_counts = np.array(
            [(group["stratum"] == stratum).sum() for stratum in STRATA_ORDER],
            dtype=float,
        )
        nyha_counts = np.array(
            [(group["NYHA"] == nyha).sum() for nyha in NYHA_ORDER],
            dtype=float,
        )
        grouped.append(
            (
                str(patient_group_id),
                stratum_counts,
                nyha_counts,
                float(len(group)),
            )
        )

    assignments = np.array(
        [
            int(
                result.loc[
                    result["patient_group_id"] == patient_group_id, "fold"
                ].iloc[0]
            )
            for patient_group_id, _, _, _ in grouped
        ],
        dtype=int,
    )
    stratum_targets = np.array(
        [
            float((result["stratum"] == stratum).sum()) / N_SPLITS
            for stratum in STRATA_ORDER
        ]
    )
    nyha_targets = np.array(
        [float((result["NYHA"] == nyha).sum()) / N_SPLITS for nyha in NYHA_ORDER]
    )
    size_target = len(result) / N_SPLITS

    fold_strata = np.zeros((N_SPLITS, len(STRATA_ORDER)), dtype=float)
    fold_nyha = np.zeros((N_SPLITS, len(NYHA_ORDER)), dtype=float)
    fold_sizes = np.zeros(N_SPLITS, dtype=float)
    for assignment, (_, stratum_vector, nyha_vector, size) in zip(
        assignments, grouped
    ):
        fold_strata[assignment] += stratum_vector
        fold_nyha[assignment] += nyha_vector
        fold_sizes[assignment] += size

    def fold_score(fold: int) -> float:
        stratum_score = np.sum(
            (fold_strata[fold] - stratum_targets) ** 2
            / np.maximum(stratum_targets, 1.0)
        )
        size_weight = 1000.0 if BALANCE_RAW_NYHA else 0.5
        size_score = size_weight * (fold_sizes[fold] - size_target) ** 2 / max(
            size_target, 1.0
        )
        nyha4_index = NYHA_ORDER.index(4)
        nyha4_target = nyha_targets[nyha4_index]
        nyha4_score = 0.75 * (
            fold_nyha[fold, nyha4_index] - nyha4_target
        ) ** 2 / max(nyha4_target, 1.0)
        raw_nyha_score = 0.0
        if BALANCE_RAW_NYHA:
            raw_nyha_score = RAW_NYHA_BALANCE_WEIGHT * np.sum(
                (fold_nyha[fold] - nyha_targets) ** 2
                / np.maximum(nyha_targets, 1.0)
            )
        return float(
            stratum_score + size_score + nyha4_score + raw_nyha_score
        )

    def move_group(group_index: int, source_fold: int, target_fold: int) -> None:
        _, stratum_vector, nyha_vector, size = grouped[group_index]
        fold_strata[source_fold] -= stratum_vector
        fold_strata[target_fold] += stratum_vector
        fold_nyha[source_fold] -= nyha_vector
        fold_nyha[target_fold] += nyha_vector
        fold_sizes[source_fold] -= size
        fold_sizes[target_fold] += size
        assignments[group_index] = target_fold

    for pass_number in range(MAX_REFINEMENT_PASSES):
        improved = False
        order = np.random.default_rng(RANDOM_SEED + pass_number).permutation(
            len(grouped)
        )
        for group_index in order:
            source_fold = int(assignments[group_index])
            best_fold = source_fold
            best_delta = 0.0
            source_score = fold_score(source_fold)

            for target_fold in range(N_SPLITS):
                if target_fold == source_fold:
                    continue
                old_score = source_score + fold_score(target_fold)
                move_group(group_index, source_fold, target_fold)
                delta = (
                    fold_score(source_fold) + fold_score(target_fold) - old_score
                )
                move_group(group_index, target_fold, source_fold)

                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_fold = target_fold

            if best_fold != source_fold:
                move_group(group_index, source_fold, best_fold)
                improved = True

        if not improved:
            break

    if BALANCE_RAW_NYHA:
        for pass_number in range(MAX_SWAP_REFINEMENT_PASSES):
            improved = False
            order = np.random.default_rng(
                RANDOM_SEED + 10_000 + pass_number
            ).permutation(len(grouped))
            for left_position, left_index in enumerate(order):
                left_fold = int(assignments[left_index])
                _, left_stratum, left_nyha, left_size = grouped[left_index]
                for right_index in order[left_position + 1 :]:
                    right_fold = int(assignments[right_index])
                    if left_fold == right_fold:
                        continue
                    _, right_stratum, right_nyha, right_size = grouped[right_index]
                    if left_size != right_size:
                        continue

                    old_score = fold_score(left_fold) + fold_score(right_fold)
                    fold_strata[left_fold] += right_stratum - left_stratum
                    fold_strata[right_fold] += left_stratum - right_stratum
                    fold_nyha[left_fold] += right_nyha - left_nyha
                    fold_nyha[right_fold] += left_nyha - right_nyha
                    new_score = fold_score(left_fold) + fold_score(right_fold)

                    if new_score < old_score - 1e-12:
                        assignments[left_index] = right_fold
                        assignments[right_index] = left_fold
                        improved = True
                        break

                    fold_strata[left_fold] += left_stratum - right_stratum
                    fold_strata[right_fold] += right_stratum - left_stratum
                    fold_nyha[left_fold] += left_nyha - right_nyha
                    fold_nyha[right_fold] += right_nyha - left_nyha
                if improved:
                    break
            if not improved:
                break

    group_to_fold = {
        patient_group_id: int(fold)
        for (patient_group_id, _, _, _), fold in zip(grouped, assignments)
    }
    result["fold"] = result["patient_group_id"].map(group_to_fold).astype("int64")
    return result


def check_group_leakage(frame: pd.DataFrame) -> None:
    fold_counts = frame.groupby("patient_group_id")["fold"].nunique()
    leaked = fold_counts[fold_counts > 1]
    if not leaked.empty:
        groups = ", ".join(leaked.index.astype(str).tolist())
        raise RuntimeError(f"同一 patient_group_id 被分到多个 fold：{groups}")


def _join_values(values: Iterable[object]) -> str:
    return "|".join(str(value) for value in values)


def build_multi_image_table(frame: pd.DataFrame) -> pd.DataFrame:
    multi = frame.groupby("patient_group_id", sort=True).filter(
        lambda group: len(group) > 1
    )
    columns = [
        "patient_group_id",
        "num_images",
        "ID_list",
        "NYHA_list",
        "label_3class_list",
        "SEX_list",
        "fold",
    ]
    records: list[dict[str, object]] = []
    for patient_group_id, group in multi.groupby("patient_group_id", sort=True):
        records.append(
            {
                "patient_group_id": patient_group_id,
                "num_images": len(group),
                "ID_list": _join_values(group["ID"]),
                "NYHA_list": _join_values(group["NYHA"]),
                "label_3class_list": _join_values(group["label_3class"]),
                "SEX_list": _join_values(group["SEX"]),
                "fold": int(group["fold"].iloc[0]),
            }
        )
    return pd.DataFrame(records, columns=columns)


def build_split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, int]] = []
    for fold in range(N_SPLITS):
        subset = frame[frame["fold"] == fold]
        record: dict[str, int] = {"fold": fold, "total_samples": len(subset)}
        for nyha in range(5):
            record[f"NYHA_{nyha}"] = int((subset["NYHA"] == nyha).sum())
        for label, name in ((0, "normal"), (1, "mild"), (2, "severe")):
            record[f"class_{name}"] = int((subset["label_3class"] == label).sum())
        for sex in (0, 1):
            record[f"SEX_{sex}"] = int((subset["SEX"] == sex).sum())
        for stratum in STRATA_ORDER:
            record[f"stratum_{stratum}"] = int(
                (subset["stratum"] == stratum).sum()
            )
        records.append(record)
    return pd.DataFrame(records)


def write_fold_files(frame: pd.DataFrame, output_dir: Path) -> None:
    ordered = frame.loc[:, OUTPUT_COLUMNS].sort_values(
        ["fold", "patient_group_id", "ID"], kind="stable"
    )
    ordered.to_csv(
        output_dir / "nyha_3class_sex_stratified_group_5fold.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    for fold in range(N_SPLITS):
        validation = ordered[ordered["fold"] == fold]
        training = ordered[ordered["fold"] != fold]
        training.to_csv(
            output_dir / f"fold_{fold}_train.csv",
            index=False,
            encoding=CSV_ENCODING,
        )
        validation.to_csv(
            output_dir / f"fold_{fold}_val.csv",
            index=False,
            encoding=CSV_ENCODING,
        )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无"
    display = frame.copy()
    display.columns = [str(column) for column in display.columns]
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _distribution_table(
    frame: pd.DataFrame, index: str, columns: str | None = None
) -> pd.DataFrame:
    if columns is None:
        result = frame[index].value_counts().sort_index().rename("count").to_frame()
    else:
        result = pd.crosstab(frame[index], frame[columns])
    return result.reset_index()


def write_quality_report(
    frame: pd.DataFrame,
    total_records: int,
    invalid_count: int,
    missing_count: int,
    multi_table: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    group_stats = frame.groupby("patient_group_id")
    multi_groups = group_stats.filter(lambda group: len(group) > 1)
    different_nyha = (
        multi_groups.groupby("patient_group_id")["NYHA"].nunique().gt(1).sum()
        if not multi_groups.empty
        else 0
    )
    different_class = (
        multi_groups.groupby("patient_group_id")["label_3class"].nunique().gt(1).sum()
        if not multi_groups.empty
        else 0
    )
    inconsistent_sex = (
        multi_groups.groupby("patient_group_id")["SEX"].nunique().gt(1).sum()
        if not multi_groups.empty
        else 0
    )
    leakage = (
        frame.groupby("patient_group_id")["fold"].nunique().gt(1).any()
    )

    fold_class = pd.crosstab(frame["fold"], frame["label_3class_name"]).reset_index()
    fold_sex = pd.crosstab(frame["fold"], frame["sex_name"]).reset_index()
    fold_stratum = pd.crosstab(frame["fold"], frame["stratum"]).reset_index()
    fold_nyha = pd.crosstab(frame["fold"], frame["NYHA"]).reset_index()
    nyha_sex = pd.crosstab(
        [frame["fold"], frame["NYHA"]], frame["sex_name"]
    ).reset_index()
    nyha4_by_fold = (
        frame[frame["NYHA"] == 4]["fold"]
        .value_counts()
        .reindex(range(N_SPLITS), fill_value=0)
        .rename_axis("fold")
        .rename("NYHA_4_count")
        .reset_index()
    )

    report = f"""# NYHA 三分类五折划分质控报告

## 配置与总体结果

- 输入文件：`{LABEL_CSV}`
- 图像目录：`{PROCESSED_IMAGE_ROOT}`
- 输出目录：`{OUTPUT_DIR}`
- 总样本数：{total_records}
- 有效样本数：{len(frame)}
- 缺失图像数：{missing_count}
- 无效记录数：{invalid_count}
- 随机种子：{RANDOM_SEED}
- 划分方法：`StratifiedGroupKFold` 初始划分 + 确定性患者组级分布优化
- 分层字段：`label_3class + "_" + SEX`
- 分组字段：`patient_group_id`
- 优化说明：仅整体移动 patient_group_id；六个联合分层为主目标，折大小与 NYHA 4 分散为软质控项
- 患者跨 fold 泄漏：{"是" if leakage else "否"}

## 全体数据分布

### 原始 NYHA 分布

{_markdown_table(_distribution_table(frame, "NYHA"))}

### 三分类标签分布

{_markdown_table(_distribution_table(frame, "label_3class_name"))}

### 性别分布

SEX=0 为 female，SEX=1 为 male。

{_markdown_table(_distribution_table(frame, "sex_name"))}

### 三分类 × 性别分布

{_markdown_table(_distribution_table(frame, "label_3class_name", "sex_name"))}

## 每折质控

### 每折总样本数及主要计数

{_markdown_table(summary)}

### 每折三分类分布

{_markdown_table(fold_class)}

### 每折性别分布

{_markdown_table(fold_sex)}

### 每折三分类 × 性别分布

{_markdown_table(fold_stratum)}

### 每折原始 NYHA 0–4 分布

{_markdown_table(fold_nyha)}

### 每折 NYHA × 性别分布

{_markdown_table(nyha_sex)}

### NYHA 4 在五折中的分散情况

{_markdown_table(nyha4_by_fold)}

NYHA 4 已作为质控指标观察，不作为硬性分层字段。

## 多图像患者与泄漏检查

- 多图像患者数量：{len(multi_table)}
- 多图像患者中存在不同 NYHA 的患者组数：{int(different_nyha)}
- 多图像患者中存在不同三分类标签的患者组数：{int(different_class)}
- 多图像患者中 SEX 不一致的患者组数：{int(inconsistent_sex)}
- 同一 patient_group_id 是否跨 fold：{"是" if leakage else "否"}

结论：{"未发现患者身份跨折泄漏。" if not leakage else "发现患者身份跨折泄漏，划分无效。"}
"""
    output_path.write_text(report, encoding="utf-8")


def print_core_summary(frame: pd.DataFrame, total_records: int) -> None:
    multi_count = int(
        frame.groupby("patient_group_id").size().gt(1).sum()
    )
    print(f"读取到的总记录数：{total_records}")
    print(f"有效记录数：{len(frame)}")
    print("NYHA 分布：")
    print(frame["NYHA"].value_counts().sort_index().to_string())
    print("三分类分布：")
    print(frame["label_3class_name"].value_counts().sort_index().to_string())
    print("性别分布（SEX=0 female，SEX=1 male）：")
    print(frame["sex_name"].value_counts().sort_index().to_string())
    print(f"多图像患者数量：{multi_count}")
    print(f"输出目录：{OUTPUT_DIR}")
    print("每折三分类分布：")
    print(
        pd.crosstab(frame["fold"], frame["label_3class_name"]).to_string()
    )
    print("patient_group_id 泄漏检查：通过")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_labels(LABEL_CSV)
    total_records = len(raw)
    clean = clean_labels(raw, OUTPUT_DIR)
    clean = build_patient_group_id(clean)
    clean = map_nyha_to_3class(clean)

    group_nyha = clean.groupby("patient_group_id")["NYHA"].nunique()
    group_class = clean.groupby("patient_group_id")["label_3class"].nunique()
    if group_nyha.gt(1).any():
        warnings.warn(
            f"{int(group_nyha.gt(1).sum())} 个多时期患者组存在不同 NYHA。",
            stacklevel=1,
        )
    if group_class.gt(1).any():
        warnings.warn(
            f"{int(group_class.gt(1).sum())} 个多时期患者组存在不同三分类标签。",
            stacklevel=1,
        )

    clean = match_images(clean, PROCESSED_IMAGE_ROOT, OUTPUT_DIR)
    clean = build_stratum(clean)
    split = run_split(clean)
    check_group_leakage(split)

    write_fold_files(split, OUTPUT_DIR)
    multi_table = build_multi_image_table(split)
    multi_table.to_csv(
        OUTPUT_DIR / "multi_image_patient_groups.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    summary = build_split_summary(split)
    summary.to_csv(
        OUTPUT_DIR / "split_summary.csv", index=False, encoding=CSV_ENCODING
    )
    write_quality_report(
        frame=split,
        total_records=total_records,
        invalid_count=0,
        missing_count=0,
        multi_table=multi_table,
        summary=summary,
        output_path=OUTPUT_DIR / "split_quality_report.md",
    )
    print_core_summary(split, total_records)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
