"""Non-mutating split/image audit used by E0B before any training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from datasets.control_patient_binary_dataset import map_three_class_to_binary


def audit_fixed_splits(config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    data = config["data"]
    split_dir = Path(data["split_dir"])
    if not split_dir.is_absolute():
        split_dir = Path(__file__).resolve().parents[1] / split_dir
    image_root = Path(data["image_root"])
    if not image_root.is_absolute():
        image_root = Path(__file__).resolve().parents[1] / image_root
    n_folds = int(data["n_folds"])
    validations: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    missing_images: list[dict[str, str]] = []
    for fold in range(n_folds):
        train = pd.read_csv(split_dir / data["train_csv_pattern"].format(fold=fold), dtype={"ID": "string", "patient_group_id": "string"})
        val = pd.read_csv(split_dir / data["val_csv_pattern"].format(fold=fold), dtype={"ID": "string", "patient_group_id": "string"})
        for frame, name in ((train, "train"), (val, "val")):
            required = {"ID", "patient_group_id", "NYHA", "label_3class", "fold"}
            if required.difference(frame.columns):
                raise ValueError(f"fold {fold} {name} lacks {sorted(required.difference(frame.columns))}")
            nyha = pd.to_numeric(frame["NYHA"], errors="coerce")
            three_class = pd.to_numeric(frame["label_3class"], errors="coerce")
            expected_three_class = nyha.map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2})
            if expected_three_class.isna().any() or three_class.isna().any() or not (three_class.astype(int) == expected_three_class.astype(int)).all():
                raise ValueError(
                    f"fold {fold} {name} has invalid NYHA or inconsistent NYHA-to-three-class labels"
                )
            frame["binary_label"] = frame["label_3class"].map(map_three_class_to_binary)
            if frame["binary_label"].isna().any():
                raise ValueError(f"fold {fold} {name} has invalid label mapping")
        overlap = set(train["patient_group_id"].astype(str)).intersection(val["patient_group_id"].astype(str))
        if overlap:
            raise ValueError(f"fold {fold} has patient-group leakage: {sorted(overlap)[:10]}")
        if not set(train["binary_label"]) == {0, 1} or not set(val["binary_label"]) == {0, 1}:
            raise ValueError(f"fold {fold} does not contain both binary classes")
        for row in val.itertuples(index=False):
            image_path = image_root / str(data.get("image_filename_template", "{ID}.png")).format(ID=str(row.ID))
            if not image_path.is_file():
                missing_images.append({"sample_id": str(row.ID), "image_path": str(image_path), "fold": str(fold)})
        rows.append({
            "fold": fold,
            "train_n": len(train), "train_normal_count": int((train.binary_label == 0).sum()), "train_patient_count": int((train.binary_label == 1).sum()),
            "val_n": len(val), "val_normal_count": int((val.binary_label == 0).sum()), "val_patient_count": int((val.binary_label == 1).sum()),
        })
        validations.append(val)
    if missing_images:
        raise FileNotFoundError(f"{len(missing_images)} images are missing: {missing_images[:5]}")
    all_val = pd.concat(validations, ignore_index=True)
    if all_val["ID"].duplicated().any() or len(all_val) != all_val["ID"].nunique():
        raise ValueError("OOF validation IDs are duplicated")
    conflicts = all_val.groupby("patient_group_id", dropna=False).filter(lambda g: g["binary_label"].nunique() > 1)
    conflict_columns = ["patient_group_id", "ID", "NYHA", "label_3class", "binary_label", "fold"]
    conflict_out = conflicts.loc[:, conflict_columns].rename(columns={"ID": "sample_id", "NYHA": "original_label"}).copy()
    conflict_out["image_path"] = conflict_out["sample_id"].map(lambda x: str(image_root / str(data.get("image_filename_template", "{ID}.png")).format(ID=x)))
    conflict_out.to_csv(output_dir / "binary_label_conflicts.csv", index=False, encoding="utf-8-sig")
    if not conflict_out.empty:
        raise ValueError("Binary label conflicts found; formal training stopped. See binary_label_conflicts.csv")
    distribution = pd.DataFrame(rows)
    distribution.to_csv(output_dir / "fold_binary_distribution.csv", index=False, encoding="utf-8-sig")
    counts = all_val["binary_label"].value_counts().to_dict()
    report = f"""# E0B 数据审计报告\n\n- 划分目录：`{split_dir}`\n- 图像目录：`{image_root}`\n- 验证集 OOF 样本：{len(all_val)}\n- 唯一样本 ID：{all_val['ID'].nunique()}\n- 唯一患者组：{all_val['patient_group_id'].nunique()}\n- 对照（0）：{counts.get(0, 0)}\n- 患者（1）：{counts.get(1, 0)}\n- 原始 NYHA 分布：{all_val['NYHA'].value_counts().sort_index().to_dict()}\n- 原始三分类分布：{all_val['label_3class'].value_counts().sort_index().to_dict()}\n- 五折 OOF 无重复、无患者组训练/验证泄漏，且所有指定图像存在。\n- 二分类患者组标签冲突：0 条。\n"""
    (output_dir / "data_audit_report.md").write_text(report, encoding="utf-8")
    return all_val
