"""Pre-training integrity audit for the R3DPR direct-label five-fold table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image


def _path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _count(frame: pd.DataFrame, label_column: str, label: int) -> int:
    return int((frame[label_column] == label).sum())


def _markdown_fold_table(rows: list[dict[str, object]]) -> str:
    """Render the small audit table without requiring pandas' optional tabulate dependency."""
    columns = ["fold", "train_n", "train_control", "train_patient", "val_n", "val_control", "val_patient", "val_sex_distribution"]
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def audit_r3dpr_binary_data(config: dict, output_dir: Path, project_root: Path) -> pd.DataFrame:
    """Audit the fixed single-table R3DPR protocol and write reproducible evidence."""
    data = config["data"]
    table_path = _path(data["table_csv"], project_root)
    image_root = _path(data["image_root"], project_root)
    id_column = data["id_column"]
    group_id_column = data["group_id_column"]
    fold_column = data["fold_column"]
    label_column = data["label_column"]
    sex_column = data["sex_column"]
    n_folds = int(data["n_folds"])
    template = str(data["image_filename_template"])
    source_height = int(data["source_image_height"])
    source_width = int(data["source_image_width"])

    table = pd.read_csv(table_path, dtype={id_column: "string", group_id_column: "string"})
    errors: list[str] = []
    required = {id_column, group_id_column, fold_column, label_column, sex_column}
    missing_columns = sorted(required.difference(table.columns))
    if missing_columns:
        raise ValueError(f"R3DPR CSV lacks required columns: {missing_columns}")
    table[id_column] = table[id_column].astype("string").str.strip()
    table[group_id_column] = table[group_id_column].astype("string").str.strip()
    try:
        table[fold_column] = pd.to_numeric(table[fold_column], errors="raise").astype(int)
        table[label_column] = pd.to_numeric(table[label_column], errors="raise").astype(int)
    except (TypeError, ValueError) as error:
        raise ValueError("R3DPR fold and binary label columns must be integer-like") from error

    if table.empty:
        errors.append("CSV contains no samples")
    if table[id_column].isna().any() or (table[id_column] == "").any():
        errors.append("ID contains empty values")
    if table[id_column].duplicated().any():
        errors.append(f"ID is not unique: {int(table[id_column].duplicated().sum())} duplicate rows")
    if table[group_id_column].isna().any() or (table[group_id_column] == "").any():
        errors.append("patient_group_id contains empty values")
    if set(table[fold_column].unique()) != set(range(n_folds)):
        errors.append(f"fold values must be exactly 0..{n_folds - 1}, found {sorted(table[fold_column].unique().tolist())}")
    if not table[label_column].isin([0, 1]).all():
        errors.append("binary label column must contain only 0 and 1")

    image_paths = table[id_column].map(lambda sample_id: image_root / template.format(ID=sample_id))
    missing_images = table.loc[~image_paths.map(Path.is_file), [id_column, group_id_column, fold_column, label_column]].copy()
    if not missing_images.empty:
        errors.append(f"{len(missing_images)} CSV samples have no image file")
    size_mismatches: list[dict[str, object]] = []
    for sample_id, image_path in zip(table[id_column], image_paths):
        if not image_path.is_file():
            continue
        with Image.open(image_path) as image:
            if image.width != source_width or image.height != source_height:
                size_mismatches.append({"ID": sample_id, "path": str(image_path), "width": image.width, "height": image.height})
    if size_mismatches:
        errors.append(f"{len(size_mismatches)} images do not match source size {source_width}x{source_height}")

    fold_rows: list[dict[str, object]] = []
    for fold in range(n_folds):
        train = table.loc[table[fold_column] != fold]
        validation = table.loc[table[fold_column] == fold]
        overlap = set(train[group_id_column]).intersection(validation[group_id_column])
        if overlap:
            errors.append(f"fold {fold} has {len(overlap)} patient_group_id values in both train and validation")
        for split, subset in (("train", train), ("validation", validation)):
            if not {0, 1}.issubset(set(subset[label_column].unique())):
                errors.append(f"fold {fold} {split} set does not contain both classes")
        fold_rows.append({
            "fold": fold,
            "train_n": len(train),
            "train_control": _count(train, label_column, 0),
            "train_patient": _count(train, label_column, 1),
            "val_n": len(validation),
            "val_control": _count(validation, label_column, 0),
            "val_patient": _count(validation, label_column, 1),
            "val_sex_distribution": table.loc[table[fold_column] == fold, sex_column].value_counts(dropna=False).sort_index().to_dict(),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_data_distribution.csv", index=False, encoding="utf-8-sig")
    if not missing_images.empty:
        missing_images.to_csv(output_dir / "missing_images.csv", index=False, encoding="utf-8-sig")
    if size_mismatches:
        pd.DataFrame(size_mismatches).to_csv(output_dir / "image_size_mismatches.csv", index=False, encoding="utf-8-sig")

    total_counts = table[label_column].value_counts().sort_index().to_dict()
    report = [
        "# R3DPR Binary Data Audit",
        "",
        f"- CSV: `{table_path}`",
        f"- Image root: `{image_root}`",
        f"- Samples: {len(table)}",
        f"- Unique IDs: {table[id_column].nunique()}",
        f"- Unique patient groups: {table[group_id_column].nunique()}",
        f"- Control (0): {total_counts.get(0, 0)}",
        f"- Patient (1): {total_counts.get(1, 0)}",
        f"- Source image size (width x height): {source_width} x {source_height}",
        f"- Missing images: {len(missing_images)}",
        f"- Image size mismatches: {len(size_mismatches)}",
        "",
        "## Fold Distribution",
        "",
        _markdown_fold_table(fold_rows),
        "",
        "## Status",
        "",
    ]
    if errors:
        report.extend(["FAILED", "", *[f"- {error}" for error in errors]])
    else:
        report.extend(["PASSED", "", "- All IDs are unique, all images exist, every fold contains both classes, and no patient group crosses a train/validation boundary."])
    (output_dir / "data_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if errors:
        raise ValueError("R3DPR data audit failed. See data_audit_report.md for details: " + "; ".join(errors))
    return table
