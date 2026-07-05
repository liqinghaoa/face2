"""Summarize all preprocessing ablation experiments into CSV/XLSX/Markdown."""

from __future__ import annotations

import math
import sys
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "preprocess_ablation_500Data"
MANIFEST_PATH = (
    PROJECT_ROOT
    / "config"
    / "train"
    / "preprocess_ablation_resnet18"
    / "config_manifest.csv"
)
SUMMARY_REQUIRED = (
    "summary/fold_metrics_all.csv",
    "summary/mean_metrics.csv",
    "summary/oof_metrics.csv",
    "summary/summary_report.md",
    "summary/oof_predictions.csv",
)
VARIANT_DEFINITIONS = {
    "hybrid_black_baseline": "aligned RGB + final mask, black background",
    "hybrid_imagenet_meanbg": "aligned RGB + final mask, ImageNet mean background",
    "hybrid_black_labl_norm": "Lab-L mild normalization, black background",
    "hybrid_imagenet_meanbg_labl_norm": "Lab-L mild normalization, ImageNet mean background",
    "hybrid_black_clahe_l": "CLAHE on Lab-L, black background",
    "hybrid_black_gray3ch": "grayscale repeated to 3 channels, black background",
    "hybrid_black_masked_grayworld_wb": "masked clipped Gray-World white balance, black background",
    "hybrid_black_retinex_msr": "traditional multi-scale Retinex, black background",
}
FOLD_METRICS = (
    "macro_auc",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "balanced_accuracy",
    "severe_vs_rest_auc",
    "normal_vs_abnormal_auc",
    "recall_normal",
    "recall_mild",
    "recall_severe",
    "f1_normal",
    "f1_mild",
    "f1_severe",
)
OOF_METRICS = (
    "macro_auc",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "balanced_accuracy",
    "severe_vs_rest_auc",
    "normal_vs_abnormal_auc",
)
RANK_METRICS = (
    "macro_auc_mean",
    "balanced_accuracy_mean",
    "macro_f1_mean",
    "recall_severe_mean",
    "f1_severe_mean",
    "oof_macro_f1",
    "oof_balanced_accuracy",
)
DELTA_METRICS = (
    ("macro_auc_mean", "delta_macro_auc"),
    ("balanced_accuracy_mean", "delta_balanced_accuracy"),
    ("macro_f1_mean", "delta_macro_f1"),
    ("recall_severe_mean", "delta_recall_severe"),
    ("f1_severe_mean", "delta_f1_severe"),
)
EXPERIMENT_COLUMNS = (
    "variant_name",
    "experiment_name",
    "config_path",
    "image_root",
    "output_dir",
    "macro_auc_mean",
    "macro_auc_std",
    "accuracy_mean",
    "accuracy_std",
    "macro_precision_mean",
    "macro_precision_std",
    "macro_recall_mean",
    "macro_recall_std",
    "macro_f1_mean",
    "macro_f1_std",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "severe_vs_rest_auc_mean",
    "normal_vs_abnormal_auc_mean",
    "recall_normal_mean",
    "recall_mild_mean",
    "recall_severe_mean",
    "f1_normal_mean",
    "f1_mild_mean",
    "f1_severe_mean",
    "oof_macro_auc",
    "oof_accuracy",
    "oof_macro_precision",
    "oof_macro_recall",
    "oof_macro_f1",
    "oof_balanced_accuracy",
    "oof_severe_vs_rest_auc",
    "oof_normal_vs_abnormal_auc",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def _manifest_config_lookup() -> dict[str, str]:
    if not MANIFEST_PATH.is_file():
        return {}
    frame = pd.read_csv(MANIFEST_PATH, dtype=str, encoding="utf-8-sig").fillna("")
    if not {"variant_name", "config_path"}.issubset(frame.columns):
        return {}
    return dict(zip(frame["variant_name"], frame["config_path"]))


def _infer_variant(experiment_name: str, image_root: str) -> str:
    normalized = image_root.replace("\\", "/")
    for variant in VARIANT_DEFINITIONS:
        if f"/{variant}/" in normalized or experiment_name.endswith(variant):
            return variant
    return ""


def _experiment_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    dirs = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "config.yaml").is_file() and any(
            (path / required).is_file() for required in SUMMARY_REQUIRED
        ):
            dirs.append(path)
    return dirs


def _metric_mean_std(frame: pd.DataFrame, metric: str) -> tuple[float, float]:
    if metric not in frame.columns:
        return float("nan"), float("nan")
    values = pd.to_numeric(frame[metric], errors="coerce")
    return float(values.mean()), float(values.std(ddof=1))


def _read_experiment(
    experiment_dir: Path,
    config_lookup: dict[str, str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    config = _load_yaml(experiment_dir / "config.yaml")
    experiment_name = str(config.get("experiment", {}).get("name", experiment_dir.name))
    image_root = str(config.get("data", {}).get("image_root", ""))
    variant = _infer_variant(experiment_name, image_root)
    config_path = config_lookup.get(variant, "")
    fold_metrics = pd.read_csv(
        experiment_dir / "summary" / "fold_metrics_all.csv",
        encoding="utf-8-sig",
    )
    oof_metrics = pd.read_csv(
        experiment_dir / "summary" / "oof_metrics.csv",
        encoding="utf-8-sig",
    )
    row: dict[str, Any] = {
        "variant_name": variant,
        "experiment_name": experiment_name,
        "config_path": config_path,
        "image_root": image_root,
        "output_dir": str(experiment_dir),
    }
    for metric in FOLD_METRICS:
        mean_value, std_value = _metric_mean_std(fold_metrics, metric)
        row[f"{metric}_mean"] = mean_value
        if f"{metric}_std" in EXPERIMENT_COLUMNS:
            row[f"{metric}_std"] = std_value
    if not oof_metrics.empty:
        oof_row = oof_metrics.iloc[0]
        for metric in OOF_METRICS:
            row[f"oof_{metric}"] = pd.to_numeric(
                pd.Series([oof_row.get(metric)]), errors="coerce"
            ).iloc[0]
    fold_metrics = fold_metrics.assign(
        variant_name=variant,
        experiment_name=experiment_name,
        output_dir=str(experiment_dir),
    )
    oof_metrics = oof_metrics.assign(
        variant_name=variant,
        experiment_name=experiment_name,
        output_dir=str(experiment_dir),
    )
    return row, fold_metrics, oof_metrics


def _ranking(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in RANK_METRICS:
        if metric not in summary.columns:
            continue
        ranked = summary.copy()
        ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
        ranked = ranked.dropna(subset=[metric]).sort_values(metric, ascending=False)
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            rows.append(
                {
                    "ranking_metric": metric,
                    "rank": rank,
                    "variant_name": row.variant_name,
                    "experiment_name": row.experiment_name,
                    "output_dir": row.output_dir,
                    "value": getattr(row, metric),
                }
            )
    return pd.DataFrame(rows)


def _delta_vs_baseline(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    baseline = summary[summary["variant_name"] == "hybrid_black_baseline"].copy()
    if baseline.empty:
        return pd.DataFrame()
    baseline_row = baseline.iloc[0]
    rows = []
    for _, row in summary.iterrows():
        delta = {
            "variant_name": row["variant_name"],
            "experiment_name": row["experiment_name"],
            "output_dir": row["output_dir"],
        }
        for metric, delta_name in DELTA_METRICS:
            delta[delta_name] = pd.to_numeric(
                pd.Series([row.get(metric)]), errors="coerce"
            ).iloc[0] - pd.to_numeric(
                pd.Series([baseline_row.get(metric)]), errors="coerce"
            ).iloc[0]
        rows.append(delta)
    return pd.DataFrame(rows)


def _fmt(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(value) else f"{value:.4f}"


def _best_line(summary: pd.DataFrame, metric: str, label: str) -> str:
    if summary.empty or metric not in summary.columns:
        return f"- {label}: NA"
    values = pd.to_numeric(summary[metric], errors="coerce")
    if values.dropna().empty:
        return f"- {label}: NA"
    row = summary.loc[values.idxmax()]
    return f"- {label}: {row['variant_name']} ({metric}={_fmt(row[metric])})"


def _write_markdown(
    path: Path,
    summary: pd.DataFrame,
    delta: pd.DataFrame,
    ranking: pd.DataFrame,
) -> None:
    lines = [
        "# Preprocessing Ablation Summary",
        "",
        "## Purpose",
        "",
        (
            "This experiment compares lighting, color and background preprocessing "
            "variants while keeping the ResNet18 five-fold NYHA three-class training "
            "pipeline fixed."
        ),
        "",
        "## Variants",
        "",
    ]
    lines.extend([f"- `{name}`: {desc}" for name, desc in VARIANT_DEFINITIONS.items()])
    lines.extend(["", "## Baseline Metrics", ""])
    baseline = summary[summary["variant_name"] == "hybrid_black_baseline"] if not summary.empty else pd.DataFrame()
    if baseline.empty:
        lines.append("No `hybrid_black_baseline` result was found.")
    else:
        row = baseline.iloc[0]
        lines.extend(
            [
                f"- macro_auc_mean: {_fmt(row.get('macro_auc_mean'))}",
                f"- balanced_accuracy_mean: {_fmt(row.get('balanced_accuracy_mean'))}",
                f"- macro_f1_mean: {_fmt(row.get('macro_f1_mean'))}",
                f"- recall_severe_mean: {_fmt(row.get('recall_severe_mean'))}",
                f"- f1_severe_mean: {_fmt(row.get('f1_severe_mean'))}",
            ]
        )
    lines.extend(["", "## Delta vs Baseline", ""])
    if delta.empty:
        lines.append("Delta table is unavailable because the baseline result is missing.")
    else:
        lines.extend(
            [
                "| variant | delta_macro_auc | delta_balanced_accuracy | delta_macro_f1 | delta_recall_severe | delta_f1_severe |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in delta.itertuples(index=False):
            lines.append(
                f"| {row.variant_name} | {_fmt(row.delta_macro_auc)} | "
                f"{_fmt(row.delta_balanced_accuracy)} | {_fmt(row.delta_macro_f1)} | "
                f"{_fmt(row.delta_recall_severe)} | {_fmt(row.delta_f1_severe)} |"
            )
    lines.extend(
        [
            "",
            "## Best Experiments",
            "",
            _best_line(summary, "macro_auc_mean", "Best macro-AUC"),
            _best_line(summary, "balanced_accuracy_mean", "Best balanced accuracy"),
            _best_line(summary, "macro_f1_mean", "Best macro-F1"),
            _best_line(summary, "recall_severe_mean", "Best severe recall"),
            "",
            "## Automatic Interpretation Rules",
            "",
            "- If Lab-L Norm improves BA/F1 without a clear AUC drop, mild brightness normalization may be useful.",
            "- If Gray3ch is close to RGB, the model may not strongly depend on color.",
            "- If Gray3ch drops clearly, color information may contribute and should be checked with EXIF stratification.",
            "- If ImageNet mean background beats black background, black borders or pretraining distribution mismatch may matter.",
            "- If Retinex drops clearly, strong illumination normalization may damage medically relevant cues.",
            "- If no preprocessing helps, BA/F1 may be limited by label boundaries, task difficulty, imbalance, or threshold calibration.",
            "",
            "## Recommended Next Step",
            "",
        ]
    )
    if ranking.empty:
        lines.append("No completed experiment is available for recommendation.")
    else:
        focus = ranking[ranking["ranking_metric"].isin(["balanced_accuracy_mean", "macro_f1_mean"])]
        top_variants = focus.head(4)["variant_name"].drop_duplicates().head(2).tolist()
        lines.append(f"- Recheck top preprocessing variants: {', '.join(top_variants) if top_variants else 'NA'}")
        lines.append("- Expand to ResNet34/ResNet50 only after confirming the best ResNet18 preprocessing candidate.")
        lines.append("- Run EXIF stratification and Grad-CAM for the top candidate and baseline.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _excel_column(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _xlsx_cell(row_index: int, column_index: int, value: Any) -> str:
    ref = f"{_excel_column(column_index)}{row_index}"
    if value is None or pd.isna(value):
        return f'<c r="{ref}"/>'
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _sheet_xml(frame: pd.DataFrame) -> str:
    rows = []
    values = [list(frame.columns)] + frame.astype(object).where(pd.notna(frame), None).values.tolist()
    for row_index, row_values in enumerate(values, start=1):
        cells = [
            _xlsx_cell(row_index, column_index, value)
            for column_index, value in enumerate(row_values)
        ]
        rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(rows)}</sheetData>'
        '</worksheet>'
    )


def _write_minimal_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    sheet_names = list(sheets)
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for index in range(1, len(sheet_names) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook_sheets = []
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for index, name in enumerate(sheet_names, start=1):
        safe_name = escape(name[:31], {'"': "&quot;"})
        workbook_sheets.append(f'<sheet name="{safe_name}" sheetId="{index}" r:id="rId{index}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
    workbook_rels.append(
        f'<Relationship Id="rId{len(sheet_names) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    workbook_rels.append("</Relationships>")
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets>'
        '</workbook>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>'
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "".join(content_types))
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        archive.writestr("xl/styles.xml", styles)
        for index, name in enumerate(sheet_names, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _sheet_xml(sheets[name]))


def _write_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name, index=False)
    except ModuleNotFoundError:
        try:
            with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                for name, frame in sheets.items():
                    frame.to_excel(writer, sheet_name=name, index=False)
        except ModuleNotFoundError:
            _write_minimal_xlsx(path, sheets)


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config_lookup = _manifest_config_lookup()
    rows: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    oof_frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for experiment_dir in _experiment_dirs(OUTPUT_ROOT):
        if not all((experiment_dir / required).is_file() for required in SUMMARY_REQUIRED):
            skipped.append(str(experiment_dir))
            continue
        row, fold_metrics, oof_metrics = _read_experiment(experiment_dir, config_lookup)
        rows.append(row)
        fold_frames.append(fold_metrics)
        oof_frames.append(oof_metrics)

    summary = pd.DataFrame(rows)
    for column in EXPERIMENT_COLUMNS:
        if column not in summary.columns:
            summary[column] = pd.NA
    summary = summary[list(EXPERIMENT_COLUMNS)]
    fold_all = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    oof_all = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    ranking = _ranking(summary)
    delta = _delta_vs_baseline(summary)

    summary.to_csv(OUTPUT_ROOT / "summary_all.csv", index=False, encoding="utf-8-sig")
    _write_xlsx(
        OUTPUT_ROOT / "summary_all.xlsx",
        {
            "experiment_summary": summary,
            "fold_metrics_all": fold_all,
            "oof_metrics_all": oof_all,
            "ranking": ranking,
            "delta_vs_baseline": delta,
        },
    )
    _write_markdown(OUTPUT_ROOT / "summary_all.md", summary, delta, ranking)

    if summary.empty:
        print(f"No completed preprocessing ablation experiments found under {OUTPUT_ROOT}")
    else:
        print(f"Summarized {len(summary)} experiments under {OUTPUT_ROOT}")
    if skipped:
        print(f"Skipped incomplete experiment dirs: {len(skipped)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
