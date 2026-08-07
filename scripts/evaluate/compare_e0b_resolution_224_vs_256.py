"""Compare E0B 224x224 and 256x256 mean-background binary experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_utils import load_yaml  # noqa: E402


DEFAULT_EXP_224 = (
    PROJECT_ROOT
    / "experiments"
    / "500Data"
    / "E0B_Global_ResNet18_ControlVsPatient_Binary_5fold"
)
DEFAULT_EXP_256 = (
    PROJECT_ROOT
    / "experiments"
    / "500Data"
    / "E0B_Global_ResNet18_ControlVsPatient_Binary_256MeanBG_5fold"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "500Data"
    / "E0B_Resolution_224_vs_256_MeanBG_Comparison"
)


MAIN_METRICS = (
    "macro_auc",
    "macro_f1",
    "balanced_accuracy",
    "patient_sensitivity",
    "control_specificity",
)


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-224", type=Path, default=DEFAULT_EXP_224)
    parser.add_argument("--exp-256", type=Path, default=DEFAULT_EXP_256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _read_confusion_metrics(exp_dir: Path) -> dict[str, float]:
    matrix_path = exp_dir / "oof_confusion_matrix.csv"
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Missing OOF confusion matrix: {matrix_path}")
    matrix = pd.read_csv(matrix_path, index_col=0)
    tn = float(matrix.loc["control", "control"])
    fp = float(matrix.loc["control", "patient"])
    fn = float(matrix.loc["patient", "control"])
    tp = float(matrix.loc["patient", "patient"])
    return {
        "patient_sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "control_specificity": tn / (tn + fp) if tn + fp else 0.0,
    }


def _experiment_row(exp_dir: Path, label: str) -> dict[str, Any]:
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")
    metrics_path = exp_dir / "oof_metrics.csv"
    config_path = exp_dir / "config_snapshot.yaml"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing OOF metrics: {metrics_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config snapshot: {config_path}")

    metrics = pd.read_csv(metrics_path, encoding="utf-8-sig").iloc[0].to_dict()
    config = load_yaml(config_path)
    row = {
        "experiment": label,
        "experiment_dir": str(exp_dir),
        "image_size": int(config["data"]["image_size"]),
        "image_root": config["data"]["image_root"],
    }
    row.update({key: float(metrics[key]) for key in ("macro_auc", "macro_f1", "balanced_accuracy")})
    row.update(_read_confusion_metrics(exp_dir))
    return row


def _write_report(comparison: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for row in comparison.itertuples(index=False):
        rows.append(
            "| {experiment} | {image_size} | {macro_auc:.6f} | {macro_f1:.6f} | "
            "{balanced_accuracy:.6f} | {patient_sensitivity:.6f} | "
            "{control_specificity:.6f} | {delta_macro_auc:+.6f} | "
            "{delta_macro_f1:+.6f} | {delta_balanced_accuracy:+.6f} |".format(
                **row._asdict()
            )
        )
    payload = "\n".join(
        [
            "# E0B mean-background 输入分辨率消融：224 vs 256",
            "",
            "固定项：E0B 二分类任务、splits_500、ResNet18 ImageNet 预训练、AdamW、学习率、权重衰减、",
            "epoch/patience、seed、ImageNet normalization 与水平翻转策略。",
            "",
            "| Experiment | Image size | Macro-AUC | Macro-F1 | Balanced Accuracy | Sensitivity | Specificity | Δ Macro-AUC vs 224 | Δ Macro-F1 vs 224 | Δ BA vs 224 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "解释边界：该表只说明在当前 500 例内部五折 OOF 协议下，输入从 224×224 改到 256×256 后的性能变化；不等价于外部泛化证明。",
        ]
    )
    (output_dir / "resolution_comparison_report.md").write_text(payload, encoding="utf-8")


def main() -> Path:
    args = parse_args()
    exp_224 = _resolve(args.exp_224)
    exp_256 = _resolve(args.exp_256)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _experiment_row(exp_224, "E0B_224_meanbg"),
        _experiment_row(exp_256, "E0B_256_meanbg"),
    ]
    comparison = pd.DataFrame(rows)
    baseline = comparison.loc[comparison["experiment"] == "E0B_224_meanbg"].iloc[0]
    for metric in MAIN_METRICS:
        comparison[f"delta_{metric}"] = comparison[metric].astype(float) - float(baseline[metric])

    comparison.to_csv(output_dir / "resolution_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    (output_dir / "resolution_comparison_metrics.json").write_text(
        json.dumps(comparison.to_dict("records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(comparison, output_dir)
    print(f"RESOLUTION_COMPARISON_DIR={output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
