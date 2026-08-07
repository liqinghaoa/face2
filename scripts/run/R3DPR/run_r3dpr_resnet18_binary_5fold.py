"""One-command GPU entry point for the R3DPR binary baseline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "train" / "R3DPR" / "r3dpr_resnet18_binary_256x320_5fold.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    output_dir = args.output_dir or Path(config["experiment"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    train = PROJECT_ROOT / "scripts" / "train" / "R3DPR" / "train_r3dpr_resnet18_binary_5fold.py"
    summarize = PROJECT_ROOT / "scripts" / "evaluate" / "R3DPR" / "summarize_r3dpr_resnet18_binary_5fold.py"
    plot = PROJECT_ROOT / "scripts" / "evaluate" / "R3DPR" / "plot_r3dpr_training_diagnostics.py"
    subprocess.run([sys.executable, str(train), "--config", str(config_path), "--output-dir", str(output_dir)], cwd=PROJECT_ROOT, check=True)
    subprocess.run([sys.executable, str(summarize), "--experiment-dir", str(output_dir)], cwd=PROJECT_ROOT, check=True)
    subprocess.run([sys.executable, str(plot), "--experiment-dir", str(output_dir)], cwd=PROJECT_ROOT, check=True)
    print(f"EXPERIMENT_COMPLETED={output_dir}")


if __name__ == "__main__":
    main()
