"""One-command GPU entry point for an R3DPR nested-refit binary experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
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
    train = PROJECT_ROOT / "scripts" / "train" / "R3DPR" / "nested-refit" / "train_r3dpr_resnet_binary_nested_refit_5fold.py"
    summarize = PROJECT_ROOT / "scripts" / "evaluate" / "R3DPR" / "nested-refit" / "summarize_r3dpr_resnet_binary_nested_refit_5fold.py"
    plot = PROJECT_ROOT / "scripts" / "evaluate" / "R3DPR" / "nested-refit" / "plot_r3dpr_nested_refit_diagnostics.py"
    subprocess.run([sys.executable, str(train), "--config", str(config_path), "--output-dir", str(output_dir)], cwd=PROJECT_ROOT, check=True)
    subprocess.run([sys.executable, str(summarize), "--experiment-dir", str(output_dir)], cwd=PROJECT_ROOT, check=True)
    subprocess.run([sys.executable, str(plot), "--experiment-dir", str(output_dir)], cwd=PROJECT_ROOT, check=True)
    print(f"NESTED_REFIT_EXPERIMENT_COMPLETED={output_dir}")


if __name__ == "__main__":
    main()
