"""Run E0B training and its mandatory OOF summary as one GPU command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    train = PROJECT_ROOT / "scripts" / "train" / "train_e0b_global_resnet18_control_patient_binary_5fold.py"
    summarize = PROJECT_ROOT / "scripts" / "evaluate" / "summarize_e0b_global_resnet18_control_patient_binary_5fold.py"
    subprocess.run([sys.executable, str(train), "--config", str(config), "--output-dir", str(output_dir)], cwd=PROJECT_ROOT, check=True)
    subprocess.run([sys.executable, str(summarize), "--experiment-dir", str(output_dir)], cwd=PROJECT_ROOT, check=True)
    print(f"EXPERIMENT_COMPLETED={output_dir}")


if __name__ == "__main__":
    main()
