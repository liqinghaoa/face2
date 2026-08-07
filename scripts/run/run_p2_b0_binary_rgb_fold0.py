"""Run the fixed P2-B0 RGB fold-0 baseline via the existing E0B trainer."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_utils import load_yaml
from utils.p2_b0_protocol import (
    check_p1_ready_case_alignment,
    project_path,
    validate_p2_b0_config,
    verify_fold0_split_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None, help="Only for an isolated smoke output directory")
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    config_path = project_path(args.config)
    output_dir = project_path(args.output_dir)
    config = load_yaml(config_path)
    validate_p2_b0_config(config)
    if args.epochs is not None and not 1 <= args.epochs < int(config["train"]["epochs"]):
        raise ValueError("--epochs is only allowed for an isolated smoke run with 1 <= epochs < 50")
    verify_fold0_split_protocol(config)
    check_p1_ready_case_alignment(config, output_dir)

    trainer = PROJECT_ROOT / "scripts" / "train" / "train_e0b_global_resnet18_control_patient_binary_5fold.py"
    summarizer = PROJECT_ROOT / "scripts" / "evaluate" / "summarize_p2_b0_binary_rgb_fold0.py"
    train_command = [sys.executable, str(trainer), "--config", str(config_path), "--output-dir", str(output_dir), "--fold", "0"]
    if args.epochs is not None:
        train_command.extend(["--epochs", str(args.epochs)])
    subprocess.run(
        train_command,
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, str(summarizer), "--experiment-dir", str(output_dir)], cwd=PROJECT_ROOT, check=True
    )
    print(f"P2_B0_COMPLETED={output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
