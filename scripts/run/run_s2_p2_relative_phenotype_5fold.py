"""End-to-end resumable runner for the fixed S2-P2 parent experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "scripts/data/build_s2_p2_inputs.py"
TRAIN = ROOT / "scripts/train/train_relative_phenotype_5fold.py"
COMPARE = ROOT / "scripts/evaluate/compare_s2_p2_vs_s2_p0.py"
CONFIG = ROOT / "config/train/relative_phenotype/s2_425_p2_relative_global_eye_cheek_resnet18.yaml"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, action="append", dest="folds")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-input-build", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if not args.skip_input_build:
        run([sys.executable, str(BUILD)])
    command = [sys.executable, str(TRAIN), "--config", str(CONFIG)]
    for fold in args.folds or []:
        command.extend(["--fold", str(fold)])
    for flag, enabled in (
        ("--resume", args.resume),
        ("--skip-existing", args.skip_existing),
        ("--smoke-test", args.smoke_test),
        ("--summarize-only", args.summarize_only),
    ):
        if enabled:
            command.append(flag)
    run(command)
    if not args.smoke_test and not args.folds:
        run([sys.executable, str(COMPARE)])


if __name__ == "__main__":
    main()
