"""End-to-end runner for the P2-1 experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/train/relative_phenotype/nyha_3class_relative_global_eye_cheek_resnet18.yaml"
TRAIN = ROOT / "scripts/train/train_relative_phenotype_5fold.py"
COMPARE = ROOT / "scripts/evaluate/compare_relative_phenotype_p2_1.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, action="append", dest="folds")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
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
    subprocess.run(command, cwd=ROOT, check=True)
    if not args.smoke_test and not args.folds:
        subprocess.run([sys.executable, str(COMPARE)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
