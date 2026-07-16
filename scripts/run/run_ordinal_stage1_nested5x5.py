"""Resumable orchestration for the N0/N1 nested 5x5 experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN = PROJECT_ROOT / "scripts/train/train_nyha_nested5x5.py"
SUMMARIZE = PROJECT_ROOT / "scripts/evaluate/summarize_ordinal_stage1_nested5x5.py"
COMPARE = PROJECT_ROOT / "scripts/evaluate/compare_ordinal_stage1_nested5x5.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["ce", "ordinal", "all"], default="all")
    parser.add_argument("--outer-fold", type=int, default=None)
    parser.add_argument("--inner-fold", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("RUNNING:", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def train_command(method: str, args: argparse.Namespace, *, smoke: bool = False) -> list[str]:
    command = [sys.executable, str(TRAIN), "--method", method]
    if args.outer_fold is not None:
        command.extend(["--outer-fold", str(args.outer_fold)])
    if args.inner_fold is not None:
        command.extend(["--inner-fold", str(args.inner_fold)])
    if args.resume:
        command.append("--resume")
    if args.skip_completed:
        command.append("--skip-completed")
    if smoke:
        command.append("--smoke-test")
        if args.outer_fold is None:
            command.extend(["--outer-fold", "0"])
    return command


def summarize(method: str, *, smoke: bool = False) -> None:
    command = [sys.executable, str(SUMMARIZE), "--method", method]
    if smoke:
        command.append("--smoke-test")
    run(command)


def main() -> int:
    args = parse_args()
    methods = ["ce", "ordinal"] if args.method == "all" else [args.method]
    if args.compare_only:
        run([sys.executable, str(COMPARE), "--repeats", "2000", "--seed", "2026"])
        return 0
    if args.protocol_only:
        run([sys.executable, str(TRAIN), "--method", "ce", "--protocol-only"])
        return 0
    if args.summarize_only:
        for method in methods:
            summarize(method)
        if args.method == "all":
            run([sys.executable, str(COMPARE), "--repeats", "2000", "--seed", "2026"])
        return 0

    run([sys.executable, str(TRAIN), "--method", "ce", "--protocol-only"])
    if not args.skip_tests:
        run(
            [
                sys.executable,
                "-m",
                "unittest",
                "tests.test_ordinal_utils",
                "tests.test_nested5x5_protocol",
                "-v",
            ]
        )
    if args.smoke_test:
        for method in methods:
            run(train_command(method, args, smoke=True))
            summarize(method, smoke=True)
        return 0
    if not args.skip_smoke:
        smoke_args = argparse.Namespace(**vars(args))
        smoke_args.outer_fold = 0
        smoke_args.inner_fold = None
        for method in ["ce", "ordinal"]:
            run(train_command(method, smoke_args, smoke=True))
            summarize(method, smoke=True)
    for method in methods:
        run(train_command(method, args, smoke=False))
        if args.outer_fold is None and args.inner_fold is None:
            summarize(method)
    if args.method == "all" and args.outer_fold is None and args.inner_fold is None:
        run([sys.executable, str(COMPARE), "--repeats", "2000", "--seed", "2026"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
