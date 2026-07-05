"""One-command controller for the full preprocessing ablation workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = Path("E:/resarch/Anaconda3/envs/face_heart/python.exe")
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "preprocess_ablation_500Data"
LOG_PATH = OUTPUT_ROOT / "full_pipeline_log.txt"
STATUS_PATH = OUTPUT_ROOT / "full_pipeline_status.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-intermediates", action="store_true")
    parser.add_argument("--skip-ablation-images", action="store_true")
    parser.add_argument("--skip-config-generation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _duration(start: str, end: str) -> float:
    return round((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60.0, 4)


def _command_string(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _write_status(stages: list[dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(
        json.dumps(stages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = ["Full preprocessing ablation pipeline log", "=" * 72, ""]
    for stage in stages:
        lines.extend(
            [
                f"stage_name: {stage['stage_name']}",
                f"status: {stage['status']}",
                f"start_time: {stage['start_time']}",
                f"end_time: {stage['end_time']}",
                f"duration_minutes: {stage['duration_minutes']}",
                f"return_code: {stage['return_code']}",
                f"command: {stage['command']}",
                f"error_message: {stage['error_message']}",
                "",
            ]
        )
    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")


def _stage(stage_name: str, command: list[str], dry_run: bool) -> dict:
    start = _now()
    if dry_run:
        print("DRY-RUN:", _command_string(command))
        end = _now()
        return {
            "stage_name": stage_name,
            "start_time": start,
            "end_time": end,
            "duration_minutes": _duration(start, end),
            "status": "DRY_RUN",
            "command": _command_string(command),
            "return_code": "",
            "error_message": "",
        }
    print("Running:", _command_string(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False, text=True)
    end = _now()
    return {
        "stage_name": stage_name,
        "start_time": start,
        "end_time": end,
        "duration_minutes": _duration(start, end),
        "status": "SUCCESS" if result.returncode == 0 else "FAILED",
        "command": _command_string(command),
        "return_code": int(result.returncode),
        "error_message": "" if result.returncode == 0 else f"exit_code={result.returncode}",
    }


def _skip(stage_name: str) -> dict:
    timestamp = _now()
    return {
        "stage_name": stage_name,
        "start_time": timestamp,
        "end_time": timestamp,
        "duration_minutes": 0.0,
        "status": "SKIPPED",
        "command": "",
        "return_code": "",
        "error_message": "",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not PYTHON_EXE.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Required Python environment not found: {PYTHON_EXE}")

    stages: list[dict] = []
    commands: list[tuple[str, list[str], bool]] = [
        (
            "intermediates",
            [str(PYTHON_EXE), str(PROJECT_ROOT / "scripts" / "run" / "run_preprocess_global_face_hybrid_intermediates.py")],
            bool(args.skip_intermediates),
        ),
        (
            "ablation_images",
            [
                str(PYTHON_EXE),
                str(PROJECT_ROOT / "preprocessing" / "build_global_face_preprocess_ablation_from_intermediates.py"),
                "--variants",
                args.only or "all",
            ],
            bool(args.skip_ablation_images),
        ),
        (
            "config_generation",
            [str(PYTHON_EXE), str(PROJECT_ROOT / "scripts" / "run" / "generate_preprocess_ablation_train_configs.py")],
            bool(args.skip_config_generation),
        ),
        (
            "training",
            [
                str(PYTHON_EXE),
                str(PROJECT_ROOT / "scripts" / "run" / "run_preprocess_ablation_resnet18_nyha3class_5fold.py"),
                *(["--only", args.only] if args.only else []),
                *(["--dry-run"] if args.dry_run else []),
                *(["--continue-on-error"] if args.continue_on_error else ["--no-continue-on-error"]),
            ],
            bool(args.skip_training),
        ),
        (
            "summary",
            [str(PYTHON_EXE), str(PROJECT_ROOT / "scripts" / "evaluate" / "summarize_preprocess_ablation_experiments.py")],
            bool(args.skip_summary),
        ),
    ]

    exit_code = 0
    for stage_name, command, skip in commands:
        record = _skip(stage_name) if skip else _stage(stage_name, command, bool(args.dry_run))
        stages.append(record)
        _write_status(stages)
        if record["status"] == "FAILED":
            exit_code = int(record["return_code"]) or 1
            if not args.continue_on_error:
                break
    print(f"Pipeline status: {STATUS_PATH}")
    print(f"Pipeline log: {LOG_PATH}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
