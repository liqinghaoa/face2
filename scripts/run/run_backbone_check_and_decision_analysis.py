"""Controller for backbone check plus decision analysis workflow."""

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
LOG_PATH = OUTPUT_ROOT / "backbone_check_and_decision_analysis_pipeline_log.txt"
STATUS_PATH = OUTPUT_ROOT / "backbone_check_and_decision_analysis_pipeline_status.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-config-generation", action="store_true")
    parser.add_argument("--skip-backbone-training", action="store_true")
    parser.add_argument("--skip-backbone-summary", action="store_true")
    parser.add_argument("--skip-decision-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args(argv)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _duration(start: str, end: str) -> float:
    return round((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60.0, 4)


def _command_string(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _write_status(stages: list[dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(stages, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["Backbone check and decision analysis pipeline", "=" * 72, ""]
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


def _skip(stage_name: str, reason: str = "") -> dict:
    timestamp = _now()
    return {
        "stage_name": stage_name,
        "start_time": timestamp,
        "end_time": timestamp,
        "duration_minutes": 0.0,
        "status": "SKIPPED",
        "command": "",
        "return_code": "",
        "error_message": reason,
    }


def _run_stage(stage_name: str, command: list[str], dry_run: bool) -> dict:
    start = _now()
    if dry_run:
        print(f"[dry-run] {stage_name}: {_command_string(command)}")
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
    print(f"[run] {stage_name}: {_command_string(command)}")
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not PYTHON_EXE.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Required Python environment not found: {PYTHON_EXE}")

    stages: list[dict] = []
    commands: list[tuple[str, list[str], bool]] = [
        (
            "config_generation",
            [str(PYTHON_EXE), str(PROJECT_ROOT / "scripts" / "run" / "generate_backbone_check_configs.py")],
            bool(args.skip_config_generation),
        ),
        (
            "backbone_training",
            [
                str(PYTHON_EXE),
                str(PROJECT_ROOT / "scripts" / "run" / "run_backbone_check_preprocess_ablation.py"),
                *(["--dry-run"] if args.dry_run else []),
                *(["--continue-on-error"] if args.continue_on_error else ["--no-continue-on-error"]),
                *(["--allow-cpu"] if args.allow_cpu else []),
            ],
            bool(args.skip_backbone_training),
        ),
        (
            "backbone_summary",
            [
                str(PYTHON_EXE),
                str(PROJECT_ROOT / "scripts" / "evaluate" / "summarize_backbone_check_preprocess_ablation.py"),
            ],
            bool(args.skip_backbone_summary),
        ),
        (
            "decision_analysis",
            [
                str(PYTHON_EXE),
                str(PROJECT_ROOT / "scripts" / "evaluate" / "analyze_preprocess_ablation_confusion_thresholds.py"),
                "--make-figures",
            ],
            bool(args.skip_decision_analysis),
        ),
    ]

    exit_code = 0
    for stage_name, command, skip in commands:
        record = _skip(stage_name, "skipped by flag") if skip else _run_stage(stage_name, command, bool(args.dry_run))
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
