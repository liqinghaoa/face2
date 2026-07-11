"""End-to-end pipeline for Global + selected ROI fusion experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "global_roi_fusion_500Data"
PIPELINE_LOG = OUTPUT_ROOT / "global_roi_fusion_pipeline_log.txt"
PIPELINE_STATUS = OUTPUT_ROOT / "global_roi_fusion_pipeline_status.json"
DEFAULT_PYTHON = Path(r"E:\resarch\Anaconda3\envs\face_heart\python.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-config-generation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--skip-diagnostic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        choices=["global_eye", "global_cheek", "global_eye_cheek"],
        default=None,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    return parser.parse_args()


def _now() -> datetime:
    return datetime.now()


def _append_log(text: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with PIPELINE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _run_stage(stage_name: str, command: list[str], dry_run: bool) -> dict[str, Any]:
    start = _now()
    record: dict[str, Any] = {
        "stage_name": stage_name,
        "command": _command_text(command),
        "start_time": start.isoformat(timespec="seconds"),
        "end_time": "",
        "duration_minutes": "",
        "status": "RUNNING",
        "return_code": "",
        "error_message": "",
    }
    _append_log(f"[{record['start_time']}] START {stage_name}: {record['command']}")
    if dry_run:
        end = _now()
        record.update(
            {
                "end_time": end.isoformat(timespec="seconds"),
                "duration_minutes": f"{(end - start).total_seconds() / 60.0:.2f}",
                "status": "SKIPPED",
                "return_code": "0",
                "error_message": "dry-run",
            }
        )
        _append_log(f"[{record['end_time']}] DRY-RUN {stage_name}")
        return record

    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, check=False)
    end = _now()
    record.update(
        {
            "end_time": end.isoformat(timespec="seconds"),
            "duration_minutes": f"{(end - start).total_seconds() / 60.0:.2f}",
            "status": "SUCCESS" if result.returncode == 0 else "FAILED",
            "return_code": str(result.returncode),
            "error_message": "" if result.returncode == 0 else f"return code {result.returncode}",
        }
    )
    _append_log(f"[{record['end_time']}] END {stage_name}: {record['status']} rc={result.returncode}")
    return record


def _write_status(records: list[dict[str, Any]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with PIPELINE_STATUS.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


def main() -> Path:
    args = parse_args()
    if not args.python_exe.is_file():
        raise FileNotFoundError(f"Python executable does not exist: {args.python_exe}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    stages: list[tuple[str, list[str], bool]] = []
    if not args.skip_config_generation:
        stages.append(
            (
                "generate_configs",
                [
                    str(args.python_exe),
                    str(PROJECT_ROOT / "scripts" / "run" / "generate_global_roi_fusion_configs.py"),
                ],
                args.dry_run,
            )
        )
    if not args.skip_training:
        command = [
            str(args.python_exe),
            str(PROJECT_ROOT / "scripts" / "run" / "run_global_roi_fusion_5fold.py"),
        ]
        if args.only:
            command.extend(["--only", args.only])
        if args.resume:
            command.append("--resume")
        if args.rerun_failed:
            command.append("--rerun-failed")
        if args.skip_existing:
            command.append("--skip-existing")
        if args.continue_on_error:
            command.append("--continue-on-error")
        if args.allow_cpu:
            command.append("--allow-cpu")
        if args.dry_run:
            command.append("--dry-run")
        stages.append(("training", command, False))
    if not args.skip_summary:
        stages.append(
            (
                "summary",
                [
                    str(args.python_exe),
                    str(PROJECT_ROOT / "scripts" / "evaluate" / "summarize_global_roi_fusion_results.py"),
                ],
                args.dry_run,
            )
        )
    if not args.skip_diagnostic:
        stages.append(
            (
                "diagnostic",
                [
                    str(args.python_exe),
                    str(PROJECT_ROOT / "scripts" / "evaluate" / "diagnose_global_roi_fusion_results.py"),
                ],
                args.dry_run,
            )
        )

    for stage_name, command, dry_run_stage in stages:
        record = _run_stage(stage_name, command, dry_run_stage)
        records.append(record)
        _write_status(records)
        if record["status"] == "FAILED" and not args.continue_on_error:
            raise RuntimeError(f"Pipeline stage failed: {stage_name}")

    print(f"PIPELINE_LOG={PIPELINE_LOG}")
    print(f"PIPELINE_STATUS={PIPELINE_STATUS}")
    return PIPELINE_STATUS


if __name__ == "__main__":
    main()
