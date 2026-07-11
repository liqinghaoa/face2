"""Pipeline controller for Swin-Tiny second-stage experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = Path(r"E:/resarch/Anaconda3/envs/face_heart/python.exe")
if not PYTHON_EXE.is_file():
    PYTHON_EXE = Path(sys.executable)

OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "swin_tiny_second_stage_500Data"
PIPELINE_LOG = OUTPUT_ROOT / "swin_tiny_second_stage_pipeline_log.txt"
PIPELINE_STATUS = OUTPUT_ROOT / "swin_tiny_second_stage_pipeline_status.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-threshold-scan", action="store_true")
    parser.add_argument("--skip-config-generation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", choices=["lr5e5", "ls005"], default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--make-figures", action="store_true")
    return parser.parse_args()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _append_log(text: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with PIPELINE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def _run_stage(stage_name: str, command: list[str], *, dry_run: bool) -> dict[str, Any]:
    start = datetime.now()
    record: dict[str, Any] = {
        "stage_name": stage_name,
        "command": subprocess.list2cmdline(command),
        "start_time": start.isoformat(timespec="seconds"),
        "end_time": "",
        "duration_minutes": "",
        "status": "RUNNING",
        "return_code": "",
        "error_message": "",
    }
    _append_log(f"[{record['start_time']}] START {stage_name}")
    _append_log(record["command"])
    if dry_run:
        end = datetime.now()
        record.update(
            {
                "end_time": end.isoformat(timespec="seconds"),
                "duration_minutes": f"{(end - start).total_seconds() / 60.0:.2f}",
                "status": "SKIPPED",
                "return_code": 0,
                "error_message": "pipeline_dry_run",
            }
        )
        _append_log(f"[{record['end_time']}] DRY_RUN {stage_name}")
        return record

    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, check=False)
    end = datetime.now()
    record.update(
        {
            "end_time": end.isoformat(timespec="seconds"),
            "duration_minutes": f"{(end - start).total_seconds() / 60.0:.2f}",
            "status": "SUCCESS" if result.returncode == 0 else "FAILED",
            "return_code": result.returncode,
            "error_message": "" if result.returncode == 0 else f"return_code={result.returncode}",
        }
    )
    _append_log(f"[{record['end_time']}] END {stage_name} status={record['status']} return_code={result.returncode}")
    return record


def main() -> int:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PIPELINE_LOG.write_text(f"Swin-Tiny second-stage pipeline started at {_now()}\n", encoding="utf-8")

    stages: list[tuple[str, list[str]]] = []
    if not args.skip_threshold_scan:
        command = [
            str(PYTHON_EXE),
            str(PROJECT_ROOT / "scripts" / "evaluate" / "analyze_swin_tiny_threshold_scan.py"),
        ]
        if args.make_figures:
            command.append("--make-figures")
        stages.append(("threshold_scan", command))
    if not args.skip_config_generation:
        stages.append(
            (
                "config_generation",
                [
                    str(PYTHON_EXE),
                    str(PROJECT_ROOT / "scripts" / "run" / "generate_swin_tiny_second_stage_configs.py"),
                ],
            )
        )
    if not args.skip_training:
        command = [
            str(PYTHON_EXE),
            str(PROJECT_ROOT / "scripts" / "run" / "run_swin_tiny_second_stage_5fold.py"),
        ]
        if args.only:
            command.extend(["--only", args.only])
        for flag_name in [
            "resume",
            "rerun_failed",
            "skip_existing",
            "continue_on_error",
            "allow_cpu",
        ]:
            if getattr(args, flag_name):
                command.append("--" + flag_name.replace("_", "-"))
        stages.append(("training", command))
    if not args.skip_summary:
        stages.append(
            (
                "summary",
                [
                    str(PYTHON_EXE),
                    str(PROJECT_ROOT / "scripts" / "evaluate" / "summarize_swin_tiny_second_stage.py"),
                ],
            )
        )

    records = []
    exit_code = 0
    for stage_name, command in stages:
        record = _run_stage(stage_name, command, dry_run=args.dry_run)
        records.append(record)
        PIPELINE_STATUS.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if record["status"] == "FAILED":
            exit_code = int(record["return_code"] or 1)
            if not args.continue_on_error:
                break

    print(f"PIPELINE_LOG={PIPELINE_LOG}")
    print(f"PIPELINE_STATUS={PIPELINE_STATUS}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
