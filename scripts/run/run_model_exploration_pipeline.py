"""One-command pipeline for model-exploration config, training and summary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = Path("E:/resarch/Anaconda3/envs/face_heart/python.exe")
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "model_exploration_500Data"
PIPELINE_LOG = EXPERIMENT_ROOT / "model_exploration_pipeline_log.txt"
PIPELINE_STATUS = EXPERIMENT_ROOT / "model_exploration_pipeline_status.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-config-generation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--start-from", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def _python() -> str:
    return str(PYTHON_EXE if PYTHON_EXE.is_file() else Path(sys.executable))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _duration_minutes(start_time: str, end_time: str) -> str:
    start = datetime.fromisoformat(start_time)
    end = datetime.fromisoformat(end_time)
    return f"{(end - start).total_seconds() / 60.0:.2f}"


def _append_training_args(command: list[str], args: argparse.Namespace) -> list[str]:
    if args.only:
        command.extend(["--only", args.only])
    if args.start_from:
        command.extend(["--start-from", args.start_from])
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
    return command


def _build_stages(args: argparse.Namespace) -> list[dict[str, object]]:
    stages: list[dict[str, object]] = []
    if not args.skip_config_generation:
        stages.append(
            {
                "stage_name": "generate_configs",
                "command": [
                    _python(),
                    str(PROJECT_ROOT / "scripts" / "run" / "generate_model_exploration_configs.py"),
                ],
            }
        )
    if not args.skip_training:
        training_command = [
            _python(),
            str(
                PROJECT_ROOT
                / "scripts"
                / "run"
                / "run_model_exploration_nyha3class_5fold.py"
            ),
        ]
        stages.append(
            {
                "stage_name": "train_serial_backbones",
                "command": _append_training_args(training_command, args),
            }
        )
    if not args.skip_summary:
        stages.append(
            {
                "stage_name": "summarize_results",
                "command": [
                    _python(),
                    str(
                        PROJECT_ROOT
                        / "scripts"
                        / "evaluate"
                        / "summarize_model_exploration_experiments.py"
                    ),
                ],
            }
        )
    return stages


def _write_status(records: list[dict[str, object]]) -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    PIPELINE_STATUS.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    stages = _build_stages(args)
    records: list[dict[str, object]] = []

    if args.dry_run:
        PIPELINE_LOG.write_text("", encoding="utf-8")
        for stage in stages:
            command = stage["command"]
            command_text = subprocess.list2cmdline(command)  # type: ignore[arg-type]
            print(command_text)
            now = _now()
            records.append(
                {
                    "stage_name": stage["stage_name"],
                    "command": command_text,
                    "start_time": now,
                    "end_time": now,
                    "duration_minutes": "0.00",
                    "status": "DRY_RUN",
                    "return_code": 0,
                    "error_message": "",
                }
            )
        _write_status(records)
        return 0

    PIPELINE_LOG.write_text("", encoding="utf-8")
    overall_exit_code = 0
    for stage in stages:
        command = stage["command"]  # type: ignore[assignment]
        command_text = subprocess.list2cmdline(command)
        start_time = _now()
        record = {
            "stage_name": stage["stage_name"],
            "command": command_text,
            "start_time": start_time,
            "end_time": "",
            "duration_minutes": "",
            "status": "RUNNING",
            "return_code": "",
            "error_message": "",
        }
        records.append(record)
        _write_status(records)

        with PIPELINE_LOG.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[{start_time}] {record['stage_name']}\n")
            log_handle.write(f"{command_text}\n")
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=log_handle,
                text=True,
                check=False,
            )

        end_time = _now()
        record["end_time"] = end_time
        record["duration_minutes"] = _duration_minutes(start_time, end_time)
        record["return_code"] = result.returncode
        if result.returncode == 0:
            record["status"] = "SUCCESS"
        else:
            record["status"] = "FAILED"
            record["error_message"] = f"Stage exited with return_code={result.returncode}"
            overall_exit_code = result.returncode or 1
            _write_status(records)
            if not (
                args.continue_on_error
                and record["stage_name"] == "train_serial_backbones"
            ):
                break
        _write_status(records)

    return int(overall_exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
