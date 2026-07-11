"""Serial runner for Swin-Tiny second-stage five-fold experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = Path(r"E:/resarch/Anaconda3/envs/face_heart/python.exe")
if not PYTHON_EXE.is_file():
    PYTHON_EXE = Path(sys.executable)

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "train"
    / "swin_tiny_second_stage"
    / "swin_tiny_second_stage_config_manifest.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "swin_tiny_second_stage_500Data"
QUEUE_PATH = OUTPUT_ROOT / "swin_tiny_second_stage_job_queue.csv"
LOG_DIR = OUTPUT_ROOT / "logs"
RUN_WITH_CONFIG = PROJECT_ROOT / "scripts" / "run" / "run_nyha3class_5fold_with_config.py"
SUMMARY_FILES = [
    "summary/fold_metrics_all.csv",
    "summary/mean_metrics.csv",
    "summary/oof_metrics.csv",
    "summary/oof_predictions.csv",
    "summary/summary_report.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--only", choices=["lr5e5", "ls005"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_manifest(path: Path) -> pd.DataFrame:
    path = _resolve(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "job_id",
        "experiment_key",
        "config_path",
        "experiment_name",
        "output_root",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    return frame


def _initial_queue(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in manifest.iterrows():
        rows.append(
            {
                "job_id": row["job_id"],
                "experiment_key": row["experiment_key"],
                "config_path": row["config_path"],
                "experiment_name": row["experiment_name"],
                "status": "PENDING",
                "start_time": "",
                "end_time": "",
                "duration_minutes": "",
                "output_dir": "",
                "exit_code": "",
                "error_message": "",
            }
        )
    return pd.DataFrame(rows)


def _load_queue(manifest: pd.DataFrame) -> pd.DataFrame:
    if not QUEUE_PATH.is_file():
        return _initial_queue(manifest)
    existing = pd.read_csv(QUEUE_PATH, encoding="utf-8-sig")
    base = _initial_queue(manifest).set_index("experiment_key")
    if "experiment_key" not in existing.columns:
        return base.reset_index()
    existing = existing.set_index("experiment_key")
    for key in base.index.intersection(existing.index):
        for column in base.columns:
            if column in existing.columns:
                base.loc[key, column] = existing.loc[key, column]
    return base.reset_index()


def _save_queue(queue: pd.DataFrame) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    queue.to_csv(QUEUE_PATH, index=False, encoding="utf-8-sig")


def _update_queue(queue: pd.DataFrame, experiment_key: str, **updates: Any) -> pd.DataFrame:
    mask = queue["experiment_key"] == experiment_key
    if not mask.any():
        raise KeyError(experiment_key)
    for key, value in updates.items():
        queue.loc[mask, key] = value
    _save_queue(queue)
    return queue


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch.cuda.is_available": bool(torch.cuda.is_available()),
        "torch.cuda.device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        info["torch.cuda.get_device_name(0)"] = torch.cuda.get_device_name(0)
        info["torch.cuda.memory_allocated"] = int(torch.cuda.memory_allocated())
        info["torch.cuda.memory_reserved"] = int(torch.cuda.memory_reserved())
    else:
        info["message"] = "CUDA is not available."
    return info


def _summary_complete(experiment_dir: Path) -> bool:
    return all((experiment_dir / rel).is_file() for rel in SUMMARY_FILES)


def _find_latest_experiment_dir(output_root: Path, experiment_name: str) -> Path | None:
    if not output_root.is_dir():
        return None
    candidates = [
        path
        for path in output_root.iterdir()
        if path.is_dir()
        and (path.name == experiment_name or path.name.startswith(f"{experiment_name}_"))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _parse_experiment_dir_from_stdout(stdout_path: Path) -> Path | None:
    if not stdout_path.is_file():
        return None
    for line in reversed(stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if line.startswith("EXPERIMENT_DIR="):
            return Path(line.split("=", 1)[1].strip())
    return None


def _command(config_path: Path) -> list[str]:
    return [
        str(PYTHON_EXE),
        str(RUN_WITH_CONFIG),
        "--config",
        str(config_path),
    ]


def _row_status(queue: pd.DataFrame, experiment_key: str) -> str:
    row = queue.loc[queue["experiment_key"] == experiment_key]
    if row.empty:
        return "PENDING"
    value = row.iloc[0].get("status", "PENDING")
    if pd.isna(value):
        return "PENDING"
    return str(value)


def _write_log_header(path: Path, header: dict[str, Any], command: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Swin-Tiny second-stage job log\n")
        handle.write(json.dumps(header, ensure_ascii=False, indent=2))
        handle.write("\nCOMMAND=" + subprocess.list2cmdline(command) + "\n\n")


def main() -> int:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    manifest = _read_manifest(args.manifest)
    if args.only:
        manifest = manifest.loc[manifest["experiment_key"] == args.only].copy()
    if manifest.empty:
        raise ValueError("No jobs selected.")

    queue = _load_queue(_read_manifest(args.manifest))
    _save_queue(queue)

    gpu_info = _gpu_info()
    print(json.dumps(gpu_info, ensure_ascii=False, indent=2))
    if not gpu_info["torch.cuda.is_available"] and not args.allow_cpu and not args.dry_run:
        raise RuntimeError("CUDA is not available. Use --allow-cpu only for explicit CPU runs.")

    for _, job in manifest.iterrows():
        experiment_key = str(job["experiment_key"])
        experiment_name = str(job["experiment_name"])
        config_path = _resolve(Path(str(job["config_path"])))
        output_root = _resolve(Path(str(job["output_root"])))
        stdout_path = LOG_DIR / f"{experiment_key}_stdout.log"
        stderr_path = LOG_DIR / f"{experiment_key}_stderr.log"
        command = _command(config_path)

        current_status = _row_status(queue, experiment_key)
        if args.resume and current_status == "SUCCESS":
            print(f"Skipping {experiment_key}: already SUCCESS in queue.")
            continue
        if args.rerun_failed and current_status != "FAILED":
            print(f"Skipping {experiment_key}: --rerun-failed and status={current_status}.")
            continue

        existing_dir = _find_latest_experiment_dir(output_root, experiment_name)
        if args.skip_existing and existing_dir is not None and _summary_complete(existing_dir):
            queue = _update_queue(
                queue,
                experiment_key,
                status="SKIPPED",
                output_dir=str(existing_dir),
                error_message="skip_existing_summary_complete",
            )
            print(f"Skipping {experiment_key}: complete summary exists at {existing_dir}")
            continue

        if args.dry_run:
            print("DRY_RUN:", subprocess.list2cmdline(command))
            queue = _update_queue(
                queue,
                experiment_key,
                status="SKIPPED",
                output_dir=str(existing_dir or ""),
                error_message="dry_run",
            )
            continue

        start = datetime.now()
        queue = _update_queue(
            queue,
            experiment_key,
            status="RUNNING",
            start_time=start.isoformat(timespec="seconds"),
            end_time="",
            duration_minutes="",
            output_dir="",
            exit_code="",
            error_message="",
        )
        header = {
            "experiment_key": experiment_key,
            "experiment_name": experiment_name,
            "start_time": start.isoformat(timespec="seconds"),
            "gpu_info": _gpu_info(),
        }
        _write_log_header(stdout_path, header, command)
        _write_log_header(stderr_path, header, command)
        print(f"Starting {experiment_key}: {subprocess.list2cmdline(command)}")

        with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr_handle:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                check=False,
            )

        end = datetime.now()
        duration = (end - start).total_seconds() / 60.0
        experiment_dir = _parse_experiment_dir_from_stdout(stdout_path)
        if experiment_dir is None:
            experiment_dir = _find_latest_experiment_dir(output_root, experiment_name)
        complete = experiment_dir is not None and _summary_complete(experiment_dir)
        if result.returncode == 0 and complete:
            status = "SUCCESS"
            error_message = ""
        elif result.returncode == 0 and not complete:
            status = "FAILED"
            error_message = "summary_files_missing_after_successful_process"
        else:
            status = "FAILED"
            error_message = f"process_exit_code={result.returncode}"

        queue = _update_queue(
            queue,
            experiment_key,
            status=status,
            end_time=end.isoformat(timespec="seconds"),
            duration_minutes=f"{duration:.2f}",
            output_dir=str(experiment_dir or ""),
            exit_code=result.returncode,
            error_message=error_message,
        )
        print(
            f"Finished {experiment_key}: status={status}, exit_code={result.returncode}, "
            f"duration_minutes={duration:.2f}, output_dir={experiment_dir}"
        )
        if status != "SUCCESS" and not args.continue_on_error:
            return int(result.returncode or 1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
