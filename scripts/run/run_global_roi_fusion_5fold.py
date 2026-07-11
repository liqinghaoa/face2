"""Serial launcher for Global + selected ROI fusion 5-fold experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_utils import resolve_project_path  # noqa: E402


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "train"
    / "global_roi_fusion"
    / "global_roi_fusion_config_manifest.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "global_roi_fusion_500Data"
JOB_QUEUE_PATH = OUTPUT_ROOT / "global_roi_fusion_job_queue.csv"
LOG_DIR = OUTPUT_ROOT / "logs"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train" / "train_global_roi_fusion_5fold.py"
DEFAULT_PYTHON = Path(r"E:\resarch\Anaconda3\envs\face_heart\python.exe")
SUMMARY_FILES = [
    "fold_metrics_all.csv",
    "mean_metrics.csv",
    "oof_metrics.csv",
    "oof_predictions.csv",
    "summary_report.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--only",
        choices=["global_eye", "global_cheek", "global_eye_cheek"],
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    return parser.parse_args()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _summary_complete(output_dir: Path) -> tuple[bool, str]:
    summary_dir = output_dir / "summary"
    missing = [name for name in SUMMARY_FILES if not (summary_dir / name).is_file()]
    if missing:
        return False, "Missing summary files: " + ";".join(missing)
    try:
        oof = pd.read_csv(summary_dir / "oof_predictions.csv")
        if len(oof) != 500:
            return False, f"OOF predictions row count is {len(oof)}, expected 500"
    except Exception as exc:
        return False, f"Failed to read OOF predictions: {exc}"
    return True, ""


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {
        "job_id",
        "experiment_key",
        "config_path",
        "experiment_name",
        "output_root",
        "backbone",
        "batch_size",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    return frame


def _existing_queue() -> pd.DataFrame:
    if JOB_QUEUE_PATH.is_file():
        return pd.read_csv(JOB_QUEUE_PATH, dtype=str).fillna("")
    return pd.DataFrame()


def _write_queue(rows: list[dict[str, Any]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(JOB_QUEUE_PATH, index=False, encoding="utf-8-sig")


def _cuda_available(python_exe: Path) -> tuple[bool, str]:
    code = (
        "import sklearn\n"
        "import torch\n"
        "print('torch.cuda.is_available()=', torch.cuda.is_available())\n"
        "print('torch.cuda.device_count()=', torch.cuda.device_count())\n"
        "print('torch.cuda.get_device_name(0)=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\n"
        "print('torch.cuda.memory_allocated()=', torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)\n"
        "print('torch.cuda.memory_reserved()=', torch.cuda.memory_reserved() if torch.cuda.is_available() else 0)\n"
    )
    result = subprocess.run(
        [str(python_exe), "-c", code],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    text = (result.stdout or "") + (result.stderr or "")
    return "torch.cuda.is_available()= True" in text, text


def _queue_status_lookup(queue: pd.DataFrame) -> dict[str, str]:
    if queue.empty or "experiment_key" not in queue.columns or "status" not in queue.columns:
        return {}
    return dict(zip(queue["experiment_key"].astype(str), queue["status"].astype(str)))


def _initial_rows(manifest: pd.DataFrame, old_queue: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in manifest.to_dict("records"):
        config_path = resolve_project_path(row["config_path"])
        output_root = resolve_project_path(row["output_root"])
        output_dir = (output_root or OUTPUT_ROOT) / row["experiment_name"]
        rows.append(
            {
                "job_id": row["job_id"],
                "experiment_key": row["experiment_key"],
                "config_path": str(config_path),
                "experiment_name": row["experiment_name"],
                "status": "PENDING",
                "start_time": "",
                "end_time": "",
                "duration_minutes": "",
                "output_dir": str(output_dir),
                "exit_code": "",
                "error_message": "",
            }
        )
    return rows


def _update_row(rows: list[dict[str, Any]], experiment_key: str, **updates: Any) -> None:
    for row in rows:
        if row["experiment_key"] == experiment_key:
            row.update(updates)
            return
    raise KeyError(f"Unknown experiment_key in queue: {experiment_key}")


def main() -> Path:
    args = parse_args()
    python_exe = args.python_exe
    if not python_exe.is_file():
        raise FileNotFoundError(f"Python executable does not exist: {python_exe}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cuda_ok, cuda_text = _cuda_available(python_exe)
    print(cuda_text.strip())
    if not cuda_ok and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Use --allow-cpu only for explicit CPU runs.")

    manifest = _load_manifest(resolve_project_path(args.manifest) or args.manifest)
    if args.only:
        manifest = manifest[manifest["experiment_key"] == args.only].copy()
    if manifest.empty:
        raise ValueError("No jobs selected from manifest")

    old_queue = _existing_queue()
    rows = _initial_rows(manifest, old_queue)
    old_status = _queue_status_lookup(old_queue)

    for job in manifest.to_dict("records"):
        key = job["experiment_key"]
        config_path = resolve_project_path(job["config_path"])
        output_root = resolve_project_path(job["output_root"])
        if config_path is None or not config_path.is_file():
            raise FileNotFoundError(f"Config does not exist for {key}: {config_path}")
        if output_root is None:
            raise ValueError(f"output_root is empty for {key}")
        output_root.mkdir(parents=True, exist_ok=True)
        output_dir = output_root / job["experiment_name"]
        complete, complete_message = _summary_complete(output_dir) if output_dir.exists() else (False, "")

        if args.resume and old_status.get(key) == "SUCCESS":
            _update_row(rows, key, status="SKIPPED", output_dir=str(output_dir), error_message="resume skipped previous SUCCESS")
            _write_queue(rows)
            print(f"Skipping {key}: previous SUCCESS")
            continue
        if args.skip_existing and complete:
            _update_row(rows, key, status="SKIPPED", output_dir=str(output_dir), error_message="existing complete summary found")
            _write_queue(rows)
            print(f"Skipping {key}: complete summary exists at {output_dir}")
            continue
        if output_dir.exists() and not complete and not args.resume and not args.rerun_failed:
            message = (
                f"Output directory exists but summary is incomplete: {output_dir}. "
                "Use --resume or --rerun-failed."
            )
            _update_row(rows, key, status="FAILED", output_dir=str(output_dir), error_message=message)
            _write_queue(rows)
            print(message)
            if not args.continue_on_error:
                raise RuntimeError(message)
            continue

        command = [
            str(python_exe),
            str(TRAIN_SCRIPT),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
        if args.resume:
            command.append("--resume")
        stdout_path = LOG_DIR / f"{key}_stdout.log"
        stderr_path = LOG_DIR / f"{key}_stderr.log"
        print("Running:", subprocess.list2cmdline(command))
        if args.dry_run:
            _update_row(rows, key, status="SKIPPED", output_dir=str(output_dir), error_message="dry-run")
            _write_queue(rows)
            continue

        start = datetime.now()
        _update_row(rows, key, status="RUNNING", start_time=start.isoformat(timespec="seconds"), output_dir=str(output_dir), error_message="")
        _write_queue(rows)
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open("w", encoding="utf-8", errors="replace") as stderr:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
        end = datetime.now()
        duration = (end - start).total_seconds() / 60.0
        complete, complete_message = _summary_complete(output_dir)
        if result.returncode == 0 and complete:
            status = "SUCCESS"
            error = ""
        else:
            status = "FAILED"
            error = complete_message or f"process exit code {result.returncode}"
        _update_row(
            rows,
            key,
            status=status,
            end_time=end.isoformat(timespec="seconds"),
            duration_minutes=f"{duration:.2f}",
            output_dir=str(output_dir),
            exit_code=str(result.returncode),
            error_message=error,
        )
        _write_queue(rows)
        if status != "SUCCESS":
            print(f"{key} failed: {error}")
            if not args.continue_on_error:
                raise RuntimeError(f"{key} failed: {error}")

    print(f"JOB_QUEUE={JOB_QUEUE_PATH}")
    return JOB_QUEUE_PATH


if __name__ == "__main__":
    main()
