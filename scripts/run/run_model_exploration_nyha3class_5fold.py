"""Serial runner for model-exploration NYHA three-class five-fold jobs."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PYTHON_EXE = Path("E:/resarch/Anaconda3/envs/face_heart/python.exe")
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "train"
    / "model_exploration_imagenet_meanbg"
    / "model_exploration_config_manifest.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "model_exploration_500Data"
QUEUE_PATH = OUTPUT_ROOT / "model_exploration_job_queue.csv"
LOG_DIR = OUTPUT_ROOT / "logs"
TRAIN_ENTRYPOINT = PROJECT_ROOT / "scripts" / "run" / "run_nyha3class_5fold_with_config.py"

QUEUE_FIELDNAMES = [
    "job_id",
    "backbone",
    "config_path",
    "image_root",
    "experiment_name",
    "status",
    "start_time",
    "end_time",
    "duration_minutes",
    "output_dir",
    "exit_code",
    "error_message",
    "total_params",
    "trainable_params",
]
REQUIRED_SUMMARY_FILES = [
    "fold_metrics_all.csv",
    "mean_metrics.csv",
    "oof_metrics.csv",
    "oof_predictions.csv",
    "summary_report.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--start-from", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def _resolve(path: str | Path | None) -> Path | None:
    if path in {None, ""}:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_queue(rows: list[dict[str, str]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with QUEUE_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in QUEUE_FIELDNAMES})


def _manifest_to_queue_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in manifest_rows:
        supported = str(row.get("supported", "")).strip().lower() == "true"
        rows.append(
            {
                "job_id": row.get("job_id", ""),
                "backbone": row.get("backbone", ""),
                "config_path": row.get("config_path", ""),
                "image_root": row.get("image_root", ""),
                "experiment_name": row.get("experiment_name", ""),
                "status": "PENDING" if supported else "UNSUPPORTED",
                "start_time": "",
                "end_time": "",
                "duration_minutes": "",
                "output_dir": "",
                "exit_code": "",
                "error_message": "" if supported else row.get("error_message", ""),
                "total_params": "",
                "trainable_params": "",
            }
        )
    return rows


def _merge_existing_queue(rows: list[dict[str, str]], existing_rows: list[dict[str, str]]):
    existing_by_job = {row.get("job_id", ""): row for row in existing_rows}
    merged = []
    for row in rows:
        previous = existing_by_job.get(row.get("job_id", ""))
        if previous and row["status"] != "UNSUPPORTED":
            merged_row = dict(row)
            for field in QUEUE_FIELDNAMES:
                if previous.get(field, ""):
                    merged_row[field] = previous.get(field, "")
            if merged_row.get("status") == "RUNNING":
                merged_row["status"] = "PENDING"
            merged.append(merged_row)
        else:
            merged.append(row)
    return merged


def _parse_backbone_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    parsed = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not parsed:
        raise ValueError("--only was provided but no backbone names were parsed")
    return parsed


def _filter_rows(
    rows: list[dict[str, str]],
    only: set[str] | None,
    start_from: str | None,
) -> list[dict[str, str]]:
    selected = [
        row for row in rows if only is None or row.get("backbone", "").lower() in only
    ]
    if start_from:
        normalized = start_from.strip().lower()
        start_index = None
        for index, row in enumerate(selected):
            if row.get("backbone", "").lower() == normalized:
                start_index = index
                break
        if start_index is None:
            raise ValueError(f"--start-from backbone not found in selected jobs: {start_from}")
        selected = selected[start_index:]
    return selected


def _command_for(row: dict[str, str]) -> list[str]:
    python_exe = PYTHON_EXE if PYTHON_EXE.is_file() else Path(sys.executable)
    config_path = _resolve(row["config_path"])
    if config_path is None:
        raise ValueError(f"Empty config_path for job {row.get('job_id')}")
    return [
        str(python_exe),
        str(TRAIN_ENTRYPOINT),
        "--config",
        str(config_path),
    ]


def _gpu_info_text() -> tuple[str, bool]:
    try:
        import torch

        available = torch.cuda.is_available()
        lines = [f"torch.cuda.is_available(): {available}"]
        lines.append(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        if available:
            lines.append(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
            lines.append(f"torch.cuda.memory_allocated(): {torch.cuda.memory_allocated()}")
            lines.append(f"torch.cuda.memory_reserved(): {torch.cuda.memory_reserved()}")
        else:
            lines.append("CUDA is not available.")
        return "\n".join(lines), bool(available)
    except Exception as exc:  # pragma: no cover - defensive runtime check.
        return f"CUDA info check failed: {exc}\nCUDA is not available.", False


def _summary_complete(experiment_dir: Path) -> tuple[bool, str]:
    summary_dir = experiment_dir / "summary"
    missing = [
        str(summary_dir / filename)
        for filename in REQUIRED_SUMMARY_FILES
        if not (summary_dir / filename).is_file()
    ]
    if missing:
        return False, "Missing required summary files: " + "; ".join(missing)
    return True, ""


def _canonical_output_dir(row: dict[str, str]) -> Path:
    root = _resolve("experiments/model_exploration_500Data")
    if root is None:
        raise ValueError("Output root is empty")
    return root / row["experiment_name"]


def _find_existing_complete_output_dir(row: dict[str, str]) -> Path | None:
    canonical = _canonical_output_dir(row)
    if canonical.is_dir() and _summary_complete(canonical)[0]:
        return canonical
    root = canonical.parent
    if not root.is_dir():
        return None
    candidates = [
        path
        for path in root.glob(f"{row['experiment_name']}*")
        if path.is_dir() and _summary_complete(path)[0]
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _parse_experiment_dir(stdout_log: Path, row: dict[str, str]) -> Path:
    text = stdout_log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"EXPERIMENT_DIR=([^\r\n]+)", text)
    if matches:
        return Path(matches[-1].strip()).expanduser().resolve()
    existing = _find_existing_complete_output_dir(row)
    if existing is not None:
        return existing
    return _canonical_output_dir(row)


def _parse_model_summary(experiment_dir: Path) -> dict[str, str]:
    path = experiment_dir / "model_summary.txt"
    if not path.is_file():
        return {}
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _set_row_status(
    row: dict[str, str],
    status: str,
    *,
    error_message: str = "",
    exit_code: str | int = "",
) -> None:
    row["status"] = status
    row["end_time"] = _now()
    row["exit_code"] = str(exit_code)
    row["error_message"] = error_message
    if row.get("start_time"):
        start = datetime.fromisoformat(row["start_time"])
        end = datetime.fromisoformat(row["end_time"])
        row["duration_minutes"] = f"{(end - start).total_seconds() / 60.0:.2f}"


def _should_run(row: dict[str, str], args: argparse.Namespace) -> bool:
    status = row.get("status", "PENDING")
    if status == "UNSUPPORTED":
        return False
    if args.rerun_failed:
        return status == "FAILED"
    if args.resume and status in {"SUCCESS", "SKIPPED"}:
        return False
    return True


def main() -> int:
    args = parse_args()
    manifest_path = _resolve(args.manifest)
    if manifest_path is None or not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    if not TRAIN_ENTRYPOINT.is_file():
        raise FileNotFoundError(f"Training entrypoint does not exist: {TRAIN_ENTRYPOINT}")

    manifest_rows = _read_csv(manifest_path)
    queue_rows = _manifest_to_queue_rows(manifest_rows)
    if args.resume or args.rerun_failed:
        queue_rows = _merge_existing_queue(queue_rows, _read_csv(QUEUE_PATH))

    only = _parse_backbone_set(args.only)
    selected_rows = _filter_rows(queue_rows, only, args.start_from)
    selected_ids = {row["job_id"] for row in selected_rows}
    queue_rows = [row for row in queue_rows if row["job_id"] in selected_ids]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _write_queue(queue_rows)

    if args.dry_run:
        print(f"DRY-RUN queue: {QUEUE_PATH}")
        for row in queue_rows:
            if row["status"] == "UNSUPPORTED":
                print(f"UNSUPPORTED: {row['backbone']} - {row.get('error_message', '')}")
                continue
            command = _command_for(row)
            print(subprocess.list2cmdline(command))
        return 0

    overall_exit_code = 0
    for row in queue_rows:
        if not _should_run(row, args):
            print(f"Skip {row['backbone']}: status={row.get('status')}")
            continue

        if args.skip_existing:
            existing_output = _find_existing_complete_output_dir(row)
            if existing_output is not None:
                row["output_dir"] = str(existing_output)
                _set_row_status(row, "SKIPPED", error_message="Existing complete summary found")
                _write_queue(queue_rows)
                print(f"Skip existing {row['backbone']}: {existing_output}")
                continue

        stdout_log = LOG_DIR / f"{row['backbone']}_stdout.log"
        stderr_log = LOG_DIR / f"{row['backbone']}_stderr.log"
        row["status"] = "RUNNING"
        row["start_time"] = _now()
        row["end_time"] = ""
        row["duration_minutes"] = ""
        row["exit_code"] = ""
        row["error_message"] = ""
        _write_queue(queue_rows)

        gpu_info, cuda_available = _gpu_info_text()
        command = _command_for(row)
        command_text = subprocess.list2cmdline(command)
        header = (
            f"job_id: {row['job_id']}\n"
            f"backbone: {row['backbone']}\n"
            f"start_time: {row['start_time']}\n"
            f"command: {command_text}\n"
            f"{gpu_info}\n"
        )
        print(header)
        stdout_log.write_text(header + "\n", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")

        if not cuda_available and not args.allow_cpu:
            error = "CUDA is not available and --allow-cpu was not set."
            with stderr_log.open("a", encoding="utf-8") as handle:
                handle.write(error + "\n")
            _set_row_status(row, "FAILED", error_message=error, exit_code=1)
            _write_queue(queue_rows)
            overall_exit_code = 1
            print(error)
            break

        with stdout_log.open("a", encoding="utf-8") as stdout_handle, stderr_log.open(
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

        output_dir = _parse_experiment_dir(stdout_log, row)
        row["output_dir"] = str(output_dir)
        model_summary = _parse_model_summary(output_dir)
        row["total_params"] = model_summary.get("total_params", "")
        row["trainable_params"] = model_summary.get("trainable_params", "")

        if result.returncode != 0:
            error = f"Training command failed with exit_code={result.returncode}"
            _set_row_status(row, "FAILED", error_message=error, exit_code=result.returncode)
            _write_queue(queue_rows)
            overall_exit_code = result.returncode or 1
            print(f"FAILED {row['backbone']}: {error}")
            if not args.continue_on_error:
                break
            continue

        complete, error = _summary_complete(output_dir)
        if not complete:
            _set_row_status(row, "FAILED", error_message=error, exit_code=2)
            _write_queue(queue_rows)
            overall_exit_code = 2
            print(f"FAILED {row['backbone']}: {error}")
            if not args.continue_on_error:
                break
            continue

        _set_row_status(row, "SUCCESS", exit_code=0)
        _write_queue(queue_rows)
        print(f"SUCCESS {row['backbone']}: {output_dir}")

    return int(overall_exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
