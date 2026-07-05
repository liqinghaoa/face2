"""Serially run ResNet34/50 backbone checks for preprocessing variants."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = Path("E:/resarch/Anaconda3/envs/face_heart/python.exe")
RUNNER = PROJECT_ROOT / "scripts" / "run" / "run_nyha3class_5fold_with_config.py"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "train"
    / "preprocess_ablation_backbone_check"
    / "backbone_check_config_manifest.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "preprocess_ablation_500Data" / "backbone_check"
LOG_DIR = OUTPUT_ROOT / "logs"
QUEUE_PATH = OUTPUT_ROOT / "backbone_check_job_queue.csv"
JOB_ORDER = ("B1", "B2", "B3", "B4")
REQUIRED_SUMMARY_FILES = (
    "summary/fold_metrics_all.csv",
    "summary/mean_metrics.csv",
    "summary/oof_metrics.csv",
    "summary/oof_predictions.csv",
    "summary/summary_report.md",
)
QUEUE_COLUMNS = (
    "job_id",
    "backbone",
    "variant_name",
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
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--start-from", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow training when CUDA is unavailable.",
    )
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _duration_minutes(start: str, end: str) -> float:
    return round((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60.0, 4)


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest does not exist: {path}. Run generate_backbone_check_configs.py first."
        )
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    required = {
        "job_id",
        "backbone",
        "variant_name",
        "config_path",
        "image_root",
        "experiment_name",
        "output_root",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    order = {job_id: index for index, job_id in enumerate(JOB_ORDER)}
    frame["_order"] = frame["job_id"].map(order)
    if frame["_order"].isna().any():
        unknown = frame.loc[frame["_order"].isna(), "job_id"].tolist()
        raise ValueError(f"Manifest contains unknown job_id values: {unknown}")
    return frame.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def _matches_selector(row: pd.Series, selectors: set[str]) -> bool:
    values = {
        str(row.get("job_id", "")),
        str(row.get("backbone", "")),
        str(row.get("variant_name", "")),
        str(row.get("experiment_name", "")),
    }
    return bool(values.intersection(selectors))


def _filter_manifest(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    result = frame.copy()
    if args.start_from:
        selectors = {item.strip() for item in args.start_from.split(",") if item.strip()}
        matches = result[result.apply(lambda row: _matches_selector(row, selectors), axis=1)]
        if matches.empty:
            raise ValueError(f"--start-from did not match any job: {args.start_from}")
        start_order = matches.index.min()
        result = result.loc[start_order:].copy()
    if args.only:
        selectors = {item.strip() for item in args.only.split(",") if item.strip()}
        result = result[result.apply(lambda row: _matches_selector(row, selectors), axis=1)].copy()
        if result.empty:
            raise ValueError(f"--only did not match any job: {args.only}")
    return result.reset_index(drop=True)


def _initial_queue(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in manifest.itertuples(index=False):
        rows.append(
            {
                "job_id": row.job_id,
                "backbone": row.backbone,
                "variant_name": row.variant_name,
                "config_path": row.config_path,
                "image_root": row.image_root,
                "experiment_name": row.experiment_name,
                "status": "PENDING",
                "start_time": "",
                "end_time": "",
                "duration_minutes": "",
                "output_dir": "",
                "exit_code": "",
                "error_message": "",
            }
        )
    return pd.DataFrame(rows, columns=QUEUE_COLUMNS)


def _save_queue(queue: pd.DataFrame) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    queue.to_csv(QUEUE_PATH, index=False, encoding="utf-8-sig")


def _load_or_create_queue(manifest: pd.DataFrame, resume: bool, rerun_failed: bool) -> pd.DataFrame:
    if QUEUE_PATH.is_file() and (resume or rerun_failed):
        existing = pd.read_csv(QUEUE_PATH, dtype=str, encoding="utf-8-sig").fillna("")
        for column in QUEUE_COLUMNS:
            if column not in existing.columns:
                existing[column] = ""
        wanted = set(manifest["job_id"].astype(str))
        existing = existing[existing["job_id"].isin(wanted)].copy()
        present = set(existing["job_id"].astype(str))
        missing = manifest[~manifest["job_id"].isin(present)]
        if not missing.empty:
            existing = pd.concat([existing[list(QUEUE_COLUMNS)], _initial_queue(missing)], ignore_index=True)
        order = {job_id: index for index, job_id in enumerate(JOB_ORDER)}
        existing["_order"] = existing["job_id"].map(order)
        return existing.sort_values("_order").drop(columns=["_order"])[list(QUEUE_COLUMNS)].reset_index(drop=True)
    return _initial_queue(manifest)


def _gpu_info() -> dict[str, Any]:
    code = (
        "import json, torch\n"
        "info={'cuda_available': torch.cuda.is_available(), "
        "'device_count': torch.cuda.device_count(), 'current_device': 'cpu', "
        "'training_device': 'cuda' if torch.cuda.is_available() else 'cpu', "
        "'device_name': '', 'memory_allocated': 0, 'memory_reserved': 0}\n"
        "if torch.cuda.is_available():\n"
        "    idx=torch.cuda.current_device()\n"
        "    info.update({'current_device': f'cuda:{idx}', "
        "'device_name': torch.cuda.get_device_name(idx), "
        "'memory_allocated': int(torch.cuda.memory_allocated(idx)), "
        "'memory_reserved': int(torch.cuda.memory_reserved(idx))})\n"
        "print(json.dumps(info, ensure_ascii=False))\n"
    )
    result = subprocess.run(
        [str(PYTHON_EXE), "-c", code],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {
            "cuda_available": False,
            "device_count": 0,
            "current_device": "unknown",
            "training_device": "unknown",
            "device_name": "",
            "memory_allocated": 0,
            "memory_reserved": 0,
            "error": result.stderr.strip(),
        }
    return json.loads(result.stdout.strip())


def _append_log_header(path: Path, job: pd.Series, gpu_info: dict[str, Any], command: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 88,
        f"JOB_ID={job['job_id']}",
        f"BACKBONE={job['backbone']}",
        f"VARIANT={job['variant_name']}",
        f"EXPERIMENT_NAME={job['experiment_name']}",
        f"CONFIG={_resolve(str(job['config_path']))}",
        f"IMAGE_ROOT={job['image_root']}",
        f"GPU_INFO={json.dumps(gpu_info, ensure_ascii=False)}",
        f"COMMAND={subprocess.list2cmdline(command)}",
        "=" * 88,
        "",
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _extract_experiment_dir(stdout_path: Path) -> str:
    if not stdout_path.is_file():
        return ""
    text = stdout_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"EXPERIMENT_DIR=(.+)", text)
    return matches[-1].strip() if matches else ""


def _completed_experiment_dir(output_root: Path, experiment_name: str) -> Path | None:
    if not output_root.is_dir():
        return None
    candidates = sorted(
        [path for path in output_root.glob(f"{experiment_name}*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if all((candidate / item).is_file() for item in REQUIRED_SUMMARY_FILES):
            return candidate
    return None


def _summary_files_complete(output_dir: Path) -> tuple[bool, str]:
    missing = [item for item in REQUIRED_SUMMARY_FILES if not (output_dir / item).is_file()]
    return (not missing, "" if not missing else f"Missing summary files: {missing}")


def _validate_inputs(manifest: pd.DataFrame) -> None:
    for row in manifest.itertuples(index=False):
        config_path = _resolve(row.config_path)
        image_root = _resolve(row.image_root)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config for {row.job_id} does not exist: {config_path}")
        if not image_root.is_dir():
            raise FileNotFoundError(f"Image root for {row.job_id} does not exist: {image_root}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not PYTHON_EXE.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Required Python environment not found: {PYTHON_EXE}")
    manifest = _filter_manifest(_load_manifest(_resolve(args.manifest)), args)
    _validate_inputs(manifest)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    queue = _load_or_create_queue(manifest, bool(args.resume), bool(args.rerun_failed))
    _save_queue(queue)

    if not args.dry_run:
        probe = _gpu_info()
        print(f"GPU_INFO={json.dumps(probe, ensure_ascii=False)}")
        if not bool(probe.get("cuda_available")) and not bool(args.allow_cpu):
            print("CUDA is not available. Training will run on CPU.", file=sys.stderr)
            print("Use --allow-cpu to continue manually.", file=sys.stderr)
            return 2

    manifest_lookup = manifest.set_index("job_id").to_dict("index")
    exit_code = 0
    for queue_index, job in queue.iterrows():
        job_id = str(job["job_id"])
        manifest_row = manifest_lookup[job_id]
        if args.resume and str(job["status"]) == "SUCCESS":
            continue
        if args.rerun_failed and str(job["status"]) != "FAILED":
            continue
        if not args.rerun_failed and str(job["status"]) in {"SUCCESS", "SKIPPED"}:
            continue

        config_path = _resolve(str(job["config_path"]))
        output_root = _resolve(str(manifest_row["output_root"]))
        experiment_name = str(job["experiment_name"])
        command = [str(PYTHON_EXE), str(RUNNER), "--config", str(config_path)]
        stdout_path = LOG_DIR / f"{experiment_name}_stdout.log"
        stderr_path = LOG_DIR / f"{experiment_name}_stderr.log"

        if args.dry_run:
            print(f"[dry-run] {job_id}: {subprocess.list2cmdline(command)}")
            continue

        start_time = _now()
        completed = _completed_experiment_dir(output_root, experiment_name)
        if args.skip_existing and completed is not None:
            end_time = _now()
            queue.loc[
                queue_index,
                ["status", "start_time", "end_time", "duration_minutes", "output_dir", "exit_code", "error_message"],
            ] = [
                "SKIPPED",
                start_time,
                end_time,
                _duration_minutes(start_time, end_time),
                str(completed),
                0,
                "existing complete summary found",
            ]
            _save_queue(queue)
            continue

        queue.loc[
            queue_index,
            ["status", "start_time", "end_time", "duration_minutes", "output_dir", "exit_code", "error_message"],
        ] = ["RUNNING", start_time, "", "", "", "", ""]
        _save_queue(queue)

        gpu_info = _gpu_info()
        print(
            f"Starting {job_id} {experiment_name}; GPU_INFO={json.dumps(gpu_info, ensure_ascii=False)}"
        )
        _append_log_header(stdout_path, job, gpu_info, command)
        with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr_handle:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=False,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )

        end_time = _now()
        output_dir_text = _extract_experiment_dir(stdout_path)
        status = "SUCCESS" if result.returncode == 0 else "FAILED"
        error_message = "" if result.returncode == 0 else f"exit_code={result.returncode}"
        if status == "SUCCESS":
            if not output_dir_text:
                status = "FAILED"
                error_message = "EXPERIMENT_DIR was not found in stdout"
            else:
                complete, message = _summary_files_complete(Path(output_dir_text))
                if not complete:
                    status = "FAILED"
                    error_message = message

        queue.loc[
            queue_index,
            ["status", "end_time", "duration_minutes", "output_dir", "exit_code", "error_message"],
        ] = [
            status,
            end_time,
            _duration_minutes(start_time, end_time),
            output_dir_text,
            int(result.returncode),
            error_message,
        ]
        _save_queue(queue)
        print(f"Finished {job_id}: status={status}, output_dir={output_dir_text}")
        if status != "SUCCESS":
            exit_code = int(result.returncode) if result.returncode else 1
            if not args.continue_on_error:
                break

    print(f"Job queue: {QUEUE_PATH}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
