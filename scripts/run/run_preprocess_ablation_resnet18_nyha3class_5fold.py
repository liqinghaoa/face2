"""Serially run ResNet18 five-fold training for preprocessing ablations."""

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
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "train"
    / "preprocess_ablation_resnet18"
    / "config_manifest.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "preprocess_ablation_500Data"
LOG_DIR = OUTPUT_ROOT / "logs"
QUEUE_PATH = OUTPUT_ROOT / "job_queue.csv"
BATCH_LOG_CSV = OUTPUT_ROOT / "batch_run_log.csv"
BATCH_LOG_MD = OUTPUT_ROOT / "batch_run_log.md"
RUNNER = PROJECT_ROOT / "scripts" / "run" / "run_nyha3class_5fold_with_config.py"
SUMMARY_SCRIPT = (
    PROJECT_ROOT / "scripts" / "evaluate" / "summarize_preprocess_ablation_experiments.py"
)
VARIANT_ORDER = (
    "hybrid_black_baseline",
    "hybrid_imagenet_meanbg",
    "hybrid_black_labl_norm",
    "hybrid_imagenet_meanbg_labl_norm",
    "hybrid_black_clahe_l",
    "hybrid_black_gray3ch",
    "hybrid_black_masked_grayworld_wb",
    "hybrid_black_retinex_msr",
)
REQUIRED_SUMMARY_FILES = (
    "summary/mean_metrics.csv",
    "summary/oof_metrics.csv",
    "summary/fold_metrics_all.csv",
    "summary/oof_predictions.csv",
    "summary/summary_report.md",
)
QUEUE_COLUMNS = (
    "job_id",
    "variant_name",
    "config_path",
    "image_root",
    "status",
    "start_time",
    "end_time",
    "duration_minutes",
    "output_dir",
    "exit_code",
    "error_message",
)
BATCH_COLUMNS = (
    "variant_name",
    "config_path",
    "image_root",
    "experiment_name",
    "start_time",
    "end_time",
    "duration_minutes",
    "status",
    "exit_code",
    "output_dir",
    "error_message",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--start-from", type=str, default=None)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow long training to continue when CUDA is unavailable.",
    )
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _duration_minutes(start: str, end: str) -> float:
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    return round((end_dt - start_dt).total_seconds() / 60.0, 4)


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest does not exist: {path}. Run generate_preprocess_ablation_train_configs.py first."
        )
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    required = {"variant_name", "config_path", "image_root", "experiment_name", "output_dir"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    order = {name: index for index, name in enumerate(VARIANT_ORDER)}
    frame["_order"] = frame["variant_name"].map(order)
    if frame["_order"].isna().any():
        unknown = frame.loc[frame["_order"].isna(), "variant_name"].tolist()
        raise ValueError(f"Manifest contains unknown variants: {unknown}")
    return frame.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def _filter_manifest(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    result = frame.copy()
    if args.start_from:
        if args.start_from not in VARIANT_ORDER:
            raise ValueError(f"--start-from must be a variant name, got {args.start_from}")
        start_index = VARIANT_ORDER.index(args.start_from)
        allowed = set(VARIANT_ORDER[start_index:])
        result = result[result["variant_name"].isin(allowed)]
    if args.only:
        names = [item.strip() for item in args.only.split(",") if item.strip()]
        unknown = sorted(set(names).difference(VARIANT_ORDER))
        if unknown:
            raise ValueError(f"--only contains unknown variants: {unknown}")
        result = result[result["variant_name"].isin(names)]
    if args.max_experiments is not None:
        result = result.head(int(args.max_experiments))
    return result.reset_index(drop=True)


def _initial_queue(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        rows.append(
            {
                "job_id": f"E{VARIANT_ORDER.index(row.variant_name)}",
                "variant_name": row.variant_name,
                "config_path": row.config_path,
                "image_root": row.image_root,
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


def _load_or_create_queue(manifest: pd.DataFrame, resume: bool, rerun_failed: bool) -> pd.DataFrame:
    if QUEUE_PATH.is_file() and (resume or rerun_failed):
        existing = pd.read_csv(QUEUE_PATH, dtype=str, encoding="utf-8-sig").fillna("")
        for column in QUEUE_COLUMNS:
            if column not in existing.columns:
                existing[column] = ""
        existing = existing[list(QUEUE_COLUMNS)]
        wanted = set(manifest["variant_name"].astype(str))
        existing = existing[existing["variant_name"].isin(wanted)].copy()
        present = set(existing["variant_name"].astype(str))
        missing = manifest[~manifest["variant_name"].isin(present)]
        if not missing.empty:
            existing = pd.concat([existing, _initial_queue(missing)], ignore_index=True)
        order = {name: index for index, name in enumerate(VARIANT_ORDER)}
        existing["_order"] = existing["variant_name"].map(order)
        return existing.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return _initial_queue(manifest)


def _save_queue(queue: pd.DataFrame) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    queue.to_csv(QUEUE_PATH, index=False, encoding="utf-8-sig")


def _completed_experiment_dir(output_root: Path, experiment_name: str) -> Path | None:
    candidates = sorted(output_root.glob(f"{experiment_name}*"), reverse=True)
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / item).is_file() for item in REQUIRED_SUMMARY_FILES):
            return candidate
    return None


def _gpu_info() -> dict[str, Any]:
    code = (
        "import json, torch\n"
        "info={'cuda_available': torch.cuda.is_available(), "
        "'device_count': torch.cuda.device_count(), 'current_device': 'cpu', "
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
            "device_name": "",
            "memory_allocated": 0,
            "memory_reserved": 0,
            "error": result.stderr.strip(),
        }
    return json.loads(result.stdout.strip())


def _extract_experiment_dir(stdout_path: Path) -> str:
    if not stdout_path.is_file():
        return ""
    text = stdout_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"EXPERIMENT_DIR=(.+)", text)
    return matches[-1].strip() if matches else ""


def _append_gpu_header(stdout_path: Path, gpu_info: dict[str, Any]) -> None:
    with stdout_path.open("a", encoding="utf-8") as handle:
        handle.write("GPU_INFO=" + json.dumps(gpu_info, ensure_ascii=False) + "\n")


def _write_batch_logs(rows: list[dict[str, Any]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=BATCH_COLUMNS)
    frame.to_csv(BATCH_LOG_CSV, index=False, encoding="utf-8-sig")
    lines = [
        "# Preprocess Ablation Batch Run Log",
        "",
        "| variant_name | status | exit_code | duration_minutes | output_dir | error_message |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {variant_name} | {status} | {exit_code} | {duration_minutes} | "
            "{output_dir} | {error_message} |".format(**{key: str(row.get(key, "")) for key in BATCH_COLUMNS})
        )
    BATCH_LOG_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_files_complete(output_dir: Path) -> tuple[bool, str]:
    missing = [item for item in REQUIRED_SUMMARY_FILES if not (output_dir / item).is_file()]
    return (not missing, "" if not missing else f"Missing summary files: {missing}")


def _run_summary_all() -> None:
    result = subprocess.run(
        [str(PYTHON_EXE), str(SUMMARY_SCRIPT)],
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        print(f"[warning] Cross-experiment summary failed: exit_code={result.returncode}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not PYTHON_EXE.is_file():
        raise FileNotFoundError(f"Required Python environment not found: {PYTHON_EXE}")
    manifest = _filter_manifest(_load_manifest(_resolve(args.manifest)), args)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    queue = _load_or_create_queue(manifest, bool(args.resume), bool(args.rerun_failed))
    _save_queue(queue)

    if not args.dry_run:
        probe = _gpu_info()
        if not bool(probe.get("cuda_available")) and not bool(args.allow_cpu):
            print("CUDA is not available. Training will run on CPU.")
            print("Use --allow-cpu to continue manually, or stop and check GPU with nvidia-smi.")
            return 2

    manifest_lookup = manifest.set_index("variant_name").to_dict("index")
    batch_rows: list[dict[str, Any]] = []
    exit_code = 0
    for queue_index, job in queue.iterrows():
        variant = str(job["variant_name"])
        manifest_row = manifest_lookup[variant]
        if args.resume and str(job["status"]) == "SUCCESS":
            continue
        if args.rerun_failed and str(job["status"]) != "FAILED":
            continue
        if not args.rerun_failed and str(job["status"]) in {"SUCCESS", "SKIPPED"}:
            continue

        config_path = _resolve(str(job["config_path"]))
        image_root = str(job["image_root"])
        experiment_name = str(manifest_row["experiment_name"])
        experiment_root = _resolve(str(manifest_row["output_dir"]))
        command = [str(PYTHON_EXE), str(RUNNER), "--config", str(config_path)]
        start_time = _now()

        completed = _completed_experiment_dir(experiment_root, experiment_name)
        if args.skip_existing and completed is not None:
            end_time = _now()
            queue.loc[queue_index, ["status", "start_time", "end_time", "duration_minutes", "output_dir", "exit_code", "error_message"]] = [
                "SKIPPED",
                start_time,
                end_time,
                _duration_minutes(start_time, end_time),
                str(completed),
                0,
                "existing complete summary found",
            ]
            _save_queue(queue)
            batch_rows.append(
                {
                    "variant_name": variant,
                    "config_path": str(config_path),
                    "image_root": image_root,
                    "experiment_name": experiment_name,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_minutes": _duration_minutes(start_time, end_time),
                    "status": "SKIPPED",
                    "exit_code": 0,
                    "output_dir": str(completed),
                    "error_message": "existing complete summary found",
                }
            )
            continue

        if args.dry_run:
            print("DRY-RUN:", subprocess.list2cmdline(command))
            end_time = _now()
            batch_rows.append(
                {
                    "variant_name": variant,
                    "config_path": str(config_path),
                    "image_root": image_root,
                    "experiment_name": experiment_name,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_minutes": _duration_minutes(start_time, end_time),
                    "status": "DRY_RUN",
                    "exit_code": "",
                    "output_dir": "",
                    "error_message": "",
                }
            )
            continue

        queue.loc[queue_index, ["status", "start_time", "end_time", "duration_minutes", "output_dir", "exit_code", "error_message"]] = [
            "RUNNING",
            start_time,
            "",
            "",
            "",
            "",
            "",
        ]
        _save_queue(queue)
        stdout_path = LOG_DIR / f"{variant}_train_stdout.log"
        stderr_path = LOG_DIR / f"{variant}_train_stderr.log"
        gpu_info = _gpu_info()
        _append_gpu_header(stdout_path, gpu_info)
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
            complete, message = _summary_files_complete(Path(output_dir_text))
            if not complete:
                status = "FAILED"
                error_message = message
        queue.loc[queue_index, ["status", "end_time", "duration_minutes", "output_dir", "exit_code", "error_message"]] = [
            status,
            end_time,
            _duration_minutes(start_time, end_time),
            output_dir_text,
            int(result.returncode),
            error_message,
        ]
        _save_queue(queue)
        batch_rows.append(
            {
                "variant_name": variant,
                "config_path": str(config_path),
                "image_root": image_root,
                "experiment_name": experiment_name,
                "start_time": start_time,
                "end_time": end_time,
                "duration_minutes": _duration_minutes(start_time, end_time),
                "status": status,
                "exit_code": int(result.returncode),
                "output_dir": output_dir_text,
                "error_message": error_message,
            }
        )
        if status != "SUCCESS":
            exit_code = int(result.returncode) if result.returncode else 1
            if not args.continue_on_error:
                break

    _write_batch_logs(batch_rows)
    if not args.dry_run:
        _run_summary_all()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
