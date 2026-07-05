"""Resume the interrupted MultiROI5 ResNet50 ROI-fusion experiment."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "ROI_Fusion_500"
    / "MultiROI5_ImageNetResNet50_SharedBackbone_ConcatFusion_NYHA3Class_WeightedCE_5Fold"
)
CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "train"
    / "roi_fusion"
    / "nyha_3class_multiroi5_shared_resnet50_concat_weightedce.yaml"
)
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train" / "train_nyha_3class_5fold.py"
SUMMARY_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate" / "summarize_nyha_3class_5fold.py"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_status(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps({"updated_at": now(), **payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_step(name: str, command: list[str], log, status_path: Path) -> None:
    write_status(
        status_path,
        {
            "status": "running",
            "step": name,
            "command": command,
            "experiment_dir": str(EXPERIMENT_DIR),
        },
    )
    log.write(f"[{now()}] START {name}\n")
    log.write(f"[{now()}] COMMAND {' '.join(command)}\n")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log.write(f"[{now()}] EXIT_CODE {completed.returncode} {name}\n")
    if completed.returncode != 0:
        write_status(
            status_path,
            {
                "status": "failed",
                "step": name,
                "returncode": completed.returncode,
                "experiment_dir": str(EXPERIMENT_DIR),
            },
        )
        raise SystemExit(completed.returncode)


def main() -> int:
    log_path = (
        PROJECT_ROOT
        / "experiments"
        / "ROI_Fusion_500"
        / f"resnet50_resume_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    status_path = log_path.with_suffix(".status.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"[{now()}] RESUME_LAUNCHER_START python={sys.executable}\n")
        run_step(
            "train_resume",
            [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--config",
                str(CONFIG_PATH),
                "--output-dir",
                str(EXPERIMENT_DIR),
                "--resume",
                "--batch-size",
                "8",
            ],
            log,
            status_path,
        )
        run_step(
            "summary",
            [
                sys.executable,
                str(SUMMARY_SCRIPT),
                "--experiment-dir",
                str(EXPERIMENT_DIR),
            ],
            log,
            status_path,
        )
        log.write(f"[{now()}] ALL_DONE\n")

    write_status(
        status_path,
        {"status": "completed", "experiment_dir": str(EXPERIMENT_DIR)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
