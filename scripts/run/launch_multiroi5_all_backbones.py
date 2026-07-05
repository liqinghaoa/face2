"""Persistent launcher for the MultiROI5 ROI-fusion backbone sweep.

This script intentionally avoids shell pipelines so it can be started as a
detached Python process on Windows and monitored via log/status files.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "ROI_Fusion_500"
RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run" / "run_exp_roi_fusion_nyha3class_5fold.py"
CONFIGS = [
    PROJECT_ROOT
    / "config"
    / "train"
    / "roi_fusion"
    / "nyha_3class_multiroi5_shared_resnet18_concat_weightedce.yaml",
    PROJECT_ROOT
    / "config"
    / "train"
    / "roi_fusion"
    / "nyha_3class_multiroi5_shared_resnet34_concat_weightedce.yaml",
    PROJECT_ROOT
    / "config"
    / "train"
    / "roi_fusion"
    / "nyha_3class_multiroi5_shared_resnet50_concat_weightedce.yaml",
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_status(path: Path, payload: dict) -> None:
    payload = {"updated_at": now(), **payload}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_ROOT / f"multiroi5_all_backbones_python_launcher_{run_id}.log"
    status_path = OUTPUT_ROOT / f"multiroi5_all_backbones_python_launcher_{run_id}.status.json"

    write_status(
        status_path,
        {
            "status": "running",
            "run_id": run_id,
            "python": sys.executable,
            "launcher_log": str(log_path),
            "configs": [str(path) for path in CONFIGS],
        },
    )

    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"[{now()}] LAUNCHER_START python={sys.executable}\n")
        log.write(f"[{now()}] project_root={PROJECT_ROOT}\n")
        for index, config_path in enumerate(CONFIGS, start=1):
            command = [
                sys.executable,
                str(RUN_SCRIPT),
                "--config",
                str(config_path),
            ]
            write_status(
                status_path,
                {
                    "status": "running",
                    "run_id": run_id,
                    "current_index": index,
                    "current_config": str(config_path),
                    "launcher_log": str(log_path),
                },
            )
            log.write(f"[{now()}] START {index}/{len(CONFIGS)} {config_path}\n")
            log.write(f"[{now()}] COMMAND {' '.join(command)}\n")
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log.write(
                f"[{now()}] EXIT_CODE {completed.returncode} "
                f"{index}/{len(CONFIGS)} {config_path}\n"
            )
            if completed.returncode != 0:
                write_status(
                    status_path,
                    {
                        "status": "failed",
                        "run_id": run_id,
                        "failed_index": index,
                        "failed_config": str(config_path),
                        "returncode": completed.returncode,
                        "launcher_log": str(log_path),
                    },
                )
                return completed.returncode

        log.write(f"[{now()}] ALL_DONE\n")

    write_status(
        status_path,
        {
            "status": "completed",
            "run_id": run_id,
            "launcher_log": str(log_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
