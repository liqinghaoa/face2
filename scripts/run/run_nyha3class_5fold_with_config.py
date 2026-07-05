"""Run one NYHA three-class five-fold experiment from an arbitrary config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run.run_exp_global224_imagenet_resnet18_nyha3class_5fold import (  # noqa: E402
    load_config,
    preflight,
    resolve_project_path,
    select_output_dir,
)
from utils.experiment_utils import save_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fold", type=int, action="append", dest="folds")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _select_output_dir(config: dict, override: Path | None) -> Path:
    if override is None:
        return select_output_dir(config)
    candidate = resolve_project_path(override)
    if candidate is None:
        raise ValueError("--output-dir must not be empty")
    if candidate.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = candidate.parent / f"{candidate.name}_{timestamp}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print("Running:", subprocess.list2cmdline(command))
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
    )


def main() -> int:
    args = parse_args()
    config_path = _resolve(args.config)
    config = load_config(config_path)
    preflight(config_path, config)
    experiment_dir = _select_output_dir(config, args.output_dir)

    if args.skip_train:
        if args.epochs is not None:
            config["train"]["epochs"] = int(args.epochs)
        save_yaml(config, experiment_dir / "config.yaml")

    train_script = PROJECT_ROOT / "scripts" / "train" / "train_nyha_3class_5fold.py"
    summary_script = (
        PROJECT_ROOT / "scripts" / "evaluate" / "summarize_nyha_3class_5fold.py"
    )

    if not args.skip_train:
        command = [
            sys.executable,
            str(train_script),
            "--config",
            str(config_path),
            "--output-dir",
            str(experiment_dir),
        ]
        for fold in args.folds or []:
            command.extend(["--fold", str(fold)])
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        result = _run(command)
        if result.returncode != 0:
            print(f"Training failed: exit_code={result.returncode}", file=sys.stderr)
            print(f"EXPERIMENT_DIR={experiment_dir}")
            return int(result.returncode)

    if not args.skip_summary:
        result = _run(
            [
                sys.executable,
                str(summary_script),
                "--experiment-dir",
                str(experiment_dir),
            ]
        )
        if result.returncode != 0:
            print(f"Summary failed: exit_code={result.returncode}", file=sys.stderr)
            print(f"EXPERIMENT_DIR={experiment_dir}")
            return int(result.returncode)

    print(f"Experiment completed: {experiment_dir}")
    print(f"EXPERIMENT_DIR={experiment_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
