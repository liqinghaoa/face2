"""Finalize corrected P1-RGB evaluation outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate.correct_p1_rgb_evaluation_protocol import run  # noqa: E402
from utils.experiment_utils import load_yaml  # noqa: E402
from utils.p1_cluster_bootstrap import compute_visit_metrics  # noqa: E402


def all_metrics(frame):
    return compute_visit_metrics(frame)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse()
    cfg = load_yaml(_resolve(args.config))
    result = run(
        experiment_dir=_resolve(args.experiment_dir),
        master_manifest=_resolve(cfg["data"]["master_manifest"]),
        fixed_split=_resolve(cfg["data"]["fixed_split"]),
    )
    print(f"P1_RGB_SUMMARY_COMPLETE={json.dumps(result, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
