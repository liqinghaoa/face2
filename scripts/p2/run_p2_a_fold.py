from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p2_counterfactual.config import resolve_p2_a_config
from p2_counterfactual.metrics import json_safe
from p2_counterfactual.trainer import P2AFoldTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one P2-A single-RGB fold.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--mode", choices=["train", "evaluate", "train-evaluate", "smoke"], default="train-evaluate")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("P2-A training requested CUDA, but torch.cuda.is_available() is false in the active environment")
        return torch.device(requested)
    return torch.device(requested)


def main() -> int:
    args = parse_args()
    config = resolve_p2_a_config(args.config)
    device = choose_device(args.device)
    smoke = args.mode == "smoke"
    mode = "train-evaluate" if smoke else args.mode
    trainer = P2AFoldTrainer(
        config,
        fold=args.fold,
        device=device,
        smoke=smoke,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
    )
    result: dict[str, object] = {
        "experiment_id": config["experiment_id"],
        "fold": int(args.fold),
        "mode": args.mode,
        "device": str(device),
        "output_dir": str(trainer.output_dir),
        "formal_five_fold_executed": False,
        "pooled_oof_generated": False,
        "p2_b_executed": False,
    }
    if mode in {"train", "train-evaluate"}:
        result["train"] = trainer.train(resume=args.resume)
    if mode in {"evaluate", "train-evaluate"}:
        result["evaluate"] = trainer.evaluate()
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
