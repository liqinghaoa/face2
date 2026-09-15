"""Run the frozen-scope SO-1C-R1 overfit16 failure diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from skin_optics_so1.decomposition.diagnostic_selection import (  # noqa: E402
    select_fixed_camera_light16, select_overfit1, select_paired_overfit2,
)
from skin_optics_so1.decomposition.overfit_diagnostics import (  # noqa: E402
    batchnorm_diagnostic, collect_predictions, diagnostic_train, load_trusted_checkpoint,
    save_diagnostic_summary, save_previews, summarize_collection, write_json,
)
from skin_optics_so1.decomposition.synthetic_dataset import SO1DecompositionDataset  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data/processed/SO1_Synthetic_Generator_v1_localfull"
PREFLIGHT = PROJECT_ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/preflight"
ROOT = PROJECT_ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/diagnostics/overfit_failure_r1"
SEED = 20260801


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("offline", "overfit1", "paired2", "fixed16", "extended16", "bs8", "bs16", "all"),
        default="all",
    )
    return parser.parse_args()


def ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def subset(dataset: SO1DecompositionDataset, sample_ids: list[str]):
    lookup = {str(row.sample_id): index for index, row in enumerate(dataset.metadata.itertuples(index=False))}
    return torch.utils.data.Subset(dataset, [lookup[sample_id] for sample_id in sample_ids])


def loader(dataset, batch_size: int, *, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available(), generator=torch.Generator().manual_seed(SEED))


def evaluate_existing(name: str, checkpoint_path: Path, selected: list[str], train: SO1DecompositionDataset, device: torch.device) -> dict[str, Any]:
    output = ROOT / "baseline_existing_run" / name
    model, checkpoint = load_trusted_checkpoint(checkpoint_path, device)
    sample_loader = loader(subset(train, selected), 4)
    collection = collect_predictions(model, sample_loader, device)
    summary = summarize_collection(collection)
    summary["checkpoint"] = {key: checkpoint.get(key) for key in ("epoch", "global_step", "best_selection_metric", "model_architecture", "target_order")}
    save_diagnostic_summary(output, summary, stem=name)
    save_previews(output, collection, summary, step=int(checkpoint["global_step"]))
    bn = batchnorm_diagnostic(model, sample_loader, device)
    write_json(output / f"{name}_bn_diagnostic.json", bn)
    return {"summary": summary, "bn": bn}


def passed_overfit1(result: dict[str, Any]) -> bool:
    c = result["best_summary"]["channels"]
    return bool(c["M"]["MAE"] <= .01 and c["H"]["MAE"] <= .01 and c["S"]["MAE"] <= .015 and c["P"]["pred"]["std"] > 0 and c["P"]["active_region_recall"] > 0)


def run() -> None:
    value = args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("SO-1C-R1 diagnostic requires CUDA in the Face2 environment")
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("baseline_existing_run", "overfit1", "paired_overfit2", "fixed_camera_light16", "original_overfit16_extended", "original_overfit16_bs8", "original_overfit16_bs16", "selected_ids"):
        (ROOT / name).mkdir(exist_ok=True)
    train = SO1DecompositionDataset(DATA_ROOT, "train")
    original_ids = ids(PREFLIGHT / "overfit16_ids.txt")
    report: dict[str, Any] = {"status": "PARTIAL", "device": str(device), "original_ids": original_ids}
    if value.phase in ("offline", "all"):
        report["best"] = evaluate_existing("best", PREFLIGHT / "overfit16/best.pt", original_ids, train, device)
        report["last"] = evaluate_existing("last", PREFLIGHT / "overfit16/last.pt", original_ids, train, device)
        write_json(ROOT / "baseline_existing_run" / "baseline_summary.json", report)
    if value.phase == "offline":
        return
    metadata = pd.read_csv(DATA_ROOT / "train/metadata.csv")
    if value.phase in ("overfit1", "all"):
        overfit1_ids = select_overfit1(metadata, ROOT / "selected_ids/overfit1_id.txt")
        result1 = diagnostic_train(output_dir=ROOT / "overfit1", train_loader=loader(subset(train, overfit1_ids), 1, shuffle=True), eval_loader=loader(subset(train, overfit1_ids), 1), device=device, seed=SEED, max_steps=1500, evaluation_interval=25)
        report["overfit1"] = result1
        if not passed_overfit1(result1):
            report["status"] = "PASS"
            report["early_stop"] = "overfit1 failed; later training diagnostics intentionally NOT_RUN"
            write_json(ROOT / "diagnostic_summary.json", report)
            return
    if value.phase == "overfit1":
        write_json(ROOT / "diagnostic_summary.json", report); return
    if value.phase in ("paired2", "all"):
        pair_ids = select_paired_overfit2(metadata, ROOT / "selected_ids/paired_overfit2_ids.txt")
        report["paired_overfit2"] = diagnostic_train(output_dir=ROOT / "paired_overfit2", train_loader=loader(subset(train, pair_ids), 2, shuffle=True), eval_loader=loader(subset(train, pair_ids), 2), device=device, seed=SEED, max_steps=1500, evaluation_interval=25)
    if value.phase in ("fixed16", "all"):
        fixed_ids = select_fixed_camera_light16(metadata, ROOT / "selected_ids/fixed_camera_light16_ids.txt")
        report["fixed_camera_light16"] = diagnostic_train(output_dir=ROOT / "fixed_camera_light16", train_loader=loader(subset(train, fixed_ids), 4, shuffle=True), eval_loader=loader(subset(train, fixed_ids), 4), device=device, seed=SEED, max_steps=2000, evaluation_interval=25)
    if value.phase in ("extended16", "all"):
        report["original_overfit16_extended"] = diagnostic_train(output_dir=ROOT / "original_overfit16_extended", train_loader=loader(subset(train, original_ids), 4, shuffle=True), eval_loader=loader(subset(train, original_ids), 4), device=device, seed=SEED, max_steps=3000, evaluation_interval=25)
    if value.phase in ("bs8", "all"):
        report["original_overfit16_bs8"] = diagnostic_train(output_dir=ROOT / "original_overfit16_bs8", train_loader=loader(subset(train, original_ids), 8, shuffle=True), eval_loader=loader(subset(train, original_ids), 8), device=device, seed=SEED, max_steps=2000, evaluation_interval=25)
    if value.phase in ("bs16", "all"):
        try:
            report["original_overfit16_bs16"] = diagnostic_train(output_dir=ROOT / "original_overfit16_bs16", train_loader=loader(subset(train, original_ids), 16, shuffle=True), eval_loader=loader(subset(train, original_ids), 16), device=device, seed=SEED, max_steps=2000, evaluation_interval=25)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache(); report["original_overfit16_bs16"] = {"status": "NOT_RUN", "reason": f"OOM: {exc}"}
    report["status"] = "PASS"
    write_json(ROOT / "diagnostic_summary.json", report)


if __name__ == "__main__":
    run()
