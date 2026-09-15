from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from skin_optics_so1.decomposition.gate_adjudication import adjudicate_learnability_gate_v2
from skin_optics_so1.decomposition.smoke_audit import benchmark_loader, resume_audit, verify_smoke_ids


def _result(h_mae: float) -> dict:
    return {"best_summary": {"channels": {"H": {"MAE": h_mae}}}}


def test_gate_v2_can_pass_while_historical_strict_r2_remains_fail() -> None:
    r2 = {"loss_reduction": .99}
    summary = {"channels": {name: {"MAE": .02, "pred": {"std": .1}, "active_region_recall": .5} for name in "MHSP"}}
    result = adjudicate_learnability_gate_v2(r2_best_record=r2, r2_best_summary=summary, overfit1_result=_result(.002), paired_overfit2_result=_result(.001), tests_passed=True)
    assert result["historical_R2_strict_gate"] == "FAIL"
    assert result["learnability_gate_v2"] == "PASS"
    assert result["gate_adjudication_decision"] == "ALLOW_SMOKE"


def test_smoke_access_and_resume_history_audit() -> None:
    access = verify_smoke_ids(["train_000001"], ["validation_000001"])
    assert access["train_validation_disjoint"] and not access["forbidden_splits_accessed"]
    checkpoint = {"epoch": 1, "global_step": 16, "scheduler_state_dict": {}, "grad_scaler_state_dict": {}, "rng_state": {}}
    audit = resume_audit(checkpoint, [{"epoch": 1}, {"epoch": 2}], expected_epoch=2, checkpoint_path=Path("last.pt"))
    assert audit["history_continuous"] and audit["scheduler_restored"] and audit["scaler_restored"] and audit["rng_restored"]


class _ToyDataset(Dataset):
    def __len__(self) -> int: return 20
    def __getitem__(self, index: int): return {"linear_rgb": torch.zeros(3, 2, 2)}


def test_benchmark_audit_returns_timing_fields() -> None:
    result = benchmark_loader(DataLoader(_ToyDataset(), batch_size=2), device=torch.device("cpu"), warmup_batches=1, timed_batches=3)
    assert result["timed_batches"] == 3
    assert result["samples_per_second"] > 0
