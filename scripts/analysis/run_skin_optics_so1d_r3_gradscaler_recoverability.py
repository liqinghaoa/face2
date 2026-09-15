"""Adjudicate recovery of the SO-1D-R2 batch-2059 AMP gradient overflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from skin_optics_so1.decomposition.amp_recovery import (  # noqa: E402
    RECOVERY_HISTORY_FIELDS,
    adjudicate_recovery_window,
    adam_state_sha256,
    optimizer_group_sha256,
    parameter_state_sha256,
    scaler_step_and_update,
)
from skin_optics_so1.decomposition.losses import masked_smooth_l1_loss  # noqa: E402
from skin_optics_so1.decomposition.numerical_diagnostics import (  # noqa: E402
    batchnorm_health,
    epoch7_order_from_generator_state,
    gradient_health,
    optimizer_state_health,
    sample_order_sha256,
    state_finite,
    tensor_health,
    write_csv,
    write_json,
)
from skin_optics_so1.decomposition.synthetic_dataset import SO1DecompositionDataset  # noqa: E402
from skin_optics_so1.decomposition.unet_decomposer import SO1UNetDecomposer  # noqa: E402


ROOT = PROJECT_ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1"
FORMAL_LAST = ROOT / "formal_train/checkpoints/last.pt"
FORMAL_BEST = ROOT / "formal_train/checkpoints/best.pt"
R2_DIR = ROOT / "diagnostics/so1d_r2_localized_fp32_repair"
R2_SNAPSHOT = R2_DIR / "repaired_epoch7/failing_batch_snapshot.pt"
OUTPUT_DIR = ROOT / "diagnostics/so1d_r3_gradscaler_recoverability"
DATA_ROOT = PROJECT_ROOT / "data/processed/SO1_Synthetic_Generator_v1_localfull"
POLICY = {"mode": "localized_fp32", "fp32_blocks": ["up1.conv"]}
EXPECTED_ORDER_HASH = "36224640f943244a6ec498226a86ce7d624f8b3486a866d69901ffc098e027b3"
EXPECTED_SAMPLE_IDS = [
    "train_000367", "train_012037", "train_017324", "train_008813",
    "train_006897", "train_018598", "train_015225", "train_005369",
]
FORMAL_HASHES = {
    "last.pt": "85e559ac4cb9a3da5e889b57db7fc45974cdc14f8f8d1312479c1a980e755349",
    "best.pt": "e8c3be6d5f818a049160a97b151cf2a14c82275c7c00f7577523b56123846704",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--pytest-passed", type=int)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_determinism() -> None:
    random.seed(20260801)
    np.random.seed(20260801)
    torch.manual_seed(20260801)
    torch.cuda.manual_seed_all(20260801)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collate_fixed_batch(dataset: SO1DecompositionDataset, indices: list[int]) -> dict[str, Any]:
    samples = [dataset[index] for index in indices]
    result: dict[str, Any] = {
        key: torch.stack([sample[key] for sample in samples])
        for key in ("linear_rgb", "target_mhsp", "valid_mask")
    }
    for key in ("sample_id", "split_index", "base_latent_id"):
        result[key] = [sample[key] for sample in samples]
    return result


def build_runtime(checkpoint: dict[str, Any]) -> tuple[Any, ...]:
    model = SO1UNetDecomposer(numerical_precision=POLICY).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=0.000001
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
    return model, optimizer, scheduler, scaler


def build_order(dataset: SO1DecompositionDataset, checkpoint: dict[str, Any]) -> tuple[list[int], str]:
    order = epoch7_order_from_generator_state(
        sample_count=len(dataset), batch_size=8,
        generator_state=checkpoint["dataloader_rng_state"]["train"],
    )
    metadata = dataset.metadata.set_index("split_index", drop=False)
    rows = []
    for position, split_index in enumerate(order):
        row = metadata.loc[split_index]
        rows.append({
            "batch_index": position // 8,
            "position_in_batch": position % 8,
            "split_index": int(split_index),
            "sample_id": str(row["sample_id"]),
            "base_latent_id": int(row["base_latent_id"]),
            "acquisition_variant_id": int(row["acquisition_variant_id"]),
            "camera": str(row["camera_name"]),
            "light": str(row["light_name"]),
            "camera_light_pair": str(row["camera_light_pair"]),
        })
    digest = sample_order_sha256(rows)
    if digest != EXPECTED_ORDER_HASH:
        write_json(OUTPUT_DIR / "final_status.json", {
            "SO-1D-R3": "FAIL", "reason": "ORDER_MISMATCH",
            "actual": digest, "expected": EXPECTED_ORDER_HASH,
        })
        raise RuntimeError("SO-1D-R3 FAIL: ORDER_MISMATCH")
    return order, digest


def finite_state(model: Any, optimizer: Any) -> tuple[dict[str, Any], bool]:
    state = state_finite(model, optimizer)
    finite = all(state[key] for key in (
        "parameters_finite", "optimizer_state_finite", "batchnorm_buffers_finite"
    ))
    return state, bool(finite)


def bn_finite(model: Any) -> bool:
    return all(row["finite"] for row in batchnorm_health(model))


def adam_finite(model: Any, optimizer: Any) -> bool:
    return all(row["finite"] for row in optimizer_state_health(optimizer, model))


def run_iteration(
    *, model: Any, optimizer: Any, scaler: Any, batch: dict[str, Any],
    batch_index: int, allow_recoverable_overflow: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x = batch["linear_rgb"].cuda(non_blocking=True)
    target = batch["target_mhsp"].cuda(non_blocking=True)
    mask = batch["valid_mask"].cuda(non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=True):
        prediction = model(x)
        losses = masked_smooth_l1_loss(prediction, target, mask, beta=0.1)
    forward_finite = tensor_health(prediction)["finite"]
    loss_finite = all(tensor_health(value)["finite"] for value in losses.values())
    bn_after_forward = bn_finite(model)
    if not forward_finite or not loss_finite or not bn_after_forward:
        raise FloatingPointError(
            f"R3 hard failure at batch {batch_index}: forward/loss/BN nonfinite"
        )
    scale_before = float(scaler.get_scale())
    scaler.scale(losses["total"]).backward()
    scaler.unscale_(optimizer)
    gradient_rows, gradient_summary = gradient_health(model)
    gradient_finite = bool(gradient_summary["finite"])
    overflow = not gradient_finite
    before = (
        {
            "parameter_sha256": parameter_state_sha256(model),
            "adam_sha256": adam_state_sha256(optimizer, model),
            "optimizer_group_sha256": optimizer_group_sha256(optimizer),
        }
        if overflow else {}
    )
    step = scaler_step_and_update(scaler, optimizer)
    after = (
        {
            "parameter_sha256": parameter_state_sha256(model),
            "adam_sha256": adam_state_sha256(optimizer, model),
            "optimizer_group_sha256": optimizer_group_sha256(optimizer),
        }
        if overflow else {}
    )
    state, state_ok = finite_state(model, optimizer)
    if overflow:
        unchanged = before == after
        if not allow_recoverable_overflow:
            raise FloatingPointError(f"Unexpected nonfinite gradient at batch {batch_index}")
        if not step["optimizer_step_skipped"] or not unchanged or not state_ok:
            raise RuntimeError("GRADSCALER_RECOVERY_FAILURE")
        if not step["scale_after"] < step["scale_before"]:
            raise RuntimeError("GRADSCALER_BACKOFF_FAIL")
    elif not step["optimizer_step_executed"]:
        raise RuntimeError(f"Finite gradient unexpectedly skipped at batch {batch_index}")
    row = {
        "batch_index": batch_index,
        "global_step": 15001 + batch_index,
        "scale_before": scale_before,
        "scale_after": step["scale_after"],
        "forward_finite": forward_finite,
        "loss_finite": loss_finite,
        "gradient_finite": gradient_finite,
        "overflow_detected": overflow,
        "optimizer_step_skipped": step["optimizer_step_skipped"],
        "optimizer_step_executed": step["optimizer_step_executed"],
        "parameters_finite": state["parameters_finite"],
        "BN_finite": state["batchnorm_buffers_finite"],
        "Adam_finite": state["optimizer_state_finite"],
    }
    detail = {
        "row": row, "sample_ids": batch["sample_id"],
        "losses": {key: float(value.detach().cpu()) for key, value in losses.items()},
        "first_nonfinite_gradient": gradient_summary["first_nonfinite_parameter"],
        "global_grad_norm": gradient_summary["global_grad_norm"],
        "gradient_summary": gradient_summary,
        "state_hashes_before_step": before,
        "state_hashes_after_step": after,
        "state_unchanged": before == after,
        "step": step,
        "scaler_state": scaler.state_dict(),
        "bn_after_forward_finite": bn_after_forward,
        "gradient_rows": gradient_rows,
    }
    return row, detail


def r2_asset_audit() -> dict[str, Any]:
    snapshot = torch.load(R2_SNAPSHOT, map_location="cpu", weights_only=False)
    audit = {
        "path": str(R2_SNAPSHOT),
        "batch_index": int(snapshot["event"]["batch_index"]),
        "global_step": int(snapshot["event"]["global_step"]),
        "has_model_state": "model_state_dict" in snapshot,
        "has_optimizer_state": "optimizer_state_dict" in snapshot,
        "has_gradscaler_state": "grad_scaler_state_dict" in snapshot,
        "has_rng_state": "rng_state" in snapshot or "torch_cpu_rng_state" in snapshot,
        "has_sample_order_state": "dataloader_rng_state" in snapshot,
        "state_semantics": "after batch2059 forward/backward, before optimizer step",
        "qualified_pre_batch2059_state": False,
        "decision": "REPLAY_FROM_FORMAL_EPOCH6_LAST",
    }
    write_json(OUTPUT_DIR / "r2_asset_audit.json", audit)
    return audit


def formal_hash_audit() -> dict[str, Any]:
    current = {"last.pt": file_sha256(FORMAL_LAST), "best.pt": file_sha256(FORMAL_BEST)}
    result = {"expected": FORMAL_HASHES, "current": current, "unchanged": current == FORMAL_HASHES}
    write_json(OUTPUT_DIR / "formal_checkpoint_hash_audit.json", result)
    if not result["unchanged"]:
        raise RuntimeError("R3 modified a formal checkpoint")
    return result


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    r2_asset_audit()
    set_determinism()
    checkpoint = torch.load(FORMAL_LAST, map_location="cuda", weights_only=False)
    model, optimizer, scheduler, scaler = build_runtime(checkpoint)
    model.train()
    dataset = SO1DecompositionDataset(DATA_ROOT, "train")
    order, order_hash = build_order(dataset, checkpoint)
    started = time.perf_counter()
    replay_overflows: list[dict[str, Any]] = []
    for batch_index in range(2059):
        batch = collate_fixed_batch(dataset, order[batch_index * 8:(batch_index + 1) * 8])
        row, _detail = run_iteration(
            model=model, optimizer=optimizer, scaler=scaler, batch=batch,
            batch_index=batch_index, allow_recoverable_overflow=True,
        )
        if row["overflow_detected"]:
            replay_overflows.append(row)
        if (batch_index + 1) % 100 == 0:
            write_json(OUTPUT_DIR / "reconstruction_progress.json", {
                "batches_completed": batch_index + 1,
                "last_global_step": 15001 + batch_index,
                "recoverable_overflows": len(replay_overflows),
                "elapsed_seconds": time.perf_counter() - started,
            })
    pre_state, pre_finite = finite_state(model, optimizer)
    actual_batch = collate_fixed_batch(dataset, order[2059 * 8:2060 * 8])
    if actual_batch["sample_id"] != EXPECTED_SAMPLE_IDS:
        raise RuntimeError("R3 batch2059 sample IDs do not match R2")
    pre_health = {
        "status": "PASS" if pre_finite else "FAIL",
        "batch_index": 2059, "global_step_before": 17059,
        "epoch7_order_sha256": order_hash,
        "sample_ids": actual_batch["sample_id"],
        **pre_state,
        "adam_exp_avg_and_exp_avg_sq_finite": adam_finite(model, optimizer),
        "gradscaler_valid": math.isfinite(float(scaler.get_scale())) and scaler.get_scale() > 0,
        "gradscaler_scale": float(scaler.get_scale()),
        "gradscaler_state": scaler.state_dict(),
        "scheduler_valid": all(
            not isinstance(value, float) or math.isfinite(value)
            for value in scheduler.state_dict().values()
        ),
        "reconstruction_recoverable_overflows": replay_overflows,
        "source": str(FORMAL_LAST),
    }
    write_json(OUTPUT_DIR / "r3_pre_batch2059_health.json", pre_health)
    if not pre_finite:
        raise RuntimeError("R3 pre-batch2059 state is not finite")

    row2059, detail2059 = run_iteration(
        model=model, optimizer=optimizer, scaler=scaler, batch=actual_batch,
        batch_index=2059, allow_recoverable_overflow=True,
    )
    reproduced = (
        row2059["overflow_detected"]
        and detail2059["first_nonfinite_gradient"] == "inc.net.0.weight"
    )
    reproduction = {
        "BATCH2059_OVERFLOW_REPRODUCED": reproduced,
        "exact_order": True,
        "sample_ids_match": actual_batch["sample_id"] == EXPECTED_SAMPLE_IDS,
        "forward": "FINITE" if row2059["forward_finite"] else "NON_FINITE",
        "loss": "FINITE" if row2059["loss_finite"] else "NON_FINITE",
        "all_model_outputs_finite": row2059["forward_finite"],
        "bn_after_forward_finite": detail2059["bn_after_forward_finite"],
        "unscaled_gradients_finite": row2059["gradient_finite"],
        "first_nonfinite_gradient": detail2059["first_nonfinite_gradient"],
        "global_grad_norm": (
            detail2059["global_grad_norm"]
            if math.isfinite(float(detail2059["global_grad_norm"])) else "Infinity"
        ),
        "scale_before": row2059["scale_before"],
    }
    write_json(OUTPUT_DIR / "batch2059_reproduction.json", reproduction)
    write_csv(
        OUTPUT_DIR / "batch2059_gradient_health.csv",
        detail2059.pop("gradient_rows"),
    )
    if not reproduced:
        final = {
            "SO-1D-R3": "INCONCLUSIVE", "primary_diagnosis": "NON_REPRODUCIBLE",
            "AMP_RECOVERY_STABLE": "UNKNOWN", "localized_fp32_repair_still_viable": False,
            "allow_SO1D_v1_1_protocol_decision": False,
            "allow_automatic_formal_training": False, "allow_SO1E": False,
        }
        write_json(OUTPUT_DIR / "final_status.json", final)
        write_report(final, reproduction, {}, {}, None)
        return

    recovery2059 = {
        "scaler_step_called": detail2059["step"]["scaler_step_called"],
        "optimizer_step_skipped": row2059["optimizer_step_skipped"],
        "parameters_unchanged": (
            detail2059["state_hashes_before_step"]["parameter_sha256"]
            == detail2059["state_hashes_after_step"]["parameter_sha256"]
        ),
        "adam_state_unchanged": (
            detail2059["state_hashes_before_step"]["adam_sha256"]
            == detail2059["state_hashes_after_step"]["adam_sha256"]
        ),
        "optimizer_group_state_unchanged": (
            detail2059["state_hashes_before_step"]["optimizer_group_sha256"]
            == detail2059["state_hashes_after_step"]["optimizer_group_sha256"]
        ),
        "BN_SAFE_AFTER_GRADIENT_OVERFLOW": row2059["BN_finite"],
        "parameters_finite_after": row2059["parameters_finite"],
        "adam_finite_after": row2059["Adam_finite"],
        "scale_before": row2059["scale_before"],
        "scale_after": row2059["scale_after"],
        "scale_decreased": row2059["scale_after"] < row2059["scale_before"],
        "backoff_factor": checkpoint["grad_scaler_state_dict"]["backoff_factor"],
        "growth_factor": checkpoint["grad_scaler_state_dict"]["growth_factor"],
        "growth_interval": checkpoint["grad_scaler_state_dict"]["growth_interval"],
        "state_hashes_before_step": detail2059["state_hashes_before_step"],
        "state_hashes_after_step": detail2059["state_hashes_after_step"],
    }
    write_json(OUTPUT_DIR / "batch2059_gradscaler_recovery.json", recovery2059)
    optimizer.zero_grad(set_to_none=True)
    gradients_cleared = all(parameter.grad is None for parameter in model.parameters())
    if not gradients_cleared:
        raise RuntimeError("R3 failed to clear overflow gradients")

    history = [row2059]
    recovery_rows: list[dict[str, Any]] = []
    overflow_events: list[dict[str, Any]] = []
    first_successful: int | None = None
    hard_failure: dict[str, Any] | None = None
    for batch_index in range(2060, 2160):
        batch = collate_fixed_batch(dataset, order[batch_index * 8:(batch_index + 1) * 8])
        try:
            row, detail = run_iteration(
                model=model, optimizer=optimizer, scaler=scaler, batch=batch,
                batch_index=batch_index, allow_recoverable_overflow=True,
            )
        except Exception as exc:
            hard_failure = {"batch_index": batch_index, "error": type(exc).__name__, "message": str(exc)}
            break
        history.append(row)
        recovery_rows.append(row)
        if row["overflow_detected"]:
            overflow_events.append({
                "overflow_event_index": len(overflow_events) + 1,
                "batch_index": batch_index,
                "scale_before": row["scale_before"], "scale_after": row["scale_after"],
                "step_skipped": row["optimizer_step_skipped"],
                "parameters_finite_after": row["parameters_finite"],
                "BN_finite_after": row["BN_finite"], "Adam_finite_after": row["Adam_finite"],
                "first_nonfinite_gradient": detail["first_nonfinite_gradient"],
            })
        elif first_successful is None and row["optimizer_step_executed"]:
            first_successful = batch_index
    write_csv(OUTPUT_DIR / "gradscaler_recovery_history.csv", history, RECOVERY_HISTORY_FIELDS)
    adjudication = adjudicate_recovery_window(recovery_rows)
    final_state, final_finite = finite_state(model, optimizer)
    summary = {
        "recovery_window": 100,
        "completed": len(recovery_rows),
        "first_successful_optimizer_update_batch": first_successful,
        "additional_recoverable_overflows": len(overflow_events),
        "overflow_events": overflow_events,
        "last_50_batches_overflow_free": adjudication["last_50_overflow_free"],
        "AMP_RECOVERY_STABLE": adjudication["AMP_RECOVERY_STABLE"] and hard_failure is None,
        "hard_failure": hard_failure,
        **final_state,
        "final_state_finite": final_finite,
        "gradscaler_valid": math.isfinite(float(scaler.get_scale())) and scaler.get_scale() > 0,
        "final_scale": float(scaler.get_scale()),
        "overflow_gradients_cleared_after_batch2059": gradients_cleared,
    }
    write_json(OUTPUT_DIR / "recovery_window_summary.json", summary)
    pass_status = all((
        reproduced, recovery2059["optimizer_step_skipped"],
        recovery2059["parameters_unchanged"], recovery2059["adam_state_unchanged"],
        recovery2059["BN_SAFE_AFTER_GRADIENT_OVERFLOW"],
        recovery2059["scale_decreased"], summary["AMP_RECOVERY_STABLE"],
        len(recovery_rows) == 100, final_finite,
    ))
    final = {
        "SO-1D-R3": "PASS" if pass_status else "FAIL",
        "primary_diagnosis": (
            "RECOVERABLE_AMP_GRADIENT_OVERFLOW"
            if pass_status else "UNRECOVERABLE_AMP_BACKWARD_INSTABILITY"
        ),
        "AMP_RECOVERY_STABLE": "YES" if summary["AMP_RECOVERY_STABLE"] else "NO",
        "localized_fp32_repair_still_viable": pass_status,
        "allow_SO1D_v1_1_protocol_decision": pass_status,
        "allow_automatic_formal_training": False,
        "allow_SO1E": False,
        "validation_run": False,
        "diagnostic_checkpoint_generated": False,
    }
    write_json(OUTPUT_DIR / "final_status.json", final)
    formal_hash_audit()
    write_report(final, reproduction, recovery2059, summary, None)


def write_report(
    final: dict[str, Any], reproduction: dict[str, Any], recovery: dict[str, Any],
    summary: dict[str, Any], pytest_result: int | None,
) -> None:
    sections = [
        "# SO-1D-R3 GradScaler Recoverability Report", "",
        "## 1. R1/R2 Background", "",
        "R1 found a forward FP16 overflow. R2 resolved that event with localized FP32 but stopped on a non-finite gradient at batch 2059 before invoking GradScaler step semantics.", "",
        "## 2. R3 Question", "",
        "Is the batch-2059 backward overflow recoverable by standard dynamic loss scaling?", "",
        "## 3. Batch2059 Reproduction", "", json.dumps(reproduction, ensure_ascii=False), "",
        "## 4. Forward And Loss", "", f"Forward={reproduction.get('forward')}; loss={reproduction.get('loss')}.", "",
        "## 5. First Nonfinite Gradient", "", str(reproduction.get("first_nonfinite_gradient")), "",
        "## 6. Scale Before", "", str(recovery.get("scale_before")), "",
        "## 7. scaler.step Behavior", "", str(recovery.get("scaler_step_called")), "",
        "## 8. Optimizer Skip", "", str(recovery.get("optimizer_step_skipped")), "",
        "## 9. Scale After", "", str(recovery.get("scale_after")), "",
        "## 10. Parameter Safety", "", str(recovery.get("parameters_unchanged")), "",
        "## 11. BatchNorm Safety", "", str(recovery.get("BN_SAFE_AFTER_GRADIENT_OVERFLOW")), "",
        "## 12. Adam Safety", "", str(recovery.get("adam_state_unchanged")), "",
        "## 13. First Recovered Update", "", str(summary.get("first_successful_optimizer_update_batch")), "",
        "## 14. Recovery Window", "", f"{summary.get('completed')} / 100 batches.", "",
        "## 15. Recoverable Overflow Count", "", str(summary.get("additional_recoverable_overflows")), "",
        "## 16. Last 50 Stability", "", str(summary.get("last_50_batches_overflow_free")), "",
        "## 17. GradScaler Trajectory", "", "See `gradscaler_recovery_history.csv`.", "",
        "## 18. Pytest", "", (f"{pytest_result} passed, 0 failed" if pytest_result is not None else "Pending final regression."), "",
        "## 19. R3 Conclusion", "", json.dumps(final, ensure_ascii=False), "",
        "## 20. AMP Route", "",
        ("AMP plus localized FP32 remains numerically viable." if final.get("SO-1D-R3") == "PASS" else "AMP plus localized FP32 is not validated by R3."), "",
        "No Validation, formal resume/restart, formal checkpoint update, epoch8, or SO-1E was executed.", "",
    ]
    (OUTPUT_DIR / "SO1D_R3_gradscaler_recoverability_report.md").write_text(
        "\n".join(sections), encoding="utf-8"
    )


def finalize(pytest_passed: int) -> None:
    final = json.loads((OUTPUT_DIR / "final_status.json").read_text(encoding="utf-8"))
    reproduction = json.loads((OUTPUT_DIR / "batch2059_reproduction.json").read_text(encoding="utf-8"))
    recovery_path = OUTPUT_DIR / "batch2059_gradscaler_recovery.json"
    summary_path = OUTPUT_DIR / "recovery_window_summary.json"
    recovery = json.loads(recovery_path.read_text(encoding="utf-8")) if recovery_path.is_file() else {}
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    write_json(OUTPUT_DIR / "regression_tests.json", {"passed": pytest_passed, "failed": 0})
    formal_hash_audit()
    write_report(final, reproduction, recovery, summary, pytest_passed)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SO-1D-R3 requires CUDA")
    if args.finalize_only:
        if args.pytest_passed is None:
            raise ValueError("--pytest-passed is required with --finalize-only")
        finalize(args.pytest_passed)
    else:
        run()


if __name__ == "__main__":
    main()
