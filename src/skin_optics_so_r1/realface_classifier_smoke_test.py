"""SO-R2-X3-S0 frozen-input classifier smoke test.

This module intentionally performs one in-memory batch only.  It never writes
model checkpoints and never reads outer-test images.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = ROOT / "data/processed/SO_R2X3_ClassifierInputs_v1"
REPORT_ROOT = ROOT / "reports/so_r2x3_classifier_smoke_test"
SPLIT_TABLE = ROOT / "data/processed/P0_Physics_Audit_v1/splits_500/nyha_2class_sex_stratified_group_5fold.csv"
CONDITIONS = ("RGB", "RGB_B1MH", "RGB_B2MH")
MEAN_RGB = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
STD_RGB = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
MEAN_MH = torch.tensor([0.5, 0.5], dtype=torch.float32)
STD_MH = torch.tensor([0.5, 0.5], dtype=torch.float32)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _finite(a: torch.Tensor) -> bool:
    return bool(torch.isfinite(a).all().item())


def _load_split() -> pd.DataFrame:
    required = {"ID", "fold", "binary_label", "SEX"}
    frame = pd.read_csv(SPLIT_TABLE, usecols=sorted(required), dtype={"ID": str})
    if set(frame.columns) != required or len(frame) != 500:
        raise RuntimeError("BLOCKED_SPLIT_SCHEMA")
    frame["fold"] = frame["fold"].astype(int)
    frame["binary_label"] = frame["binary_label"].astype(int)
    frame["SEX"] = frame["SEX"].astype(int)
    if set(frame["binary_label"]) != {0, 1} or frame["fold"].nunique() != 5:
        raise RuntimeError("BLOCKED_SPLIT_VALUES")
    return frame


def _select_batch(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_test = frame[frame.fold == 0].copy()
    development = frame[frame.fold != 0].copy()
    if len(development) != 400 or len(outer_test) != 100:
        raise RuntimeError("BLOCKED_OUTER_SPLIT")
    # Deterministic stratified 80/20 inner holdout using only the four allowed fields.
    rng = np.random.RandomState(12026)
    groups = list(development.groupby(["binary_label", "SEX"], sort=True))
    quotas = [len(group) * 0.20 for _, group in groups]
    val_counts = [int(np.floor(q)) for q in quotas]
    for index in np.argsort([q - np.floor(q) for q in quotas])[::-1][: 80 - sum(val_counts)]:
        val_counts[int(index)] += 1
    train_parts, val_parts = [], []
    for (_, group), n_val in zip(groups, val_counts):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        val_parts.append(indices[:n_val])
        train_parts.append(indices[n_val:])
    train = development.loc[np.concatenate(train_parts)].copy()
    validation = development.loc[np.concatenate(val_parts)].copy()
    if len(train) != 320 or len(validation) != 80:
        raise RuntimeError("BLOCKED_INNER_SPLIT_SIZE")
    canonical_path = INPUT_ROOT / "common/canonical_case_ids.txt"
    if not canonical_path.is_file():
        raise RuntimeError("BLOCKED_CANONICAL_ID_ORDER")
    canonical = [line.strip() for line in canonical_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rank = {case_id: index for index, case_id in enumerate(canonical)}
    if len(rank) != 500 or not set(frame.ID.astype(str)).issubset(rank):
        raise RuntimeError("BLOCKED_CANONICAL_ID_ORDER")
    train = train.assign(_canonical_rank=train.ID.astype(str).map(rank)).sort_values("_canonical_rank").drop(columns="_canonical_rank").reset_index(drop=True)
    for start in range(len(train) - 15):
        candidate = train.iloc[start : start + 16]
        if set(candidate.binary_label) == {0, 1}:
            return train, validation, candidate
    raise RuntimeError("BLOCKED_SMOKE_BATCH_CLASS_BALANCE")


def _model(in_channels: int) -> torch.nn.Module:
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if in_channels == 5:
        old = net.conv1
        new = torch.nn.Conv2d(5, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            mean_kernel = old.weight.mean(dim=1)
            new.weight[:, 3] = mean_kernel
            new.weight[:, 4] = mean_kernel
        net.conv1 = new
    net.fc = torch.nn.Linear(net.fc.in_features, 2)
    return net


def _normalise(x: torch.Tensor, condition: str) -> torch.Tensor:
    mean = MEAN_RGB.to(x.device)
    std = STD_RGB.to(x.device)
    if condition != "RGB":
        mean = torch.cat((mean, MEAN_MH.to(x.device)))
        std = torch.cat((std, STD_MH.to(x.device)))
    return (x - mean[None, :, None, None]) / std[None, :, None, None]


def _hash_selected(ids: list[str]) -> dict[str, str]:
    paths = [INPUT_ROOT / "CLASSIFIER_INPUT_ACCEPTANCE.json", INPUT_ROOT / "classifier_input_manifest.csv", INPUT_ROOT / "classifier_input_protocol_lock.json"]
    for case_id in ids:
        paths += [INPUT_ROOT / "common/valid_masks" / f"{case_id}.png"]
        paths += [INPUT_ROOT / name / "images" / f"{case_id}.npy" for name in ("rgb", "rgb_b1mh", "rgb_b2mh")]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths if p.is_file()}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n", encoding="utf-8")


def run_smoke(root: Path = ROOT) -> dict[str, Any]:
    global ROOT, INPUT_ROOT, REPORT_ROOT, SPLIT_TABLE
    ROOT = root.resolve()
    INPUT_ROOT = ROOT / "data/processed/SO_R2X3_ClassifierInputs_v1"
    REPORT_ROOT = ROOT / "reports/so_r2x3_classifier_smoke_test"
    SPLIT_TABLE = ROOT / "data/processed/P0_Physics_Audit_v1/splits_500/nyha_2class_sex_stratified_group_5fold.csv"
    acceptance = json.loads((INPUT_ROOT / "CLASSIFIER_INPUT_ACCEPTANCE.json").read_text(encoding="utf-8"))
    if acceptance.get("status") != "COMPLETE_CLASSIFIER_INPUT_PREPARATION" or acceptance.get("case_count") != 500:
        raise RuntimeError("BLOCKED_CLASSIFIER_INPUT_ACCEPTANCE")
    manifest = list(csv.DictReader((INPUT_ROOT / "classifier_input_manifest.csv").open(encoding="utf-8-sig", newline="")))
    if len(manifest) != 1500:
        raise RuntimeError("BLOCKED_CLASSIFIER_INPUT_MANIFEST")
    train, validation, batch_meta = _select_batch(_load_split())
    ids = batch_meta.ID.astype(str).tolist()
    before_hash = _hash_selected(ids)
    labels = torch.tensor(batch_meta.binary_label.to_numpy(), dtype=torch.long)
    arrays: dict[str, torch.Tensor] = {}
    for condition, directory in (("RGB", "rgb"), ("RGB_B1MH", "rgb_b1mh"), ("RGB_B2MH", "rgb_b2mh")):
        arrays[condition] = torch.from_numpy(np.stack([np.load(INPUT_ROOT / directory / "images" / f"{i}.npy", allow_pickle=False) for i in ids]).astype(np.float32, copy=False))
    shape_dtype = {k: {"shape": list(v.shape), "dtype": str(v.numpy().dtype), "finite": _finite(v)} for k, v in arrays.items()}
    identity = {"RGB_equal_RGB_B1MH_first3": bool(torch.equal(arrays["RGB"], arrays["RGB_B1MH"][:, :3])), "RGB_equal_RGB_B2MH_first3": bool(torch.equal(arrays["RGB"], arrays["RGB_B2MH"][:, :3]))}
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_CUDA_REQUIRED")
    models, audits = {}, {}
    class_counts = torch.bincount(labels, minlength=2).float()
    weights = class_counts.sum() / (2.0 * class_counts)
    for condition in CONDITIONS:
        x = _normalise(arrays[condition].to(device), condition)
        model = _model(x.shape[1]).to(device).float().train()
        logits = model(x)
        loss = F.cross_entropy(logits, labels.to(device), weight=weights.to(device), label_smoothing=0.05)
        model.zero_grad(set_to_none=True)
        loss.backward()
        grads = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
        conv_grad = grads["conv1.weight"]
        fc_grad = grads["fc.weight"]
        audits[condition] = {"input_shape": list(x.shape), "normalised_finite": _finite(x), "logits_shape": list(logits.shape), "logits_finite": _finite(logits), "loss": float(loss.detach().cpu()), "loss_finite_positive": bool(_finite(loss) and loss.item() > 0), "conv1_gradient_finite_nonzero": bool(_finite(conv_grad) and torch.any(conv_grad != 0)), "fc_gradient_finite_nonzero": bool(_finite(fc_grad) and torch.any(fc_grad != 0)), "all_trainable_gradients_finite": all(_finite(g) for g in grads.values() if g is not None), "input_channels": int(x.shape[1]), "optimizer_step_count": 0, "checkpoint_write_count": 0}
        models[condition] = model
    after_hash = _hash_selected(ids)
    hash_unchanged = before_hash == after_hash
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    batch_meta.loc[:, ["ID", "fold", "binary_label", "SEX"]].to_csv(REPORT_ROOT / "smoke_batch_manifest.csv", index=False, encoding="utf-8-sig")
    _write(REPORT_ROOT / "model_interface_audit.json", {"device": str(device), "dtype": "float32", "models": {k: {"architecture": "ResNet-18", "pretrained": True, "input_channels": v["input_channels"], "logits": 2, "class_order": ["Control", "Patient"], "full_finetune": True, "five_channel_conv1_init": "RGB ImageNet weights preserved; M/H each initialized from mean of original RGB conv1 channels" if v["input_channels"] == 5 else None} for k, v in audits.items()}, "normalization": {"rgb_mean": MEAN_RGB.tolist(), "rgb_std": STD_RGB.tolist(), "mh_mean": MEAN_MH.tolist(), "mh_std": STD_MH.tolist(), "horizontal_flip": False}, "loss": {"type": "weighted_cross_entropy", "label_smoothing_alpha": 0.05}})
    _write(REPORT_ROOT / "gradient_audit.json", audits)
    _write(REPORT_ROOT / "input_provenance_audit.json", {"p0_acceptance_status": acceptance.get("status"), "manifest_rows": len(manifest), "outer_fold": 0, "development_count": len(train) + len(validation), "inner_train_count": len(train), "inner_validation_count": len(validation), "inner_split_seed": 12026, "stratification": ["binary_label", "SEX"], "batch_case_ids": ids, "cross_condition_rgb_identity": identity, "b1_mh_source": "rgb_b1mh/*.npy", "b2_mh_source": "rgb_b2mh/*.npy", "p0_source_tensor_mask_unchanged": hash_unchanged, "outer_test_forward_count": 0})
    _write(REPORT_ROOT / "protected_asset_hash_audit.json", {"unchanged": hash_unchanged, "before": before_hash, "after": after_hash})
    _write(REPORT_ROOT / "test_results.txt", {"status": "PASS_SMOKE_TEST", "optimizer_step_count": 0, "checkpoint_write_count": 0})
    result = {"status": "PASS_SMOKE_TEST" if hash_unchanged and all(a["loss_finite_positive"] and a["all_trainable_gradients_finite"] for a in audits.values()) and all(identity.values()) else "FAIL_SMOKE_TEST", "classification_started": False, "full_training_started": False, "outer_test_forward_count": 0, "optimizer_step_count": 0, "checkpoint_write_count": 0, "formal_SO_R1_C_status": "FAIL", "official_SO_R2_authorization": False, "SO_R3_authorization": False, "next_stage_authorized": False, "case_count": 16, "input_shape_dtype_finite": {k: {"shape": v["input_shape"], "finite": v["normalised_finite"]} for k, v in audits.items()}, "models": audits}
    _write(ROOT / "data/processed/SO_R2X3_ClassifierSmokeTest_v1/S0_ACCEPTANCE.json", result)
    (REPORT_ROOT / "SO_R2X3_S0_Classifier_Smoke_Test_Report.md").write_text("# SO-R2-X3-S0 Classifier Smoke Test\n\nStatus: `" + result["status"] + "`\n\nOnly one 16-case inner-train batch was used. Outer-test images were not read; no optimizer step or checkpoint write occurred. P0 classifier inputs and their hashes remained unchanged.\n", encoding="utf-8")
    return result
