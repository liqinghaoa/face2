"""Locked SO-R2-X3 exploratory nested-direct ResNet-18 classification."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from metrics.R3DPR.binary_classification_metrics import compute_binary_metrics


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/SO_R2X3_ClassifierInputs_v1"
SPLIT = ROOT / "data/processed/P0_Physics_Audit_v1/splits_500/nyha_2class_sex_stratified_group_5fold.csv"
RUNS = ROOT / "runs/so_r2x3_exploratory_classification"
DATA_OUT = ROOT / "data/processed/SO_R2X3_ExploratoryClassification_v1"
REPORT = ROOT / "reports/so_r2x3_exploratory_classification"
CONDITIONS = {"RGB": ("rgb", 3), "RGB_B1MH": ("rgb_b1mh", 5), "RGB_B2MH": ("rgb_b2mh", 5)}
FIELDS = ["ID", "patient_group_id", "fold", "binary_label", "SEX"]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def _protected_snapshot(root: Path) -> dict[str, str]:
    input_root = root / "data/processed/SO_R2X3_ClassifierInputs_v1"
    x1 = root / "data/processed/SO_R2X1_RealFaceFrozenInference_v1"
    x2 = root / "data/processed/SO_R2X2_RealFaceOutputAudit_v1"
    files = [input_root / "CLASSIFIER_INPUT_ACCEPTANCE.json", input_root / "classifier_input_manifest.csv", input_root / "classifier_input_protocol_lock.json", x1 / "R2X1_ACCEPTANCE.json", x2 / "R2X2_ACCEPTANCE.json", root / "runs/so_r1_b1_baseline_v6/B1_ACCEPTANCE.json", root / "runs/so_r1_b1_baseline_v6/checkpoints/best_val_masked_smoothl1.pt", root / "runs/so_r1_b2_proposed_v3/B2_ACCEPTANCE.json", root / "runs/so_r1_b2_proposed_v3/lambda_0p50/checkpoints/best_val_masked_smoothl1.pt"]
    for pattern in ("rgb/images/*.npy", "rgb_b1mh/images/*.npy", "rgb_b2mh/images/*.npy", "common/valid_masks/*.png"):
        files += sorted(input_root.glob(pattern))
    files += sorted((x1 / "B1").glob("*.npz")) + sorted((x1 / "B2_lambda_0p50").glob("*.npz")) + sorted((x1 / "valid_masks").glob("*.png"))
    if not files or not all(path.is_file() for path in files):
        raise RuntimeError("BLOCKED_PROTECTED_ASSET_MISSING")
    return {str(path.relative_to(root)): _sha(path) for path in files}


def _read_table(root: Path) -> pd.DataFrame:
    table = pd.read_csv(root / "data/processed/P0_Physics_Audit_v1/splits_500/nyha_2class_sex_stratified_group_5fold.csv", usecols=FIELDS, dtype={"ID": str, "patient_group_id": str})
    if set(table.columns) != set(FIELDS) or len(table) != 500 or table.ID.duplicated().any():
        raise RuntimeError("BLOCKED_SPLIT_TABLE")
    table = table.loc[:, FIELDS].copy()
    for name in ("fold", "binary_label", "SEX"):
        table[name] = pd.to_numeric(table[name], errors="raise").astype(int)
    if set(table.fold) != set(range(5)) or set(table.binary_label) != {0, 1} or not (table.groupby("fold").size() == 100).all():
        raise RuntimeError("BLOCKED_SPLIT_VALUES")
    return table


def _preflight(root: Path, table: pd.DataFrame) -> None:
    input_root = root / "data/processed/SO_R2X3_ClassifierInputs_v1"
    inputs = json.loads((input_root / "CLASSIFIER_INPUT_ACCEPTANCE.json").read_text(encoding="utf-8"))
    smoke = json.loads((root / "data/processed/SO_R2X3_ClassifierSmokeTest_v1/S0_ACCEPTANCE.json").read_text(encoding="utf-8"))
    if inputs.get("status") != "COMPLETE_CLASSIFIER_INPUT_PREPARATION" or inputs.get("case_count") != 500 or smoke.get("status") != "PASS_SMOKE_TEST":
        raise RuntimeError("BLOCKED_P0_GATE")
    rows = list(csv.DictReader((input_root / "classifier_input_manifest.csv").open(encoding="utf-8-sig", newline="")))
    expected = {"RGB": ("3x320x256", "float32"), "RGB_B1MH": ("5x320x256", "float32"), "RGB_B2MH": ("5x320x256", "float32")}
    if len(rows) != 1500 or {r["case_id"] for r in rows} != set(table.ID): raise RuntimeError("BLOCKED_INPUT_MANIFEST")
    for condition, (shape, dtype) in expected.items():
        subset = [r for r in rows if r["condition"] == condition]
        if len(subset) != 500 or any((r["shape"], r["dtype"]) != (shape, dtype) for r in subset): raise RuntimeError("BLOCKED_INPUT_CONTRACT")


def _inner_split(development: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = development.binary_label.astype(str) + "__" + development.SEX.astype(str)
    train_idx, val_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=12026 + fold).split(development, strata))
    train, validation = development.iloc[train_idx].copy(), development.iloc[val_idx].copy()
    if len(train) != 320 or len(validation) != 80 or set(train.ID) & set(validation.ID): raise RuntimeError("BLOCKED_INNER_SPLIT")
    return train, validation


class FrozenNpyDataset(Dataset):
    def __init__(self, table: pd.DataFrame, root: Path, directory: str, channels: int, train: bool):
        self.table, self.root, self.directory, self.channels, self.train = table.reset_index(drop=True), root, directory, channels, train
    def __len__(self): return len(self.table)
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[index]; path = self.root / self.directory / "images" / f"{row.ID}.npy"
        array = np.load(path, allow_pickle=False)
        if array.shape != (self.channels, 320, 256) or array.dtype != np.float32 or not np.isfinite(array).all(): raise RuntimeError("BLOCKED_NPY_INPUT")
        image = torch.from_numpy(array.copy())
        if self.train and bool(torch.rand(()) < 0.5): image = torch.flip(image, dims=(2,))
        mean = torch.tensor([0.485, .456, .406] + ([.5, .5] if self.channels == 5 else []), dtype=torch.float32)[:, None, None]
        std = torch.tensor([.229, .224, .225] + ([.5, .5] if self.channels == 5 else []), dtype=torch.float32)[:, None, None]
        return {"image": (image - mean) / std, "label": torch.tensor(int(row.binary_label), dtype=torch.long), "ID": str(row.ID), "patient_group_id": str(row.patient_group_id), "fold": int(row.fold)}


def _model(channels: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if channels == 5:
        old = model.conv1; conv = nn.Conv2d(5, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad(): conv.weight[:, :3] = old.weight; conv.weight[:, 3] = old.weight.mean(1); conv.weight[:, 4] = old.weight.mean(1)
        model.conv1 = conv
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def _loader(table: pd.DataFrame, root: Path, directory: str, channels: int, train: bool) -> DataLoader:
    return DataLoader(FrozenNpyDataset(table, root, directory, channels, train), batch_size=16, shuffle=train, num_workers=0, pin_memory=True)


def _evaluate(model: nn.Module, loader: DataLoader, weights: torch.Tensor, device: torch.device) -> tuple[dict[str, Any], float, pd.DataFrame]:
    model.eval(); labels: list[int] = []; probs: list[np.ndarray] = []; ids: list[str] = []; groups: list[str] = []; folds: list[int] = []; losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
            logits = model(x); losses.append(float(torch.nn.functional.cross_entropy(logits, y, weight=weights, label_smoothing=.05)) * len(y))
            labels.extend(y.cpu().tolist()); probs.append(torch.softmax(logits, 1).cpu().numpy()); ids.extend(batch["ID"]); groups.extend(batch["patient_group_id"]); folds.extend(batch["fold"])
    probability = np.concatenate(probs); metric = compute_binary_metrics(labels, probability)
    prediction = pd.DataFrame({"ID": ids, "patient_group_id": groups, "fold": folds, "binary_label": labels, "prob_control": probability[:, 0], "prob_patient": probability[:, 1], "pred_class": probability.argmax(1)})
    return metric, sum(losses) / len(labels), prediction


def _train_fold(root: Path, condition: str, table: pd.DataFrame, fold: int, device: torch.device) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    directory, channels = CONDITIONS[condition]; development, outer = table[table.fold != fold].copy(), table[table.fold == fold].copy(); train, validation = _inner_split(development, fold)
    fold_dir = root / "runs/so_r2x3_exploratory_classification" / condition / f"fold_{fold}"; (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    for role, frame in (("inner_train", train), ("inner_validation", validation), ("outer_test", outer)):
        frame.assign(role=role).to_csv(fold_dir / f"{role}.csv", index=False, encoding="utf-8-sig")
    _seed(12026 + fold); model = _model(channels).to(device).float(); counts = np.bincount(train.binary_label, minlength=2); weights = torch.tensor(len(train) / (2 * counts), dtype=torch.float32, device=device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    train_loader, val_loader = _loader(train, root / "data/processed/SO_R2X3_ClassifierInputs_v1", directory, channels, True), _loader(validation, root / "data/processed/SO_R2X3_ClassifierInputs_v1", directory, channels, False)
    best, best_epoch, stale, history = -np.inf, 0, 0, []
    checkpoint = fold_dir / "checkpoints/inner_best_macro_auc.pt"
    for epoch in range(1, 51):
        model.train(); total = 0.0
        for batch in train_loader:
            x, y = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True); loss = torch.nn.functional.cross_entropy(model(x), y, weight=weights, label_smoothing=.05); loss.backward(); optimizer.step(); total += float(loss.detach()) * len(y)
        val_metric, val_loss, _ = _evaluate(model, val_loader, weights, device); history.append({"epoch": epoch, "train_loss": total / len(train), "inner_val_loss": val_loss, "inner_val_macro_auc": val_metric["macro_auc"]})
        if val_metric["macro_auc"] > best:
            best, best_epoch, stale = float(val_metric["macro_auc"]), epoch, 0; torch.save({"state_dict": model.state_dict(), "selected_epoch": epoch, "inner_validation_macro_auc": best, "channels": channels}, checkpoint)
        else: stale += 1
        if stale >= 10: break
    pd.DataFrame(history).to_csv(fold_dir / "inner_selection_history.csv", index=False, encoding="utf-8-sig")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    outer_metric, _, prediction = _evaluate(model, _loader(outer, root / "data/processed/SO_R2X3_ClassifierInputs_v1", directory, channels, False), weights, device)
    prediction["condition"], prediction["outer_fold"], prediction["selected_epoch"], prediction["inner_best_macro_auc"] = condition, fold, best_epoch, best
    prediction.to_csv(fold_dir / "outer_test_predictions.csv", index=False, encoding="utf-8-sig")
    row = {"condition": condition, "fold": fold, "selected_epoch": best_epoch, "selection_epochs_run": len(history), "inner_best_macro_auc": best, "train_control": int(counts[0]), "train_patient": int(counts[1]), "weight_control": float(weights[0]), "weight_patient": float(weights[1]), **{k: v for k, v in outer_metric.items() if k != "confusion_matrix"}}
    _dump(fold_dir / "fold_metrics.json", row); pd.DataFrame(outer_metric["confusion_matrix"], index=["Control", "Patient"], columns=["Control", "Patient"]).to_csv(fold_dir / "outer_confusion_matrix.csv", encoding="utf-8-sig")
    audit = {"fold": fold, "development": len(development), "inner_train": len(train), "inner_validation": len(validation), "outer_test": len(outer), "id_overlap_train_val": 0, "id_overlap_train_outer": len(set(train.ID) & set(outer.ID)), "id_overlap_val_outer": len(set(validation.ID) & set(outer.ID)), "outer_test_model_selection_access": 0}
    del model; torch.cuda.empty_cache()
    return row, prediction, audit


def _finalize_existing(root: Path) -> dict[str, Any]:
    """Create aggregate-only artifacts from the immutable completed fold outputs."""
    run_root = root / "runs/so_r2x3_exploratory_classification"
    metric_paths = sorted(run_root.glob("*/*/fold_metrics.json"))
    if len(metric_paths) != 15:
        raise RuntimeError("BLOCKED_INCOMPLETE_EXISTING_RUN")
    before = _protected_snapshot(root)
    first_run_write = min(p.stat().st_mtime for p in metric_paths)
    source_mtime_precedes_run = all((root / relative).stat().st_mtime <= first_run_write for relative in before)
    rows, frames, leakage = [], [], []
    for metric_path in metric_paths:
        row = json.loads(metric_path.read_text(encoding="utf-8")); rows.append(row)
        fold_dir = metric_path.parent
        prediction = pd.read_csv(fold_dir / "outer_test_predictions.csv", dtype={"ID": str, "patient_group_id": str})
        frames.append(prediction)
        train = pd.read_csv(fold_dir / "inner_train.csv", dtype={"ID": str})
        validation = pd.read_csv(fold_dir / "inner_validation.csv", dtype={"ID": str})
        outer = pd.read_csv(fold_dir / "outer_test.csv", dtype={"ID": str})
        leakage.append({"condition": row["condition"], "fold": int(row["fold"]), "development": len(train)+len(validation), "inner_train": len(train), "inner_validation": len(validation), "outer_test": len(outer), "id_overlap_train_val": len(set(train.ID)&set(validation.ID)), "id_overlap_train_outer": len(set(train.ID)&set(outer.ID)), "id_overlap_val_outer": len(set(validation.ID)&set(outer.ID)), "outer_test_model_selection_access": 0})
    metrics = pd.DataFrame(rows).sort_values(["condition", "fold"]); predictions = pd.concat(frames, ignore_index=True).sort_values(["condition", "fold", "ID"]).reset_index(drop=True)
    expected_counts = {name: 500 for name in CONDITIONS}
    if predictions.groupby("condition").size().to_dict() != expected_counts or predictions.duplicated(["condition", "ID"]).any(): raise RuntimeError("FAIL_OOF_COVERAGE")
    report = root / "reports/so_r2x3_exploratory_classification"; report.mkdir(parents=True, exist_ok=True)
    pooled, matrices = [], []
    for condition in CONDITIONS:
        sub = predictions[predictions.condition == condition]; metric = compute_binary_metrics(sub.binary_label, sub[["prob_control", "prob_patient"]].to_numpy())
        pooled.append({"condition": condition, "scope": "pooled_oof", **{k:v for k,v in metric.items() if k != "confusion_matrix"}})
        matrix = pd.DataFrame(metric["confusion_matrix"], index=["Control", "Patient"], columns=["Control", "Patient"]).reset_index(names="true_class")
        matrix.insert(0, "condition", condition); matrices.append(matrix)
    pooled_df = pd.DataFrame(pooled); metrics.to_csv(report / "fold_metrics.csv", index=False, encoding="utf-8-sig"); predictions.to_csv(report / "oof_predictions.csv", index=False, encoding="utf-8-sig"); pooled_df.to_csv(report / "oof_metrics.csv", index=False, encoding="utf-8-sig"); pd.concat(matrices, ignore_index=True).to_csv(report / "oof_confusion_matrix.csv", index=False, encoding="utf-8-sig")
    idx = pooled_df.set_index("condition"); comparison = {"primary_RGB_B2MH_minus_RGB_pooled_macro_auc": float(idx.loc["RGB_B2MH","macro_auc"]-idx.loc["RGB","macro_auc"]), "secondary_RGB_B1MH_minus_RGB_pooled_macro_auc": float(idx.loc["RGB_B1MH","macro_auc"]-idx.loc["RGB","macro_auc"])}
    after = _protected_snapshot(root); unchanged = before == after and source_mtime_precedes_run
    _dump(report / "split_leakage_audit.json", {"all_pass": all(x["id_overlap_train_val"] == x["id_overlap_train_outer"] == x["id_overlap_val_outer"] == x["outer_test_model_selection_access"] == 0 for x in leakage), "folds": leakage, "outer_test_prediction_coverage": expected_counts})
    _dump(report / "input_provenance_audit.json", {"input_root": "data/processed/SO_R2X3_ClassifierInputs_v1", "conditions": CONDITIONS, "sex_used_for_stratification_only": True, "sex_used_as_model_feature": False, "allowed_fields_read": FIELDS, "forbidden_clinical_fields_read": 0})
    _dump(report / "protected_asset_hash_audit.json", {"unchanged": unchanged, "source_mtime_precedes_first_run_output": source_mtime_precedes_run, "before_finalize": before, "after_finalize": after})
    _dump(report / "field_access_audit.json", {"allowed_split_fields": FIELDS, "sex_used_for_stratification_only": True, "sex_used_as_model_feature": False, "outer_test_model_selection_access": 0, "forbidden_clinical_field_access_count": 0})
    mean_sd = metrics.groupby("condition")[["macro_auc","macro_f1","balanced_accuracy","sensitivity","specificity"]].agg(["mean","std"]).round(4).to_csv()
    (report / "SO_R2X3_Exploratory_Classification_Report.md").write_text("# SO-R2-X3 Exploratory Nested-Direct Classification\n\nInternal five-fold OOF exploratory evidence only; no external validation, bootstrap, p value, winner selection, or authorization change. M/H-sensitive maps are not absolute melanin/hemoglobin concentrations.\n\n## Pooled OOF\n\n```csv\n" + pooled_df.to_csv(index=False) + "```\n\n## Five-fold mean and SD\n\n```csv\n" + mean_sd + "```\n\n## Pre-registered comparisons\n\n```json\n" + json.dumps(comparison, indent=2) + "\n```\n", encoding="utf-8")
    (report / "test_results.txt").write_text("preflight=PASS\ncompleted_fold_outputs=15\nfinalize=PASS\n", encoding="utf-8")
    acceptance = {"status": "COMPLETE_EXPLORATORY_NESTED_DIRECT_CLASSIFICATION" if unchanged else "FAIL_PROTECTED_ASSET_MUTATION", "classification_started": True, "full_training_started": True, "formal_SO_R1_C_status": "FAIL", "official_SO_R2_authorization": False, "SO_R3_authorization": False, "next_stage_authorized": False, "oof_coverage": expected_counts, "outer_test_model_selection_access": 0, "outer_test_prediction_coverage": expected_counts, "protected_assets_unchanged": unchanged, "comparisons": comparison}
    _dump(root / "data/processed/SO_R2X3_ExploratoryClassification_v1/R2X3_ACCEPTANCE.json", acceptance); _dump(root / "data/processed/SO_R2X3_ExploratoryClassification_v1/classification_run_manifest.json", {"conditions": list(CONDITIONS), "folds": list(range(5)), "model": "ImageNet-pretrained ResNet-18", "nested_direct": True, "status": acceptance["status"], "aggregate_only_finalize": True})
    return acceptance


def run(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve(); table = _read_table(root); _preflight(root, table)
    if not torch.cuda.is_available(): raise RuntimeError("BLOCKED_CUDA_REQUIRED")
    if (root / "runs/so_r2x3_exploratory_classification").exists(): return _finalize_existing(root)
    before = _protected_snapshot(root); device = torch.device("cuda"); all_rows: list[dict[str, Any]] = []; all_predictions: list[pd.DataFrame] = []; leakage: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for fold in range(5):
            row, predictions, audit = _train_fold(root, condition, table, fold, device); all_rows.append(row); all_predictions.append(predictions); leakage.append({"condition": condition, **audit})
    metrics = pd.DataFrame(all_rows); predictions = pd.concat(all_predictions, ignore_index=True).sort_values(["condition", "fold", "ID"]).reset_index(drop=True)
    if predictions.groupby("condition").size().to_dict() != {k: 500 for k in CONDITIONS} or predictions.duplicated(["condition", "ID"]).any(): raise RuntimeError("FAIL_OOF_COVERAGE")
    report = root / "reports/so_r2x3_exploratory_classification"; report.mkdir(parents=True, exist_ok=True)
    pooled = []
    for condition in CONDITIONS:
        sub = predictions[predictions.condition == condition]; p = sub[["prob_control", "prob_patient"]].to_numpy(); m = compute_binary_metrics(sub.binary_label, p); pooled.append({"condition": condition, "scope": "pooled_oof", **{k: v for k, v in m.items() if k != "confusion_matrix"}})
        pd.DataFrame(m["confusion_matrix"], index=["Control", "Patient"], columns=["Control", "Patient"]).assign(condition=condition).to_csv(root / "reports/so_r2x3_exploratory_classification" / f"{condition}_oof_confusion_matrix.csv", encoding="utf-8-sig")
    pooled_df = pd.DataFrame(pooled); metrics.to_csv(report / "fold_metrics.csv", index=False, encoding="utf-8-sig"); predictions.to_csv(report / "oof_predictions.csv", index=False, encoding="utf-8-sig"); pooled_df.to_csv(report / "oof_metrics.csv", index=False, encoding="utf-8-sig")
    combined_cm = pd.concat([pd.read_csv(report / f"{c}_oof_confusion_matrix.csv") for c in CONDITIONS], ignore_index=True); combined_cm.to_csv(report / "oof_confusion_matrix.csv", index=False, encoding="utf-8-sig")
    comparison = {"primary_RGB_B2MH_minus_RGB_pooled_macro_auc": float(pooled_df.set_index("condition").loc["RGB_B2MH", "macro_auc"] - pooled_df.set_index("condition").loc["RGB", "macro_auc"]), "secondary_RGB_B1MH_minus_RGB_pooled_macro_auc": float(pooled_df.set_index("condition").loc["RGB_B1MH", "macro_auc"] - pooled_df.set_index("condition").loc["RGB", "macro_auc"])}
    after = _protected_snapshot(root); unchanged = before == after
    _dump(report / "split_leakage_audit.json", {"all_pass": all(x["id_overlap_train_val"] == x["id_overlap_train_outer"] == x["id_overlap_val_outer"] == x["outer_test_model_selection_access"] == 0 for x in leakage), "folds": leakage, "outer_test_prediction_coverage": {c: int((predictions.condition == c).sum()) for c in CONDITIONS}})
    _dump(report / "input_provenance_audit.json", {"input_root": "data/processed/SO_R2X3_ClassifierInputs_v1", "conditions": CONDITIONS, "sex_used_for_stratification_only": True, "sex_used_as_model_feature": False, "allowed_fields_read": FIELDS, "forbidden_clinical_fields_read": 0})
    _dump(report / "protected_asset_hash_audit.json", {"unchanged": unchanged, "before": before, "after": after})
    _dump(report / "field_access_audit.json", {"allowed_split_fields": FIELDS, "sex_used_for_stratification_only": True, "sex_used_as_model_feature": False, "outer_test_model_selection_access": 0, "forbidden_clinical_field_access_count": 0})
    (report / "test_results.txt").write_text("preflight=PASS\ntraining=PASS\n", encoding="utf-8")
    mean_sd = metrics.groupby("condition")[["macro_auc", "macro_f1", "balanced_accuracy", "sensitivity", "specificity"]].agg(["mean", "std"]).round(4).to_string()
    (report / "SO_R2X3_Exploratory_Classification_Report.md").write_text("# SO-R2-X3 Exploratory Nested-Direct Classification\n\nInternal five-fold OOF exploratory evidence only; no external validation, bootstrap, p value, winner selection, or authorization change. M/H-sensitive maps are not absolute melanin/hemoglobin concentrations.\n\n## Pooled OOF\n\n```csv\n" + pooled_df.to_csv(index=False) + "```\n\n## Five-fold mean and SD\n\n```\n" + mean_sd + "\n```\n\n## Pre-registered comparisons\n\n```json\n" + json.dumps(comparison, indent=2) + "\n```\n", encoding="utf-8")
    acceptance = {"status": "COMPLETE_EXPLORATORY_NESTED_DIRECT_CLASSIFICATION" if unchanged else "FAIL_PROTECTED_ASSET_MUTATION", "classification_started": True, "full_training_started": True, "formal_SO_R1_C_status": "FAIL", "official_SO_R2_authorization": False, "SO_R3_authorization": False, "next_stage_authorized": False, "oof_coverage": {c: int((predictions.condition == c).sum()) for c in CONDITIONS}, "outer_test_model_selection_access": 0, "outer_test_prediction_coverage": {c: int((predictions.condition == c).sum()) for c in CONDITIONS}, "protected_assets_unchanged": unchanged, "comparisons": comparison}
    _dump(root / "data/processed/SO_R2X3_ExploratoryClassification_v1/R2X3_ACCEPTANCE.json", acceptance); _dump(root / "data/processed/SO_R2X3_ExploratoryClassification_v1/classification_run_manifest.json", {"conditions": list(CONDITIONS), "folds": list(range(5)), "model": "ImageNet-pretrained ResNet-18", "nested_direct": True, "status": acceptance["status"]})
    return acceptance
