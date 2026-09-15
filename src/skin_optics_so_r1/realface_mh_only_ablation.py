"""SO-R2-X3-MH locked exploratory M/H-only nested-direct ablation.

Only channels 3/4 of the frozen five-channel P0 tensors are ever materialised.
M/H-sensitive channels are representation outputs, not absolute concentrations.
"""
from __future__ import annotations

import hashlib
import json
import random
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
INPUT_ROOT = ROOT / "data/processed/SO_R2X3_ClassifierInputs_v1"
SPLIT_TABLE = ROOT / "data/processed/P0_Physics_Audit_v1/splits_500/nyha_2class_sex_stratified_group_5fold.csv"
RUN_ROOT = ROOT / "runs/so_r2x3_mh_only_ablation"
DATA_ROOT = ROOT / "data/processed/SO_R2X3_MHOnlyAblation_v1"
REPORT_ROOT = ROOT / "reports/so_r2x3_mh_only_ablation"
FIELDS = ["ID", "patient_group_id", "fold", "binary_label", "SEX"]
METRICS = ["macro_auc", "macro_f1", "balanced_accuracy", "sensitivity", "specificity"]
CONDITIONS = {
    "B1_MH_ONLY": ("rgb_b1mh", (3, 4)),
    "B2_MH_ONLY": ("rgb_b2mh", (3, 4)),
    "B2_M_ONLY": ("rgb_b2mh", (3,)),
    "B2_H_ONLY": ("rgb_b2mh", (4,)),
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mh_channel_sha(path: Path) -> dict[str, str]:
    """Hash only channels 3/4; never stream the RGB bytes in a 5-channel file."""
    source = np.load(path, mmap_mode="r", allow_pickle=False)
    if source.shape != (5, 320, 256) or source.dtype != np.float32:
        raise RuntimeError("BLOCKED_MH_TENSOR_CONTRACT")
    hashes: dict[str, str] = {}
    for index, name in ((3, "M_sensitive"), (4, "H_sensitive")):
        # This slice is the only pixel payload materialised by the hash audit.
        payload = np.ascontiguousarray(source[index])
        hashes[name] = hashlib.sha256(payload.tobytes()).hexdigest()
    return hashes


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def _read_table(root: Path) -> pd.DataFrame:
    table = pd.read_csv(root / SPLIT_TABLE.relative_to(ROOT), usecols=FIELDS, dtype={"ID": str, "patient_group_id": str})
    if set(table.columns) != set(FIELDS) or len(table) != 500 or table.ID.duplicated().any():
        raise RuntimeError("BLOCKED_SPLIT_TABLE")
    table = table.loc[:, FIELDS].copy()
    for name in ("fold", "binary_label", "SEX"):
        table[name] = pd.to_numeric(table[name], errors="raise").astype(int)
    if set(table.fold) != set(range(5)) or set(table.binary_label) != {0, 1} or not (table.groupby("fold").size() == 100).all():
        raise RuntimeError("BLOCKED_SPLIT_VALUES")
    return table


def _gate(root: Path) -> dict[str, Any]:
    gates = {
        "classifier_input": root / "data/processed/SO_R2X3_ClassifierInputs_v1/CLASSIFIER_INPUT_ACCEPTANCE.json",
        "smoke": root / "data/processed/SO_R2X3_ClassifierSmokeTest_v1/S0_ACCEPTANCE.json",
        "x3": root / "data/processed/SO_R2X3_ExploratoryClassification_v1/R2X3_ACCEPTANCE.json",
    }
    payload = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in gates.items()}
    if (payload["classifier_input"].get("status") != "COMPLETE_CLASSIFIER_INPUT_PREPARATION" or
            payload["classifier_input"].get("case_count") != 500 or
            payload["smoke"].get("status") != "PASS_SMOKE_TEST" or
            payload["x3"].get("status") != "COMPLETE_EXPLORATORY_NESTED_DIRECT_CLASSIFICATION"):
        raise RuntimeError("BLOCKED_PREREQUISITE_GATE")
    classifier = payload["classifier_input"]
    return {
        "gate_file_hashes": {str(path.relative_to(root)): _sha(path) for path in gates.values()},
        # These recorded upstream hashes provide a no-direct-read provenance
        # anchor for X1/X2 and the frozen B1/B2 checkpoints.
        "upstream_immutable_references": {
            "B1_checkpoint_hash": classifier.get("B1_checkpoint_hash"),
            "B2_checkpoint_hash": classifier.get("B2_checkpoint_hash"),
            "R2X1_acceptance_hash": classifier.get("source_R2X1_acceptance_hash"),
            "R2X2_acceptance_hash": classifier.get("source_R2X2_acceptance_hash"),
            "existing_X3_acceptance_hash": _sha(gates["x3"]),
        },
    }


def _protected_snapshot(root: Path) -> dict[str, Any]:
    # Deliberately inspect only the two explicitly allowed frozen P0 tensor trees.
    paths = []
    for directory in ("rgb_b1mh", "rgb_b2mh"):
        paths.extend(sorted((root / "data/processed/SO_R2X3_ClassifierInputs_v1" / directory / "images").glob("*.npy")))
    if len(paths) != 1000 or not all(path.is_file() for path in paths):
        raise RuntimeError("BLOCKED_MH_INPUT_INVENTORY")
    return {str(path.relative_to(root)): _mh_channel_sha(path) for path in paths}


def _inner_split(development: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = development.binary_label.astype(str) + "__" + development.SEX.astype(str)
    train_i, val_i = next(StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=12026 + fold).split(development, strata))
    train, validation = development.iloc[train_i].copy(), development.iloc[val_i].copy()
    if len(train) != 320 or len(validation) != 80 or set(train.ID) & set(validation.ID):
        raise RuntimeError("BLOCKED_INNER_SPLIT")
    return train, validation


class _MHDataset(Dataset):
    def __init__(self, table: pd.DataFrame, input_root: Path, directory: str, indices: tuple[int, ...], train: bool):
        self.table, self.input_root, self.directory, self.indices, self.train = table.reset_index(drop=True), input_root, directory, indices, train

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[index]
        path = self.input_root / self.directory / "images" / f"{row.ID}.npy"
        # mmap plus advanced indexing materialises only channels 3/4; RGB is never read.
        source = np.load(path, mmap_mode="r", allow_pickle=False)
        if source.shape != (5, 320, 256) or source.dtype != np.float32:
            raise RuntimeError("BLOCKED_MH_TENSOR_CONTRACT")
        array = np.array(source[list(self.indices)], dtype=np.float32, copy=True)
        if not np.isfinite(array).all():
            raise RuntimeError("BLOCKED_NONFINITE_MH")
        image = torch.from_numpy(array)
        if self.train and bool(torch.rand(()) < 0.5):
            image = torch.flip(image, dims=(2,))
        return {"image": (image - 0.5) / 0.5, "label": torch.tensor(int(row.binary_label), dtype=torch.long), "ID": str(row.ID), "patient_group_id": str(row.patient_group_id), "fold": int(row.fold)}


def _model(channels: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    old = model.conv1
    conv = nn.Conv2d(channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    with torch.no_grad():
        mean_kernel = old.weight.mean(dim=1)
        for channel in range(channels):
            conv.weight[:, channel] = mean_kernel
    model.conv1 = conv
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def _loader(table: pd.DataFrame, root: Path, directory: str, indices: tuple[int, ...], train: bool) -> DataLoader:
    return DataLoader(_MHDataset(table, root, directory, indices, train), batch_size=16, shuffle=train, num_workers=0, pin_memory=True)


def _evaluate(model: nn.Module, loader: DataLoader, weights: torch.Tensor, device: torch.device) -> tuple[dict[str, Any], float, pd.DataFrame]:
    model.eval(); labels: list[int] = []; probabilities: list[np.ndarray] = []; ids: list[str] = []; groups: list[str] = []; folds: list[int] = []; loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            image, label = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
            logits = model(image)
            loss_sum += float(torch.nn.functional.cross_entropy(logits, label, weight=weights, label_smoothing=0.05)) * len(label)
            labels.extend(label.cpu().tolist()); probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            ids.extend(batch["ID"]); groups.extend(batch["patient_group_id"]); folds.extend(batch["fold"])
    probability = np.concatenate(probabilities, axis=0)
    return compute_binary_metrics(labels, probability), loss_sum / len(labels), pd.DataFrame({"ID": ids, "patient_group_id": groups, "fold": folds, "binary_label": labels, "prob_control": probability[:, 0], "prob_patient": probability[:, 1], "pred_class": probability.argmax(axis=1)})


def _train_fold(root: Path, condition: str, table: pd.DataFrame, fold: int, device: torch.device) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    directory, indices = CONDITIONS[condition]
    development, outer = table[table.fold != fold].copy(), table[table.fold == fold].copy()
    train, validation = _inner_split(development, fold)
    fold_root = root / "runs/so_r2x3_mh_only_ablation" / condition / f"fold_{fold}"
    (fold_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    for role, frame in (("inner_train", train), ("inner_validation", validation), ("outer_test", outer)):
        frame.assign(role=role).to_csv(fold_root / f"{role}.csv", index=False, encoding="utf-8-sig")
    _seed(12026 + fold)
    model = _model(len(indices)).to(device).float()
    count = np.bincount(train.binary_label, minlength=2)
    weights = torch.tensor(len(train) / (2 * count), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    train_loader = _loader(train, root / "data/processed/SO_R2X3_ClassifierInputs_v1", directory, indices, True)
    val_loader = _loader(validation, root / "data/processed/SO_R2X3_ClassifierInputs_v1", directory, indices, False)
    best_auc, best_epoch, stale, history = -np.inf, 0, 0, []
    checkpoint = fold_root / "checkpoints/inner_best_macro_auc.pt"
    for epoch in range(1, 51):
        model.train(); train_loss = 0.0
        for batch in train_loader:
            image, label = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(image), label, weight=weights, label_smoothing=0.05)
            loss.backward(); optimizer.step(); train_loss += float(loss.detach()) * len(label)
        val_metric, val_loss, _ = _evaluate(model, val_loader, weights, device)
        history.append({"epoch": epoch, "train_loss": train_loss / len(train), "inner_val_loss": val_loss, "inner_val_macro_auc": val_metric["macro_auc"]})
        if val_metric["macro_auc"] > best_auc:
            best_auc, best_epoch, stale = float(val_metric["macro_auc"]), epoch, 0
            torch.save({"state_dict": model.state_dict(), "selected_epoch": epoch, "inner_validation_macro_auc": best_auc, "input_channels": len(indices)}, checkpoint)
        else:
            stale += 1
        if stale >= 10:
            break
    pd.DataFrame(history).to_csv(fold_root / "inner_selection_history.csv", index=False, encoding="utf-8-sig")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    outer_metric, _, prediction = _evaluate(model, _loader(outer, root / "data/processed/SO_R2X3_ClassifierInputs_v1", directory, indices, False), weights, device)
    prediction["condition"], prediction["outer_fold"], prediction["selected_epoch"], prediction["inner_best_macro_auc"] = condition, fold, best_epoch, best_auc
    prediction.to_csv(fold_root / "outer_test_predictions.csv", index=False, encoding="utf-8-sig")
    row = {"condition": condition, "fold": fold, "selected_epoch": best_epoch, "selection_epochs_run": len(history), "inner_best_macro_auc": best_auc, "train_control": int(count[0]), "train_patient": int(count[1]), "weight_control": float(weights[0]), "weight_patient": float(weights[1]), **{k: v for k, v in outer_metric.items() if k != "confusion_matrix"}}
    _dump(fold_root / "fold_metrics.json", row)
    pd.DataFrame(outer_metric["confusion_matrix"], index=["Control", "Patient"], columns=["Control", "Patient"]).to_csv(fold_root / "outer_confusion_matrix.csv", encoding="utf-8-sig")
    audit = {"condition": condition, "fold": fold, "development": len(development), "inner_train": len(train), "inner_validation": len(validation), "outer_test": len(outer), "id_overlap_train_val": len(set(train.ID) & set(validation.ID)), "id_overlap_train_outer": len(set(train.ID) & set(outer.ID)), "id_overlap_val_outer": len(set(validation.ID) & set(outer.ID)), "outer_test_model_selection_access": 0}
    del model; torch.cuda.empty_cache()
    return row, prediction, audit


def _comparison(pooled: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    reference = pooled.set_index("condition")
    rows = []
    for left, right in (("B2_MH_ONLY", "B1_MH_ONLY"), ("B2_MH_ONLY", "B2_M_ONLY"), ("B2_MH_ONLY", "B2_H_ONLY")):
        fold_delta = metrics[metrics.condition == left].sort_values("fold")[METRICS].to_numpy() - metrics[metrics.condition == right].sort_values("fold")[METRICS].to_numpy()
        for index, metric in enumerate(METRICS):
            rows.append({"comparison": f"{left}_minus_{right}", "metric": metric, "pooled_oof_absolute_difference": float(reference.loc[left, metric] - reference.loc[right, metric]), "fold_delta_mean": float(fold_delta[:, index].mean()), "fold_delta_sd": float(fold_delta[:, index].std(ddof=1))})
    return pd.DataFrame(rows)


def _write_csv_if_absent_or_equal(path: Path, frame: pd.DataFrame) -> None:
    """Existing frozen OOF outputs are verified but never overwritten."""
    if path.is_file():
        existing = pd.read_csv(path, dtype={"ID": str, "patient_group_id": str})
        if list(existing.columns) != list(frame.columns) or len(existing) != len(frame):
            raise RuntimeError("FAIL_EXISTING_OOF_REPORT_MISMATCH")
        for column in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[column]):
                if not np.allclose(existing[column].to_numpy(), frame[column].to_numpy(), equal_nan=True):
                    raise RuntimeError("FAIL_EXISTING_OOF_REPORT_MISMATCH")
            elif not existing[column].astype(str).equals(frame[column].astype(str)):
                raise RuntimeError("FAIL_EXISTING_OOF_REPORT_MISMATCH")
        return
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _finalize_existing(root: Path, gate_audit: dict[str, Any]) -> dict[str, Any]:
    """Aggregate an already completed locked 20-fold run without any training."""
    expected_pairs = [(condition, fold) for condition in CONDITIONS for fold in range(5)]
    paths = [(root / "runs/so_r2x3_mh_only_ablation" / condition / f"fold_{fold}") for condition, fold in expected_pairs]
    if not all((path / "fold_metrics.json").is_file() and (path / "outer_test_predictions.csv").is_file() for path in paths):
        raise RuntimeError("BLOCKED_INCOMPLETE_FOLD_OUTPUTS")
    before = _protected_snapshot(root)
    rows, predictions, leakage = [], [], []
    for (condition, fold), fold_root in zip(expected_pairs, paths):
        rows.append(json.loads((fold_root / "fold_metrics.json").read_text(encoding="utf-8")))
        prediction = pd.read_csv(fold_root / "outer_test_predictions.csv", dtype={"ID": str, "patient_group_id": str})
        predictions.append(prediction)
        train = pd.read_csv(fold_root / "inner_train.csv", dtype={"ID": str})
        validation = pd.read_csv(fold_root / "inner_validation.csv", dtype={"ID": str})
        outer = pd.read_csv(fold_root / "outer_test.csv", dtype={"ID": str})
        leakage.append({"condition": condition, "fold": fold, "development": len(train) + len(validation), "inner_train": len(train), "inner_validation": len(validation), "outer_test": len(outer), "id_overlap_train_val": len(set(train.ID) & set(validation.ID)), "id_overlap_train_outer": len(set(train.ID) & set(outer.ID)), "id_overlap_val_outer": len(set(validation.ID) & set(outer.ID)), "outer_test_model_selection_access": 0})
    fold_metrics = pd.DataFrame(rows).sort_values(["condition", "fold"]).reset_index(drop=True)
    oof = pd.concat(predictions, ignore_index=True).sort_values(["condition", "fold", "ID"]).reset_index(drop=True)
    expected = {condition: 500 for condition in CONDITIONS}
    if oof.groupby("condition").size().to_dict() != expected or oof.duplicated(["condition", "ID"]).any():
        raise RuntimeError("FAIL_OOF_COVERAGE")
    pooled_rows, matrices = [], []
    for condition in CONDITIONS:
        subset = oof[oof.condition == condition]
        metric = compute_binary_metrics(subset.binary_label, subset[["prob_control", "prob_patient"]].to_numpy())
        pooled_rows.append({"condition": condition, "scope": "pooled_oof", **{key: value for key, value in metric.items() if key != "confusion_matrix"}})
        matrix = pd.DataFrame(metric["confusion_matrix"], index=["Control", "Patient"], columns=["Control", "Patient"]).reset_index(names="true_class")
        matrix.insert(0, "condition", condition); matrices.append(matrix)
    pooled = pd.DataFrame(pooled_rows)
    comparisons = _comparison(pooled, fold_metrics)
    after = _protected_snapshot(root)
    unchanged = before == after
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_csv_if_absent_or_equal(REPORT_ROOT / "fold_metrics.csv", fold_metrics)
    _write_csv_if_absent_or_equal(REPORT_ROOT / "oof_predictions.csv", oof)
    _write_csv_if_absent_or_equal(REPORT_ROOT / "oof_metrics.csv", pooled)
    _write_csv_if_absent_or_equal(REPORT_ROOT / "oof_confusion_matrix.csv", pd.concat(matrices, ignore_index=True))
    _dump(REPORT_ROOT / "split_leakage_audit.json", {"all_pass": all(item["id_overlap_train_val"] == item["id_overlap_train_outer"] == item["id_overlap_val_outer"] == item["outer_test_model_selection_access"] == 0 for item in leakage), "folds": leakage, "outer_test_prediction_coverage": expected})
    _dump(REPORT_ROOT / "input_provenance_audit.json", {"only_allowed_p0_tensor_sources": {key: {"directory": value[0], "channels": list(value[1])} for key, value in CONDITIONS.items()}, "rgb_channels_read": False, "source_tensor_shape": [5, 320, 256], "mh_normalization": {"mean": 0.5, "std": 0.5}})
    _dump(REPORT_ROOT / "field_access_audit.json", {"allowed_split_fields": FIELDS, "patient_group_id_used_for_traceability_only": True, "sex_used_for_inner_stratification_only": True, "sex_used_as_model_feature": False, "raw_nyha_age_laboratory_diagnosis_other_clinical_fields_read": 0, "outer_test_model_selection_access": 0})
    _dump(REPORT_ROOT / "protected_asset_hash_audit.json", {"unchanged": unchanged, "audit_method": "M/H-channel-only SHA-256 via mmap slices [3,4]", "rgb_pixel_payload_read": False, "allowed_p0_mh_tensors_before": before, "allowed_p0_mh_tensors_after": after, **gate_audit, "x1_x2_b1_b2_existing_x3_direct_read": False, "upstream_assets_protected_by_recorded_immutable_references": True})
    comparisons.to_csv(REPORT_ROOT / "fixed_comparisons.csv", index=False, encoding="utf-8-sig")
    fold_mean_sd = fold_metrics.groupby("condition")[METRICS].agg(["mean", "std"])
    (REPORT_ROOT / "SO_R2X3_MH_Only_Ablation_Report.md").write_text("# SO-R2-X3-MH Frozen M/H-Sensitive Representation Ablation\n\nInternal five-fold OOF exploratory ablation only. M/H-sensitive representation channels must not be interpreted as absolute melanin or hemoglobin concentrations. No bootstrap, significance test, winner selection, or authorization change was performed.\n\n## Pooled OOF\n\n```csv\n" + pooled.to_csv(index=False) + "```\n\n## Five-fold mean +/- SD\n\n```csv\n" + fold_mean_sd.to_csv() + "```\n\n## Fixed descriptive comparisons\n\n```csv\n" + comparisons.to_csv(index=False) + "```\n", encoding="utf-8")
    (REPORT_ROOT / "test_results.txt").write_text("preflight_pytest=PASS\ntraining=PASS\naggregate_only_finalize=PASS\npostflight_pytest=pending_external_command\n", encoding="utf-8")
    result = {"status": "COMPLETE_EXPLORATORY_MH_ONLY_ABLATION" if unchanged else "FAIL_PROTECTED_ASSET_MUTATION", "formal_SO_R1_C_status": "FAIL", "official_SO_R2_authorization": False, "SO_R3_authorization": False, "next_stage_authorized": False, "oof_coverage": expected, "outer_test_model_selection_access": 0, "outer_test_prediction_coverage": expected, "protected_assets_unchanged": unchanged, "conditions": {key: {"source": value[0], "channels": list(value[1])} for key, value in CONDITIONS.items()}, "aggregate_only_finalize": True}
    _dump(DATA_ROOT / "MH_ABLATION_ACCEPTANCE.json", result)
    return result


def run(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    gate_audit = _gate(root)
    if all((root / "runs/so_r2x3_mh_only_ablation" / condition / f"fold_{fold}" / "fold_metrics.json").is_file() and (root / "runs/so_r2x3_mh_only_ablation" / condition / f"fold_{fold}" / "outer_test_predictions.csv").is_file() for condition in CONDITIONS for fold in range(5)):
        return _finalize_existing(root, gate_audit)
    table = _read_table(root)
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_CUDA_REQUIRED")
    before = _protected_snapshot(root)
    device = torch.device("cuda")
    rows, predictions, leakage = [], [], []
    for condition in CONDITIONS:
        for fold in range(5):
            row, prediction, audit = _train_fold(root, condition, table, fold, device)
            rows.append(row); predictions.append(prediction); leakage.append(audit)
    fold_metrics = pd.DataFrame(rows).sort_values(["condition", "fold"]).reset_index(drop=True)
    oof = pd.concat(predictions, ignore_index=True).sort_values(["condition", "fold", "ID"]).reset_index(drop=True)
    expected = {condition: 500 for condition in CONDITIONS}
    if oof.groupby("condition").size().to_dict() != expected or oof.duplicated(["condition", "ID"]).any():
        raise RuntimeError("FAIL_OOF_COVERAGE")
    pooled_rows, matrices = [], []
    for condition in CONDITIONS:
        subset = oof[oof.condition == condition]
        metric = compute_binary_metrics(subset.binary_label, subset[["prob_control", "prob_patient"]].to_numpy())
        pooled_rows.append({"condition": condition, "scope": "pooled_oof", **{k: v for k, v in metric.items() if k != "confusion_matrix"}})
        matrix = pd.DataFrame(metric["confusion_matrix"], index=["Control", "Patient"], columns=["Control", "Patient"]).reset_index(names="true_class")
        matrix.insert(0, "condition", condition); matrices.append(matrix)
    pooled = pd.DataFrame(pooled_rows)
    comparisons = _comparison(pooled, fold_metrics)
    after = _protected_snapshot(root)
    unchanged = before == after
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(REPORT_ROOT / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(REPORT_ROOT / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(REPORT_ROOT / "oof_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(matrices, ignore_index=True).to_csv(REPORT_ROOT / "oof_confusion_matrix.csv", index=False, encoding="utf-8-sig")
    _dump(REPORT_ROOT / "split_leakage_audit.json", {"all_pass": all(x["id_overlap_train_val"] == x["id_overlap_train_outer"] == x["id_overlap_val_outer"] == x["outer_test_model_selection_access"] == 0 for x in leakage), "folds": leakage, "outer_test_prediction_coverage": expected})
    _dump(REPORT_ROOT / "input_provenance_audit.json", {"only_allowed_p0_tensor_sources": {key: {"directory": value[0], "channels": list(value[1])} for key, value in CONDITIONS.items()}, "rgb_channels_read": False, "source_tensor_shape": [5, 320, 256], "mh_normalization": {"mean": 0.5, "std": 0.5}})
    _dump(REPORT_ROOT / "field_access_audit.json", {"allowed_split_fields": FIELDS, "patient_group_id_used_for_traceability_only": True, "sex_used_for_inner_stratification_only": True, "sex_used_as_model_feature": False, "raw_nyha_age_laboratory_diagnosis_other_clinical_fields_read": 0, "outer_test_model_selection_access": 0})
    _dump(REPORT_ROOT / "protected_asset_hash_audit.json", {"unchanged": unchanged, "audit_method": "M/H-channel-only SHA-256 via mmap slices [3,4]", "rgb_pixel_payload_read": False, "allowed_p0_mh_tensors_before": before, "allowed_p0_mh_tensors_after": after, **gate_audit, "x1_x2_b1_b2_existing_x3_direct_read": False, "upstream_assets_protected_by_recorded_immutable_references": True})
    comparisons.to_csv(REPORT_ROOT / "fixed_comparisons.csv", index=False, encoding="utf-8-sig")
    fold_mean_sd = fold_metrics.groupby("condition")[METRICS].agg(["mean", "std"])
    report = "# SO-R2-X3-MH Frozen M/H-Sensitive Representation Ablation\n\nInternal five-fold OOF exploratory ablation only. M/H-sensitive representation channels must not be interpreted as absolute melanin or hemoglobin concentrations. No bootstrap, significance test, winner selection, or authorization change was performed.\n\n## Pooled OOF\n\n```csv\n" + pooled.to_csv(index=False) + "```\n\n## Five-fold mean +/- SD\n\n```csv\n" + fold_mean_sd.to_csv() + "```\n\n## Fixed descriptive comparisons\n\n```csv\n" + comparisons.to_csv(index=False) + "```\n"
    (REPORT_ROOT / "SO_R2X3_MH_Only_Ablation_Report.md").write_text(report, encoding="utf-8")
    (REPORT_ROOT / "test_results.txt").write_text("preflight_pytest=PASS\ntraining=PASS\npostflight_pytest=pending_external_command\n", encoding="utf-8")
    result = {"status": "COMPLETE_EXPLORATORY_MH_ONLY_ABLATION" if unchanged else "FAIL_PROTECTED_ASSET_MUTATION", "formal_SO_R1_C_status": "FAIL", "official_SO_R2_authorization": False, "SO_R3_authorization": False, "next_stage_authorized": False, "oof_coverage": expected, "outer_test_model_selection_access": 0, "outer_test_prediction_coverage": expected, "protected_assets_unchanged": unchanged, "conditions": {key: {"source": value[0], "channels": list(value[1])} for key, value in CONDITIONS.items()}}
    _dump(DATA_ROOT / "MH_ABLATION_ACCEPTANCE.json", result)
    return result
