"""GPU trainer for the fixed-fold P2-A1 RGB + frozen-DECA auxiliary screen."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.nyha_3class_face_dataset import build_transforms
from datasets.p2_a1_rgb_deca_aux_dataset import P2A1Dataset, train_stats
from losses.classification_losses import compute_class_weights
from metrics.binary_classification_metrics import compute_binary_metrics, flatten_metrics
from models.p2_a1_rgb_deca_residual_fusion import P2A1Fusion
from scripts.train.train_e0b_global_resnet18_control_patient_binary_5fold import loader, save_environment
from utils.experiment_utils import configure_logging, load_yaml, save_yaml, set_random_seed

LOGGER = logging.getLogger("p2_a1")


def project_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, help="Smoke-only override supplied by the runner")
    return parser.parse_args()


def validate_protocol(config: dict) -> None:
    expected_root = Path("data/processed/P0_Physics_Audit_v1")
    if config["task"]["type"] != "binary_control_vs_patient" or config["task"]["num_classes"] != 2:
        raise ValueError("P2-A1 is fixed to Control vs Patient binary classification")
    if config["data"]["fold"] != 0 or config["model"]["backbone"] != "resnet18":
        raise ValueError("P2-A1 is fixed to fold 0 and ResNet18")
    if Path(config["assets"]["p0a_root"]) != expected_root:
        raise ValueError("P2-A1 RGB provenance must be P0_Physics_Audit_v1")
    if Path(config["data"]["image_root"]) != expected_root / "images/e0b_meanbg_224":
        raise ValueError("P2-A1 image root must be P0-A e0b_meanbg_224")
    if config["evaluation"]["threshold_search"] or config["training"]["amp"]:
        raise ValueError("P2-A1 forbids threshold search and AMP")
    if config["auxiliary"]["feature_order"] != ["tex_code", "shape_code", "detail_code"]:
        raise ValueError("P2-A1 auxiliary feature order is fixed")


def evaluate(model, data_loader, device, epoch: int, checkpoint: Path) -> tuple[pd.DataFrame, dict]:
    model.eval()
    rows, targets, probabilities = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch["image"].to(device), batch["aux_vector"].to(device))
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            labels = batch["label"].numpy()
            targets.extend(labels.tolist())
            probabilities.append(probs)
            for index, label in enumerate(labels):
                prediction = int(probs[index].argmax())
                rows.append({
                    "sample_id": batch["sample_id"][index],
                    "patient_group_id": batch["patient_group_id"][index],
                    "fold": 0,
                    "original_label": int(batch["original_nyha"][index]),
                    "original_three_class_label": int(batch["original_three_class_label"][index]),
                    "binary_label": int(label),
                    "logit_normal": float(logits[index, 0].cpu()),
                    "logit_patient": float(logits[index, 1].cpu()),
                    "prob_normal": float(probs[index, 0]),
                    "prob_patient": float(probs[index, 1]),
                    "pred_class": prediction,
                    "is_correct": int(prediction == label),
                    "selected_epoch": epoch,
                    "checkpoint_path": str(checkpoint),
                })
    return pd.DataFrame(rows), compute_binary_metrics(np.asarray(targets), np.concatenate(probabilities, axis=0))


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    validate_protocol(config)
    if args.epochs is not None:
        config["training"]["max_epochs"] = args.epochs

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "experiment.log")
    save_yaml(config, output_dir / "config_snapshot.yaml")
    save_environment(output_dir)
    set_random_seed(config["training"]["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("P2-A1 formal/smoke training requires CUDA; refusing CPU fallback")
    device = torch.device("cuda")

    data = config["data"]
    train_transform = build_transforms("train", 224, config["normalize"]["mean"], config["normalize"]["std"], True)
    validation_transform = build_transforms("val", 224, config["normalize"]["mean"], config["normalize"]["std"], False)
    train_set = P2A1Dataset(project_path(data["split_table"]), project_path(data["image_root"]), project_path(config["assets"]["p1_root"]), config["p1"], 0, True, train_transform)
    validation_set = P2A1Dataset(project_path(data["split_table"]), project_path(data["image_root"]), project_path(config["assets"]["p1_root"]), config["p1"], 0, False, validation_transform)
    if set(train_set.frame.patient_group_id.astype(str)) & set(validation_set.frame.patient_group_id.astype(str)):
        raise ValueError("P2-A1 detected train/validation patient-group leakage")

    statistics = train_stats(train_set, config["auxiliary"]["epsilon"])
    train_set.set_stats(statistics)
    validation_set.set_stats(statistics)
    fold_dir, checkpoint_dir = output_dir / "fold_0", output_dir / "fold_0" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.savez(fold_dir / "aux_normalization_stats.npz", **statistics)

    training = config["training"]
    train_loader = loader(train_set, training["batch_size"], True, training["num_workers"], training["seed"], training["pin_memory"])
    validation_loader = loader(validation_set, training["batch_size"], False, training["num_workers"], training["seed"] + 1000, training["pin_memory"])
    model = P2A1Fusion(TOTAL_FEATURE_DIM := 278, config["model"]["aux_hidden_dim"], config["model"]["aux_dropout"], config["model"]["initial_alpha"]).to(device)
    train_labels = [int(item["label"]) for item in train_set]
    class_weights = compute_class_weights(train_labels, 2).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    label_array = np.asarray(train_labels)
    (fold_dir / "training_parameters.json").write_text(json.dumps({
        "train_control_count": int((label_array == 0).sum()), "train_patient_count": int((label_array == 1).sum()),
        "weight_control": float(class_weights[0]), "weight_patient": float(class_weights[1]),
        "batch_size": training["batch_size"], "total_params": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "new_auxiliary_and_fusion_params": sum(parameter.numel() for name, parameter in model.named_parameters() if not name.startswith("rgb.")),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    best_auc, patience = -1.0, 0
    history, best_behavior, final_behavior = [], None, None
    for epoch in range(1, training["max_epochs"] + 1):
        model.train()
        losses, rgb_norms, aux_norms, scaled_aux_norms = [], [], [], []
        for batch in train_loader:
            outputs = model(batch["image"].to(device), batch["aux_vector"].to(device))
            loss = criterion(outputs["logits"], batch["label"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            rgb_norms.append(float(outputs["z_rgb"].norm(dim=1).mean().item()))
            aux_norms.append(float(outputs["z_aux"].norm(dim=1).mean().item()))
            scaled_aux_norms.append(float((outputs["alpha"] * outputs["z_aux"]).norm(dim=1).mean().item()))
        _, validation_metrics = evaluate(model, validation_loader, device, epoch, checkpoint_dir / "best_macro_auc.pth")
        rgb_norm, scaled_norm = float(np.mean(rgb_norms)), float(np.mean(scaled_aux_norms))
        row = {
            "epoch": epoch, "learning_rate": training["learning_rate"], "train_loss": float(np.mean(losses)),
            **{f"val_{key}": value for key, value in flatten_metrics(validation_metrics).items()},
            "alpha": float(torch.sigmoid(model.alpha_logit).item()), "mean_rgb_feature_norm": rgb_norm,
            "mean_aux_feature_norm": float(np.mean(aux_norms)), "mean_scaled_aux_norm": scaled_norm,
            "scaled_aux_to_rgb_norm_ratio": scaled_norm / (rgb_norm + 1e-6),
        }
        history.append(row)
        final_behavior = row
        if validation_metrics["macro_auc"] > best_auc:
            best_auc, patience, best_behavior = validation_metrics["macro_auc"], 0, row.copy()
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_macro_auc": best_auc, "config": config}, checkpoint_dir / "best_macro_auc.pth")
        else:
            patience += 1
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_macro_auc": best_auc, "config": config}, checkpoint_dir / "last.pth")
        LOGGER.info("epoch=%d auc=%.4f best=%.4f patience=%d/%d", epoch, validation_metrics["macro_auc"], best_auc, patience, training["early_stopping_patience"])
        if patience >= training["early_stopping_patience"]:
            break

    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    best_checkpoint = torch.load(checkpoint_dir / "best_macro_auc.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    predictions, metrics = evaluate(model, validation_loader, device, best_checkpoint["epoch"], checkpoint_dir / "best_macro_auc.pth")
    predictions.to_csv(fold_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    (fold_dir / "metrics.json").write_text(json.dumps({"best_epoch": best_checkpoint["epoch"], **flatten_metrics(metrics)}, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(metrics["confusion_matrix"], index=["control", "patient"], columns=["control", "patient"]).to_csv(fold_dir / "confusion_matrix.csv", index_label="true\\pred")
    (output_dir / "p2_a1_aux_behavior.json").write_text(json.dumps({
        "initial_alpha": config["model"]["initial_alpha"], "best_epoch_alpha": best_behavior["alpha"],
        "final_epoch_alpha": final_behavior["alpha"], "best_epoch_norm_ratio": best_behavior["scaled_aux_to_rgb_norm_ratio"],
        "final_epoch_norm_ratio": final_behavior["scaled_aux_to_rgb_norm_ratio"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"P2_A1_EXPERIMENT_DIR={output_dir}")


if __name__ == "__main__":
    main()
