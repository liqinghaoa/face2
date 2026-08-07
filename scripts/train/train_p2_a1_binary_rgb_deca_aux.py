"""One independently initialized P2-A1 fold for the immutable P0-A five-fold split."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
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

LOGGER = logging.getLogger("p2_a1_5fold")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def class_metrics(confusion: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = [int(value) for value in confusion.reshape(-1)]
    return {
        "patient_sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "control_specificity": tn / (tn + fp) if tn + fp else 0.0,
        "patient_ppv": tp / (tp + fp) if tp + fp else 0.0,
        "patient_npv": tn / (tn + fn) if tn + fn else 0.0,
    }


def validate_protocol(config: dict, fold: int) -> None:
    data, model, training = config["data"], config["model"], config["training"]
    if config["task"]["type"] != "binary_control_vs_patient" or config["task"]["num_classes"] != 2:
        raise ValueError("P2-A1 requires the frozen binary Control-vs-Patient task")
    if list(data["folds"]) != [0, 1, 2, 3, 4] or fold not in data["folds"]:
        raise ValueError("P2-A1 full validation requires exactly folds 0-4")
    if Path(config["assets"]["p0a_root"]) != Path("data/processed/P0_Physics_Audit_v1"):
        raise ValueError("P2-A1 must use P0_Physics_Audit_v1")
    if Path(data["image_root"]) != Path("data/processed/P0_Physics_Audit_v1/images/e0b_meanbg_224"):
        raise ValueError("P2-A1 must use P0-A e0b_meanbg_224 RGB")
    if model["backbone"] != "resnet18" or model["pretrained"] != "imagenet" or model["num_classes"] != 2 or model["freeze_backbone"]:
        raise ValueError("P2-A1 ResNet18 protocol mismatch")
    if (model["aux_hidden_dim"], model["aux_output_dim"], model["aux_dropout"], model["initial_alpha"]) != (128, 512, 0.10, 0.10):
        raise ValueError("P2-A1 frozen auxiliary architecture mismatch")
    if config["auxiliary"]["feature_order"] != ["tex_code", "shape_code", "detail_code"] or config["auxiliary"]["feature_dim"] != 278:
        raise ValueError("P2-A1 frozen latent interface mismatch")
    if training["optimizer"].lower() != "adamw" or training["learning_rate"] != 1.0e-4 or training["weight_decay"] != 1.0e-4 or training["batch_size"] != 16:
        raise ValueError("P2-A1 frozen optimizer protocol mismatch")
    if training["max_epochs"] != 50 or training["early_stopping_patience"] != 10 or training["amp"] or training["monitor"] != "macro_auc" or training["monitor_mode"] != "max":
        raise ValueError("P2-A1 frozen selection protocol mismatch")
    if config["evaluation"]["threshold_search"]:
        raise ValueError("P2-A1 prohibits threshold search")


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader, device: torch.device, fold: int, epoch: int, checkpoint: Path) -> tuple[pd.DataFrame, dict]:
    model.eval()
    labels_all, probabilities, rows = [], [], []
    for batch in data_loader:
        outputs = model(batch["image"].to(device), batch["aux_vector"].to(device))
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        labels = batch["label"].numpy()
        labels_all.extend(labels.tolist())
        probabilities.append(probs)
        for index, label in enumerate(labels):
            prediction = int(probs[index].argmax())
            original_nyha = int(batch["original_nyha"][index])
            rows.append({
                "sample_id": batch["sample_id"][index], "patient_group_id": batch["patient_group_id"][index], "fold": fold,
                "true_label": int(label), "binary_label": int(label), "original_nyha": original_nyha, "original_label": original_nyha,
                "original_three_class_label": int(batch["original_three_class_label"][index]),
                "logit_control": float(logits[index, 0].cpu()), "logit_patient": float(logits[index, 1].cpu()),
                "logit_normal": float(logits[index, 0].cpu()), "prob_control": float(probs[index, 0]),
                "prob_normal": float(probs[index, 0]), "prob_patient": float(probs[index, 1]),
                "pred_label": prediction, "pred_class": prediction, "correct": int(prediction == label), "is_correct": int(prediction == label),
                "selected_epoch": epoch, "checkpoint_path": str(checkpoint),
            })
    return pd.DataFrame(rows), compute_binary_metrics(np.asarray(labels_all), np.concatenate(probabilities, axis=0))


def required_fold_outputs(fold_dir: Path) -> list[Path]:
    return [
        fold_dir / "aux_normalization_stats.npz", fold_dir / "checkpoints" / "best_macro_auc.pth",
        fold_dir / "checkpoints" / "last.pth", fold_dir / "training_history.csv", fold_dir / "val_predictions.csv",
        fold_dir / "metrics.json", fold_dir / "confusion_matrix.csv", fold_dir / "training_parameters.json",
        fold_dir / "aux_behavior.json", fold_dir / "_SUCCESS.json",
    ]


def run_fold(config: dict, output_dir: Path, fold: int, epochs: int | None = None, smoke: bool = False, overwrite: bool = False) -> Path:
    validate_protocol(config, fold)
    if not torch.cuda.is_available():
        raise RuntimeError("P2-A1 five-fold training requires CUDA; CPU fallback is prohibited")
    fold_dir = output_dir / f"fold_{fold}"
    if (fold_dir / "_SUCCESS.json").is_file() and not overwrite:
        raise FileExistsError(f"fold {fold} is already complete; pass --overwrite-fold explicitly to replace it")
    if fold_dir.exists() and overwrite:
        shutil.rmtree(fold_dir)
    checkpoint_dir = fold_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not (output_dir / "config_snapshot.yaml").exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        save_yaml(config, output_dir / "config_snapshot.yaml")
        save_environment(output_dir)
    configure_logging(output_dir / "experiment.log")
    set_random_seed(int(config["training"]["seed"]))

    data, training = config["data"], config["training"]
    train_transform = build_transforms("train", 224, config["normalize"]["mean"], config["normalize"]["std"], True)
    validation_transform = build_transforms("val", 224, config["normalize"]["mean"], config["normalize"]["std"], False)
    split = project_path(data["split_csv"])
    train_set = P2A1Dataset(split, project_path(data["image_root"]), project_path(config["assets"]["p1_root"]), config["p1"], fold, True, train_transform)
    validation_set = P2A1Dataset(split, project_path(data["image_root"]), project_path(config["assets"]["p1_root"]), config["p1"], fold, False, validation_transform)
    if set(train_set.frame.patient_group_id.astype(str)) & set(validation_set.frame.patient_group_id.astype(str)):
        raise ValueError(f"fold {fold}: patient-group leakage")
    statistics = train_stats(train_set, config["auxiliary"]["epsilon"], len(validation_set))
    train_set.set_stats(statistics)
    validation_set.set_stats(statistics)
    np.savez(fold_dir / "aux_normalization_stats.npz", **statistics)

    # Reuse E0B's fixed fold-specific DataLoader seed convention: base+fold and base+1000+fold.
    train_loader = loader(train_set, training["batch_size"], True, training["num_workers"], training["seed"] + fold, training["pin_memory"])
    validation_loader = loader(validation_set, training["batch_size"], False, training["num_workers"], training["seed"] + 1000 + fold, training["pin_memory"])
    device = torch.device("cuda")
    model = P2A1Fusion(278, 128, 0.10, 0.10).to(device)
    labels = [int(item["label"]) for item in train_set]
    weights = compute_class_weights(labels, 2).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    label_array = np.asarray(labels)
    parameters = {
        "fold": fold, "train_control_count": int((label_array == 0).sum()), "train_patient_count": int((label_array == 1).sum()),
        "validation_control_count": int((validation_set.frame.label_3class.map(lambda x: 0 if int(x) == 0 else 1) == 0).sum()),
        "validation_patient_count": int((validation_set.frame.label_3class.map(lambda x: 0 if int(x) == 0 else 1) == 1).sum()),
        "weight_control": float(weights[0]), "weight_patient": float(weights[1]), "batch_size": training["batch_size"],
        "total_params": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "new_auxiliary_and_fusion_params": sum(parameter.numel() for name, parameter in model.named_parameters() if not name.startswith("rgb.")),
    }
    (fold_dir / "training_parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")

    max_epochs = int(epochs if epochs is not None else training["max_epochs"])
    best_auc, patience, history, best_behavior, final_behavior = -1.0, 0, [], None, None
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses, rgb_norms, aux_norms, scaled_norms = [], [], [], []
        for batch in train_loader:
            outputs = model(batch["image"].to(device), batch["aux_vector"].to(device))
            loss = criterion(outputs["logits"], batch["label"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            rgb_norms.append(float(outputs["z_rgb"].norm(dim=1).mean().item()))
            aux_norms.append(float(outputs["z_aux"].norm(dim=1).mean().item()))
            scaled_norms.append(float((outputs["alpha"] * outputs["z_aux"]).norm(dim=1).mean().item()))
        _, validation_metrics = evaluate(model, validation_loader, device, fold, epoch, checkpoint_dir / "best_macro_auc.pth")
        rgb_norm, scaled_norm = float(np.mean(rgb_norms)), float(np.mean(scaled_norms))
        row = {"epoch": epoch, "learning_rate": training["learning_rate"], "train_loss": float(np.mean(losses)), **{f"val_{key}": value for key, value in flatten_metrics(validation_metrics).items()}, "alpha": float(torch.sigmoid(model.alpha_logit).item()), "mean_rgb_feature_norm": rgb_norm, "mean_aux_feature_norm": float(np.mean(aux_norms)), "mean_scaled_aux_norm": scaled_norm, "scaled_aux_to_rgb_norm_ratio": scaled_norm / (rgb_norm + 1e-6)}
        history.append(row)
        final_behavior = row
        if validation_metrics["macro_auc"] > best_auc:
            best_auc, patience, best_behavior = validation_metrics["macro_auc"], 0, row.copy()
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_macro_auc": best_auc, "config": config, "fold": fold}, checkpoint_dir / "best_macro_auc.pth")
        else:
            patience += 1
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_macro_auc": best_auc, "config": config, "fold": fold}, checkpoint_dir / "last.pth")
        LOGGER.info("fold=%d epoch=%d auc=%.4f best=%.4f patience=%d/%d", fold, epoch, validation_metrics["macro_auc"], best_auc, patience, training["early_stopping_patience"])
        if patience >= training["early_stopping_patience"]:
            break

    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    checkpoint = torch.load(checkpoint_dir / "best_macro_auc.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions, metrics = evaluate(model, validation_loader, device, fold, int(checkpoint["epoch"]), checkpoint_dir / "best_macro_auc.pth")
    predictions.to_csv(fold_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    final_metrics = {"fold": fold, "best_epoch": int(checkpoint["epoch"]), "stopped_epoch": int(history[-1]["epoch"]), "training_seconds": time.perf_counter() - started, **flatten_metrics(metrics), **class_metrics(metrics["confusion_matrix"])}
    (fold_dir / "metrics.json").write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(metrics["confusion_matrix"], index=["control", "patient"], columns=["control", "patient"]).to_csv(fold_dir / "confusion_matrix.csv", index_label="true\\pred", encoding="utf-8-sig")
    behavior = {"fold": fold, "best_epoch": int(checkpoint["epoch"]), "stopped_epoch": int(history[-1]["epoch"]), "initial_alpha": 0.10, "best_epoch_alpha": best_behavior["alpha"], "final_epoch_alpha": final_behavior["alpha"], "best_epoch_norm_ratio": best_behavior["scaled_aux_to_rgb_norm_ratio"], "final_epoch_norm_ratio": final_behavior["scaled_aux_to_rgb_norm_ratio"]}
    (fold_dir / "aux_behavior.json").write_text(json.dumps(behavior, ensure_ascii=False, indent=2), encoding="utf-8")
    success = {"fold": fold, "smoke": smoke, "best_epoch": int(checkpoint["epoch"]), "stopped_epoch": int(history[-1]["epoch"]), "required_outputs": [str(path.relative_to(fold_dir)) for path in required_fold_outputs(fold_dir)[:-1]]}
    (fold_dir / "_SUCCESS.json").write_text(json.dumps(success, ensure_ascii=False, indent=2), encoding="utf-8")
    return fold_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite-fold", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke != (args.epochs is not None):
        raise ValueError("--epochs is reserved for explicit --smoke runs")
    config = load_yaml(args.config)
    output_dir = project_path(args.output_dir)
    fold_dir = run_fold(config, output_dir, args.fold, args.epochs, args.smoke, args.overwrite_fold)
    print(f"P2_A1_FOLD_COMPLETED={fold_dir}")


if __name__ == "__main__":
    main()
