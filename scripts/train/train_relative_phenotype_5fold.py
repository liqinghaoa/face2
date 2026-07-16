"""Train P2-1 on the original fixed five-fold protocol."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.nyha_relative_eye_cheek_dataset import (  # noqa: E402
    RelativeEyeCheekDataset,
    raw_roi_and_mask_samples,
)
from losses.classification_losses import build_criterion, compute_class_weights  # noqa: E402
from metrics.classification_metrics import CLASS_NAMES, compute_classification_metrics, flatten_metrics  # noqa: E402
from models.relative_optical_phenotype import RelativeOpticalPhenotypeModel, count_parameters  # noqa: E402
from utils.experiment_utils import load_yaml, save_yaml, set_random_seed  # noqa: E402
from utils.optical_channel_stats import compute_shared_stats, save_fold_stats, sha256_file  # noqa: E402


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, action="append", dest="folds")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def split_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
    expected = pd.to_numeric(frame.NYHA).map(lambda x: 0 if x == 0 else (1 if x in (1, 2) else 2))
    if not expected.equals(pd.to_numeric(frame.label_3class)):
        raise ValueError(f"label mapping mismatch: {path}")
    return frame


def paths(config: dict[str, Any]) -> dict[str, Path]:
    data = config["data"]
    return {
        "split": resolve(data["split_dir"]),
        "global": resolve(data["global_image_root"]),
        "eye": resolve(data["eye_image_root"]),
        "cheek": resolve(data["cheek_image_root"]),
        "eye_mask": resolve(data["eye_mask_root"]),
        "cheek_mask": resolve(data["cheek_mask_root"]),
    }


def experiment_dir(config: dict[str, Any], smoke: bool = False) -> Path:
    base = resolve(config["experiment"]["output_dir"])
    name = config["experiment"]["name"] + ("_smoke" if smoke else "")
    return base / name


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    p = paths(config)
    n_folds = int(config["data"].get("n_folds", 5))
    val_frames = [split_frame(p["split"] / f"fold_{fold}_val.csv") for fold in range(n_folds)]
    all_validation = pd.concat(val_frames, ignore_index=True)
    expected_count = int(config["data"].get("expected_sample_count", len(all_validation)))
    if len(all_validation) != expected_count or all_validation.ID.nunique() != expected_count:
        raise ValueError(
            f"validation folds must cover {expected_count} unique IDs exactly once; "
            f"got rows={len(all_validation)}, unique={all_validation.ID.nunique()}"
        )
    all_ids = set(all_validation.ID.astype(str))
    expected_groups = config["data"].get("expected_patient_groups")
    if expected_groups is not None and all_validation.patient_group_id.nunique() != int(expected_groups):
        raise ValueError("patient-group count differs from configured S2 protocol")
    required_ids: list[str] = []
    audits = []
    for fold in range(n_folds):
        train_path, val_path = p["split"] / f"fold_{fold}_train.csv", p["split"] / f"fold_{fold}_val.csv"
        train, val = split_frame(train_path), split_frame(val_path)
        if set(train.ID.astype(str)) != all_ids.difference(set(val.ID.astype(str))):
            raise ValueError(f"fold {fold} train/validation sets do not close the fixed cohort")
        if len(train) + len(val) != expected_count:
            raise ValueError(f"fold {fold} train/validation sizes do not sum to {expected_count}")
        overlap = set(train.patient_group_id.astype(str)) & set(val.patient_group_id.astype(str))
        if overlap:
            raise ValueError(f"fold {fold} patient group leakage")
        if set(train.label_3class) != {0, 1, 2} or set(val.label_3class) != {0, 1, 2}:
            raise ValueError(f"fold {fold} lacks a class")
        required_ids.extend(val.ID.astype(str))
        audits.append({"fold": fold, "train_rows": len(train), "val_rows": len(val), "group_overlap": len(overlap), "status": "PASS"})
    if len(required_ids) != expected_count or len(set(required_ids)) != expected_count:
        raise ValueError(f"fixed validation folds do not cover {expected_count} unique IDs")
    for identifier in required_ids:
        for key in ("global", "eye", "cheek", "eye_mask", "cheek_mask"):
            if not (p[key] / f"{identifier}.png").is_file():
                raise FileNotFoundError(f"missing {key} asset for a required ID")
    out = experiment_dir(config)
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audits).to_csv(out / "protocol/preflight_audit.csv", index=False, encoding="utf-8-sig")
    payload = {
        "sample_count": expected_count,
        "unique_ids": expected_count,
        "patient_groups": int(all_validation.patient_group_id.nunique()),
        "folds": n_folds,
        "status": "PASS",
    }
    (out / "protocol/preflight_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def fold_stats(config: dict[str, Any], fold: int, out: Path) -> dict[str, Any]:
    stats_path = out / f"fold_{fold}/protocol/optical_channel_stats.json"
    train_csv = paths(config)["split"] / f"fold_{fold}_train.csv"
    if stats_path.is_file():
        loaded = json.loads(stats_path.read_text(encoding="utf-8"))
        if loaded.get("train_split_sha256") == sha256_file(train_csv):
            return loaded
    p = paths(config)
    train = split_frame(train_csv)
    samples = raw_roi_and_mask_samples(train, p["eye"], p["cheek"], p["eye_mask"], p["cheek_mask"])
    stats = compute_shared_stats(samples, float(config["optical"]["epsilon"]))
    save_fold_stats(stats_path, stats, fold, train_csv)
    return json.loads(stats_path.read_text(encoding="utf-8"))


def dataset(config: dict[str, Any], csv_path: Path, stats: dict, train: bool) -> RelativeEyeCheekDataset:
    p = paths(config)
    return RelativeEyeCheekDataset(
        csv_path, p["global"], p["eye"], p["cheek"], p["eye_mask"], p["cheek_mask"],
        stats["mean"], stats["std"], int(config["data"]["image_size"]), train,
        bool(config["augmentation"]["horizontal_flip"]), float(config["optical"]["epsilon"]),
    )


def loader(ds: RelativeEyeCheekDataset, config: dict[str, Any], fold: int, train: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(int(config["train"]["random_seed"]) + fold)
    data_loader = DataLoader(
        ds, batch_size=int(config["train"]["batch_size"]), shuffle=train,
        num_workers=int(config["train"]["num_workers"]), pin_memory=bool(config["train"]["pin_memory"]),
        generator=generator,
    )
    data_loader._s2_generator = generator  # type: ignore[attr-defined]
    return data_loader


def random_state_payload(data_loader: DataLoader) -> dict[str, Any]:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_loader_generator_state": data_loader._s2_generator.get_state(),  # type: ignore[attr-defined]
    }


def restore_random_state(payload: dict[str, Any], data_loader: DataLoader) -> None:
    if "python_random_state" in payload:
        random.setstate(payload["python_random_state"])
        np.random.set_state(payload["numpy_random_state"])
        # Checkpoints are loaded with ``map_location=device`` so tensor-valued
        # RNG states may be remapped to CUDA together with model weights.  Both
        # CPU and DataLoader generators require a CPU ByteTensor state.
        torch.set_rng_state(payload["torch_random_state"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_random_state_all") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_random_state_all"]])
        data_loader._s2_generator.set_state(payload["data_loader_generator_state"].cpu())  # type: ignore[attr-defined]


def model(config: dict[str, Any]) -> RelativeOpticalPhenotypeModel:
    section = config["model"]
    return RelativeOpticalPhenotypeModel(
        pretrained=section["pretrained"], projection_dim=int(section["projection_dim"]),
        dropout=float(section["dropout"]), num_classes=int(section["num_classes"]),
    )


def inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: batch[key].to(device, non_blocking=True) for key in ("global_image", "eye_optical", "cheek_optical")}


def metrics(true: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    result = compute_classification_metrics(true, prob, 3)
    pred = prob.argmax(1)
    result.update({
        "ordinal_mae": float(np.abs(pred - true).mean()),
        "within_one_accuracy": float((np.abs(pred - true) <= 1).mean()),
        "extreme_error_rate": float((np.abs(pred - true) == 2).mean()),
        "quadratic_weighted_kappa": float(cohen_kappa_score(true, pred, weights="quadratic")),
    })
    return result


def validate(net: nn.Module, data: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, dict[str, Any]]:
    net.eval(); total = 0.0; count = 0; labels, probs = [], []
    with torch.no_grad():
        for batch in data:
            target = batch["label"].to(device)
            logits = net(**inputs(batch, device)); loss = criterion(logits, target)
            total += float(loss) * len(target); count += len(target)
            labels.append(target.cpu().numpy()); probs.append(torch.softmax(logits, 1).cpu().numpy())
    return total / count, metrics(np.concatenate(labels), np.concatenate(probs))


def completed_fold(config: dict[str, Any], fold: int, out: Path) -> bool:
    p = out / f"fold_{fold}"
    needed = [p / "checkpoints/best_macro_auc.pth", p / "predictions/val_predictions.csv", p / "metrics/fold_metrics.csv", p / "protocol/optical_channel_stats.json"]
    if not all(x.is_file() for x in needed):
        return False
    pred = pd.read_csv(needed[1])
    stats = json.loads(needed[3].read_text(encoding="utf-8"))
    train_csv = paths(config)["split"] / f"fold_{fold}_train.csv"
    prob = pred[["prob_normal", "prob_mild", "prob_severe"]].to_numpy(float)
    expected = len(split_frame(paths(config)["split"] / f"fold_{fold}_val.csv"))
    return len(pred) == expected and np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=1e-6) and stats.get("train_split_sha256") == sha256_file(train_csv)


def evaluate(net: nn.Module, data: DataLoader, checkpoint: Path, device: torch.device, fold_dir: Path) -> None:
    payload = torch.load(checkpoint, map_location=device)
    net.load_state_dict(payload["model_state_dict"]); net.eval()
    rows, true, probabilities = [], [], []
    with torch.no_grad():
        for batch in data:
            prob = torch.softmax(net(**inputs(batch, device)), 1).cpu().numpy()
            label = batch["label"].numpy(); pred = prob.argmax(1)
            true.extend(label.tolist()); probabilities.append(prob)
            for i in range(len(label)):
                y, yhat = int(label[i]), int(pred[i])
                rows.append({
                    "ID": str(batch["ID"][i]), "patient_group_id": str(batch["patient_group_id"][i]),
                    "NYHA": int(batch["NYHA"][i]), "SEX": int(batch["SEX"][i]),
                    "label_3class": y, "label_3class_name": CLASS_NAMES[y],
                    "pred_class": yhat, "pred_class_name": CLASS_NAMES[yhat],
                    "prob_normal": float(prob[i, 0]), "prob_mild": float(prob[i, 1]), "prob_severe": float(prob[i, 2]),
                    "correct": int(y == yhat), "fold": int(batch["fold"][i]),
                })
    pred_frame = pd.DataFrame(rows)
    (fold_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (fold_dir / "metrics").mkdir(parents=True, exist_ok=True)
    pred_frame.to_csv(fold_dir / "predictions/val_predictions.csv", index=False, encoding="utf-8-sig")
    result = metrics(np.asarray(true), np.concatenate(probabilities))
    row = flatten_metrics(result); row["fold"] = int(pred_frame.fold.iloc[0]); row["selected_epoch"] = int(payload["epoch"])
    pd.DataFrame([row]).to_csv(fold_dir / "metrics/fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result["confusion_matrix"], index=list(CLASS_NAMES.values()), columns=list(CLASS_NAMES.values())).to_csv(fold_dir / "metrics/confusion_matrix.csv", encoding="utf-8-sig")


def train_fold(config: dict[str, Any], fold: int, out: Path, resume: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = paths(config); stats = fold_stats(config, fold, out)
    train_ds = dataset(config, p["split"] / f"fold_{fold}_train.csv", stats, True)
    val_ds = dataset(config, p["split"] / f"fold_{fold}_val.csv", stats, False)
    train_loader, val_loader = loader(train_ds, config, fold, True), loader(val_ds, config, fold, False)
    net = model(config).to(device)
    weights = compute_class_weights(train_ds.frame.label_3class.to_numpy(), 3).to(device)
    criterion = build_criterion("weighted_cross_entropy", class_weights=weights)
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(config["train"]["lr"]), weight_decay=float(config["train"]["weight_decay"]))
    fold_dir = out / f"fold_{fold}"; checkpoint_dir = fold_dir / "checkpoints"; log_dir = fold_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, fold_dir / "resolved_config.yaml")
    epochs, patience = int(config["train"]["epochs"]), int(config["train"]["early_stopping_patience"])
    start, best, stale, records = 1, -math.inf, 0, []
    last = checkpoint_dir / "last.pth"
    if resume and last.is_file():
        payload = torch.load(last, map_location=device); net.load_state_dict(payload["model_state_dict"]); optimizer.load_state_dict(payload["optimizer_state_dict"])
        start, best, stale = int(payload["epoch"]) + 1, float(payload["best_macro_auc"]), int(payload["epochs_without_improvement"])
        log_path = log_dir / "train_log.csv"
        records = pd.read_csv(log_path).to_dict("records") if log_path.is_file() else []
        restore_random_state(payload, train_loader)
    for epoch in range(start, epochs + 1):
        net.train(); total = 0.0; count = 0
        for batch in train_loader:
            target = batch["label"].to(device); optimizer.zero_grad(set_to_none=True)
            logits = net(**inputs(batch, device)); loss = criterion(logits, target); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(target); count += len(target)
        val_loss, result = validate(net, val_loader, criterion, device)
        score = float(result["macro_auc"])
        record = {"epoch": epoch, "train_loss": total / count, "val_loss": val_loss, **{f"val_{k}": v for k, v in flatten_metrics(result).items()}}
        records.append(record); pd.DataFrame(records).to_csv(log_dir / "train_log.csv", index=False, encoding="utf-8-sig")
        improved = math.isfinite(score) and score > best
        stale = 0 if improved else stale + 1; best = score if improved else best
        payload = {"epoch": epoch, "fold": fold, "model_state_dict": net.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_macro_auc": best, "epochs_without_improvement": stale, "config": config, **random_state_payload(train_loader)}
        torch.save(payload, last)
        if improved:
            torch.save(payload, checkpoint_dir / "best_macro_auc.pth")
        print(f"fold={fold} epoch={epoch}/{epochs} loss={total/count:.5f} val_auc={score:.4f} best={best:.4f} patience={stale}/{patience}", flush=True)
        if stale >= patience:
            break
    best_path = checkpoint_dir / "best_macro_auc.pth"
    if not best_path.is_file(): shutil.copy2(last, best_path)
    evaluate(net, val_loader, best_path, device, fold_dir)
    history = pd.DataFrame(records)
    history.to_csv(fold_dir / "train_history.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(fold_dir / "predictions/val_predictions.csv", fold_dir / "fold_predictions.csv")
    best_payload = torch.load(best_path, map_location="cpu")
    selected = {
        "fold": fold,
        "selected_epoch": int(best_payload["epoch"]),
        "actual_training_epochs": int(history["epoch"].max()),
        "best_macro_auc": float(best_payload["best_macro_auc"]),
        "early_stopped": int(history["epoch"].max()) < epochs,
        "stop_reason": "early_stopping_patience_reached" if int(history["epoch"].max()) < epochs else "max_epochs_reached",
        "best_checkpoint": str(best_path.resolve()),
    }
    (fold_dir / "selected_epoch.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics_frame = pd.read_csv(fold_dir / "metrics/fold_metrics.csv")
    (fold_dir / "fold_metrics.json").write_text(
        json.dumps(json.loads(metrics_frame.to_json(orient="records"))[0], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def summarize(config: dict[str, Any], out: Path) -> Path:
    fold_frames, preds = [], []
    for fold in range(5):
        if not completed_fold(config, fold, out): raise RuntimeError(f"fold {fold} incomplete")
        fold_frames.append(pd.read_csv(out / f"fold_{fold}/metrics/fold_metrics.csv")); preds.append(pd.read_csv(out / f"fold_{fold}/predictions/val_predictions.csv", dtype={"ID": str, "patient_group_id": str}))
    fold_metrics = pd.concat(fold_frames).sort_values("fold"); oof = pd.concat(preds).sort_values(["fold", "ID"])
    n_folds = int(config["data"].get("n_folds", 5))
    expected_count = int(config["data"].get("expected_sample_count", len(oof)))
    if len(oof) != expected_count or oof.ID.nunique() != expected_count:
        raise ValueError(f"OOF must contain {expected_count} unique IDs")
    reference = pd.concat([split_frame(paths(config)["split"] / f"fold_{f}_val.csv") for f in range(n_folds)])[ ["ID", "patient_group_id", "NYHA", "SEX", "label_3class", "fold"] ].sort_values("ID")
    aligned = oof.sort_values("ID").reset_index(drop=True)
    if not aligned[["ID", "patient_group_id", "NYHA", "SEX", "label_3class", "fold"]].astype(str).equals(reference.reset_index(drop=True).astype(str)):
        raise ValueError("P2-1 OOF does not align with fixed split truth")
    prob = oof[["prob_normal", "prob_mild", "prob_severe"]].to_numpy(); result = metrics(oof.label_3class.to_numpy(), prob)
    summary = out / "summary"; summary.mkdir(parents=True, exist_ok=True)
    oof.to_csv(summary / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(summary / "fold_metrics_all.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(summary / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([flatten_metrics(result)]).to_csv(summary / "oof_metrics.csv", index=False, encoding="utf-8-sig")
    numeric = fold_metrics.select_dtypes(include=[np.number]).drop(columns=["fold"], errors="ignore")
    pd.DataFrame({"metric": numeric.columns, "mean": numeric.mean().values, "std": numeric.std(ddof=1).values}).to_csv(summary / "mean_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"metric": numeric.columns, "mean": numeric.mean().values, "std": numeric.std(ddof=1).values}).to_csv(summary / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    args = args_parser(); config = load_yaml(resolve(args.config))
    if args.epochs is not None: config["train"]["epochs"] = args.epochs
    if args.batch_size is not None: config["train"]["batch_size"] = args.batch_size
    if args.smoke_test: config["train"]["epochs"] = 1
    out = experiment_dir(config, args.smoke_test); out.mkdir(parents=True, exist_ok=True)
    save_yaml(config, out / "config.yaml")
    preflight(config)
    # Match P1 exactly: set the global seed once before the sequential fold loop.
    # DataLoader generators still use seed + fold, as in the parent runner.
    set_random_seed(int(config["train"]["random_seed"]))
    if args.summarize_only: summarize(config, out); return
    n_folds = int(config["data"].get("n_folds", 5))
    folds = args.folds or ([0] if args.smoke_test else list(range(n_folds)))
    for fold in folds:
        if args.skip_existing and completed_fold(config, fold, out):
            print(f"fold={fold} complete; skipped", flush=True); continue
        train_fold(config, fold, out, args.resume)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not args.smoke_test and all(completed_fold(config, fold, out) for fold in range(n_folds)):
        summarize(config, out)
    counts = count_parameters(model(config))
    (out / "model_summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(f"output={out}", flush=True)


if __name__ == "__main__":
    main()
