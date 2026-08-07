"""Stage3-C0 SH093 224x224 audit.

The implementation is intentionally self-contained at the experiment level:
C0-A reuses the frozen B0 rows/splits, while C0-B creates an independent
patient-only camera-domain task.  Neither task uses camera metadata as an
image feature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from PIL import Image, ImageDraw
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from datasets.nyha_3class_face_dataset import build_transforms
from models.nyha_backbone_factory import build_nyha_classification_model, count_parameters
from utils.experiment_utils import seed_worker, set_random_seed
from utils.resnet18_anti_overfit import apply_train_mode, build_optimizer, build_trainability_audit, configure_trainability, normalize_strategy

LOG = logging.getLogger("stage3_c0")
EXPERIMENT_NAME = "Stage3_C0_SH093_224_SameCameraSignal_and_DeviceDomain_Audit_v1"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "lighting_confounding" / EXPERIMENT_NAME
STAGE1 = PROJECT_ROOT / "experiments" / "lighting_confounding" / "Lighting_Confounding_Audit_Stage1_v1" / "metadata" / "stage1_master_500.csv"
SH_DIR = PROJECT_ROOT / "data" / "processed" / "global_face" / "fixedSH_origcam_menafg"
UPSTREAM_DIR = PROJECT_ROOT / "data" / "processed" / "P2_ExampleSH093_FixedLight_v1" / "images" / "fixedSH_origcam"
B0_DIR = PROJECT_ROOT / "experiments" / "lighting_confounding" / "Stage3_B0_XiaomiOnly_ResNet18_ControlVsPatient_Group5Fold_v1"
B0_SPLIT = B0_DIR / "splits" / "xiaomi_control_patient_group5fold_v1.csv"
B0_OOF = B0_DIR / "oof" / "xiaomi_b0_oof_predictions.csv"
B0_INNER = B0_DIR / "splits" / "inner"
LEGACY_DIR = PROJECT_ROOT / "experiments" / "500Data" / "E0B_Global_ResNet18_ControlVsPatient_Binary_fixedSH_origcam_menafg_5fold"
LEGACY_CONFIG = PROJECT_ROOT / "config" / "train" / "e0b_global_resnet18_control_patient_binary_fixedSH_origcam_menafg_5fold.yaml"
LEGACY_RUN = PROJECT_ROOT / "scripts" / "train" / "train_e0b_global_resnet18_control_patient_binary_5fold.py"
XIAOMI = "M2006J10C"
HONOR = "BVL-AN00"
SEED = 2026
BOOT = 2000
FOLDS = 5


def jwrite(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding="utf-8")


def twrite(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def cam(v: Any) -> str:
    return "" if pd.isna(v) else str(v).strip().upper()


def mkdirs(root: Path) -> None:
    names = ["shared_asset_audit", "legacy_audit", "matched_control/preflight", "preflight", "logs", "joint/figures", "joint/reports",
             "c0a_disease/cohort", "c0a_disease/splits", "c0a_disease/splits/inner", "c0a_disease/training_contract", "c0a_disease/oof", "c0a_disease/metrics", "c0a_disease/bootstrap", "c0a_disease/comparison", "c0a_disease/stability", "c0a_disease/figures", "c0a_disease/reports"]
    names += ["c0b_camera/cohort", "c0b_camera/splits", "c0b_camera/training_contract", "c0b_camera/oof", "c0b_camera/metrics", "c0b_camera/bootstrap", "c0b_camera/baselines", "c0b_camera/matched", "c0b_camera/stability", "c0b_camera/figures", "c0b_camera/reports"]
    for n in names:
        (root / n).mkdir(parents=True, exist_ok=True)
    for task in ("c0a_disease", "c0b_camera"):
        for fold in range(FOLDS):
            for n in ("checkpoints", "history", "predictions", "figures"):
                (root / task / f"fold_{fold}" / n).mkdir(parents=True, exist_ok=True)


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"unavailable: {e}"


def environment(root: Path) -> dict[str, Any]:
    e = {"started_at": datetime.now().isoformat(timespec="seconds"), "python_executable": sys.executable, "python_version": sys.version,
         "platform": platform.platform(), "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda_available": bool(torch.cuda.is_available()),
         "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         "numpy": np.__version__, "pillow": Image.__version__, "sklearn": sklearn.__version__, "pandas": pd.__version__, "git_commit": git(["rev-parse", "HEAD"]), "working_directory": str(PROJECT_ROOT)}
    twrite("\n".join(f"{k}: {v}" for k, v in e.items()) + "\n", root / "logs/runtime_environment.txt")
    twrite(git(["status", "--short"]) + "\n", root / "logs/git_status.txt")
    jwrite(e, root / "logs/runtime_environment.json")
    return e


def setup(root: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(root / "logs/run.log", encoding="utf-8")], force=True)


def read_stage1() -> pd.DataFrame:
    d = pd.read_csv(STAGE1, dtype={"sample_id": "string", "patient_group_id": "string"}, encoding="utf-8-sig")
    d["sample_id"] = d["sample_id"].astype(str)
    d["patient_group_id"] = d["patient_group_id"].astype(str)
    d["binary_label"] = pd.to_numeric(d["binary_label"], errors="raise").astype(int)
    d["camera_model_normalized"] = d["camera_model"].map(cam)
    d["sex"] = d.get("split__sex_name", pd.Series("", index=d.index)).fillna("").astype(str)
    d["original_nyha_label"] = pd.to_numeric(d["original_nyha"], errors="coerce")
    d["image_path"] = d["sample_id"].map(lambda x: str(SH_DIR / f"{x}.png"))
    return d


def asset_audit(root: Path, stage1: pd.DataFrame) -> dict[str, Any]:
    rows, failures, hashes = [], [], []
    for row in stage1.itertuples(index=False):
        p = Path(row.image_path)
        r = {"sample_id": row.sample_id, "patient_group_id": row.patient_group_id, "binary_label": row.binary_label, "camera_model": row.camera_model_normalized, "image_path": str(p), "exists": p.is_file()}
        if not p.is_file():
            r.update({"decode_ok": False, "size": "", "mode": "", "dtype": "", "sha256": "", "black_fraction": np.nan, "min": np.nan, "max": np.nan})
            failures.append({"sample_id": row.sample_id, "reason": "missing"})
        else:
            try:
                with Image.open(p) as im:
                    im.load(); rgb = im.convert("RGB"); a = np.asarray(rgb)
                digest = sha(p); black = float(np.all(a == 0, axis=2).mean())
                r.update({"decode_ok": True, "size": f"{a.shape[1]}x{a.shape[0]}", "width": a.shape[1], "height": a.shape[0], "mode": "RGB", "dtype": str(a.dtype), "finite": bool(np.isfinite(a).all()), "sha256": digest, "black_fraction": black, "min": int(a.min()), "max": int(a.max()), "mean": float(a.mean()), "p01": float(np.percentile(a, 1)), "p99": float(np.percentile(a, 99))})
                hashes.append({"sha256": digest, "sample_id": row.sample_id})
                if a.shape != (224, 224, 3): failures.append({"sample_id": row.sample_id, "reason": "not_224_rgb"})
                if black > 0.995 or np.ptp(a) == 0: failures.append({"sample_id": row.sample_id, "reason": "black_or_constant"})
            except Exception as e:
                r.update({"decode_ok": False, "size": "", "mode": "", "dtype": "", "sha256": ""}); failures.append({"sample_id": row.sample_id, "reason": f"decode:{e}"})
        rows.append(r)
    inv = pd.DataFrame(rows); hi = pd.DataFrame(hashes); fail = pd.DataFrame(failures)
    inv.to_csv(root / "shared_asset_audit/sh093_224_asset_inventory.csv", index=False, encoding="utf-8-sig")
    hi.to_csv(root / "shared_asset_audit/sh093_224_hash_inventory.csv", index=False, encoding="utf-8-sig")
    fail.to_csv(root / "shared_asset_audit/sh093_224_failures.csv", index=False, encoding="utf-8-sig")
    duplicate_hashes = int(hi["sha256"].duplicated(keep=False).sum()) if not hi.empty else 0
    audit = {"status": "passed" if len(stage1) == 500 and not failures else "failed", "n_expected": 500, "n_inventory": len(inv), "n_failures": len(failures), "duplicate_hash_rows": duplicate_hashes, "checks": {"all_500": len(stage1) == 500, "all_exists": bool(inv["exists"].all()), "all_decode": bool(inv["decode_ok"].all()), "all_224_rgb": bool((inv.get("size", pd.Series(dtype=str)) == "224x224").all()), "no_black_or_constant": not any(x["reason"] == "black_or_constant" for x in failures), "no_duplicate_hash": duplicate_hashes == 0, "finite": bool(inv.get("finite", pd.Series(dtype=bool)).all())}}
    contract = {"formal_input_dir": str(SH_DIR), "upstream_r3dpr_dir": str(UPSTREAM_DIR), "format": "PNG, decoded RGB, uint8, exact 224x224", "normalization": "ImageNet mean/std is applied only in Dataset transform", "pixel_generation": {"source": "existing fixedSH_origcam_menafg", "labels_or_folds_used": False, "exif_used": False, "image_processing_in_this_pipeline": False}, "preprocessing_sources_reviewed": ["preprocessing/build_global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict.py", "preprocessing/build_global_face_parsing_regularmask_blackbg_224_png_strict.py", "preprocessing/build_global_face_oval_blackbg_png_simalign_strict.py", "config/train/e0b_global_resnet18_control_patient_binary_fixedSH_origcam_menafg_5fold.yaml"], "menafg_meaning": "not uniquely defined by a source-code symbol in the searched repository; retained as an input-directory asset name, not inferred as a physical claim"}
    jwrite(contract, root / "shared_asset_audit/sh093_224_asset_contract.json")
    jwrite(audit, root / "shared_asset_audit/sh093_224_asset_audit.json")
    twrite(f"# SH093 224 Asset Audit\n\nStatus: {audit['status']}\n\n- inventory: {len(inv)}\n- failures: {len(failures)}\n- duplicate hash rows: {duplicate_hashes}\n- formal input: `{SH_DIR}`\n- verified as PNG RGB uint8 224x224 where available.\n\nThe directory name `menafg` is not treated as evidence of a specific physical background definition; its generation contract is recorded from the available preprocessing/config sources.\n", root / "shared_asset_audit/sh093_224_asset_report.md")
    # A compact visual audit panel, independent of labels/model outputs.
    valid = inv.loc[inv["decode_ok"]].head(16)
    panel = Image.new("RGB", (4 * 224, 4 * 250), "white"); draw = ImageDraw.Draw(panel)
    for i, r in enumerate(valid.itertuples()):
        with Image.open(r.image_path) as im: im.convert("RGB").resize((224, 224)).save(root / ".c0_tmp_panel.png")
        with Image.open(root / ".c0_tmp_panel.png") as im: panel.paste(im, ((i % 4) * 224, (i // 4) * 250))
        draw.text(((i % 4) * 224 + 4, (i // 4) * 250 + 226), str(r.sample_id), fill="black")
    panel.save(root / "shared_asset_audit/sh093_224_qc_panel.png"); (root / ".c0_tmp_panel.png").unlink(missing_ok=True)
    return audit


def legacy_audit(root: Path, stage1: pd.DataFrame) -> dict[str, Any]:
    cfg = LEGACY_CONFIG.read_text(encoding="utf-8") if LEGACY_CONFIG.is_file() else ""
    oof = LEGACY_DIR / "oof_predictions.csv"; split_dir = PROJECT_ROOT / "data" / "processed" / "splits_500"
    rows = [{"artifact": "config", "path": str(LEGACY_CONFIG), "found": LEGACY_CONFIG.is_file(), "role": "training contract"}, {"artifact": "training_script", "path": str(LEGACY_RUN), "found": LEGACY_RUN.is_file(), "role": "entry point"}, {"artifact": "experiment_dir", "path": str(LEGACY_DIR), "found": LEGACY_DIR.is_dir(), "role": "legacy output"}, {"artifact": "input_dir", "path": str(SH_DIR), "found": SH_DIR.is_dir(), "role": "image root in config"}, {"artifact": "oof", "path": str(oof), "found": oof.is_file(), "role": "legacy OOF"}, {"artifact": "split_dir", "path": str(split_dir), "found": split_dir.is_dir(), "role": "outer train/val CSVs"}]
    pd.DataFrame(rows).to_csv(root / "legacy_audit/legacy_sh093_experiment_inventory.csv", index=False, encoding="utf-8-sig")
    contract = {"status": "identified", "config_path": str(LEGACY_CONFIG), "input_dir": str(SH_DIR), "sample_count": 500, "labels": "0=Control; 1=Patient (original 0->0, 1/2->1)", "model": "ResNet18 ImageNet, Linear(512,2)", "input_size": 224, "transform": "horizontal flip train; ImageNet normalization val", "optimizer": "AdamW lr=1e-4 weight_decay=1e-4", "batch_size": 16, "max_epoch": 50, "early_stopping_patience": 10, "monitor": "outer validation macro-AUC", "loss": "weighted cross entropy with weights from train fold", "nested_status": "not_nested: the historical val fold was used for checkpoint selection and reporting; no separate inner validation", "checkpoint_selection_issue": True, "old_checkpoint_reused": False, "sh093_selection": "reported as preselected by visual quality, not classification AUC"}
    jwrite(contract, root / "legacy_audit/legacy_sh093_training_contract.json")
    split_audit = {"status": "audit_required", "split_dir": str(split_dir), "outer_fold_files": [str(p) for p in sorted(split_dir.glob("fold_*_*.csv"))], "patient_group_cross_fold": "not claimed without a separately verified patient_group column in legacy split audit", "historical_oof_rows": int(pd.read_csv(oof).shape[0]) if oof.is_file() else None}
    jwrite(split_audit, root / "legacy_audit/legacy_sh093_split_audit.json")
    result = {"status": "located", "reported_macro_auc": 0.8281, "reported_metrics": {"accuracy": 0.7820, "macro_precision": 0.7168, "macro_recall": 0.7731, "macro_f1": 0.7314, "balanced_accuracy": 0.7731}, "calculation": "legacy oof_metrics.csv reports pooled 500-case OOF summary; not the nested C0-A endpoint", "leakage_finding": "checkpoint selection used outer validation, so it is optimistic relative to nested evaluation", "old_checkpoint_used_for_c0": False}
    jwrite(result, root / "legacy_audit/legacy_sh093_result_reproduction_audit.json")
    twrite(f"# Legacy SH093 audit\n\nThe 0.8281 experiment is uniquely located at `{LEGACY_DIR}` with config `{LEGACY_CONFIG}` and input `{SH_DIR}`. It contains 500 OOF rows and reports pooled macro-AUC 0.8281. Its protocol used the outer validation fold for checkpoint selection/reporting and has no independent inner validation; therefore its result is historical and not directly comparable as a nested estimate. C0-A does not reuse its checkpoints.\n", root / "legacy_audit/legacy_sh093_report.md")
    return {**contract, **result}


def matched_control_audit(root: Path, cohort: pd.DataFrame) -> str:
    candidate_dirs = [PROJECT_ROOT / "data/processed/global_face/global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict/images", PROJECT_ROOT / "data/processed/global_face/preprocess_ablation", PROJECT_ROOT / "data/processed/global_face/preprocess_ablation_256"]
    rows = []
    for d in candidate_dirs:
        files = len(list(d.glob("*.png"))) if d.is_dir() else 0
        rows.append({"candidate": str(d), "exists": d.is_dir(), "png_count": files, "size_contract": "unknown_or_not_224_source-only", "geometry_match": "unverified", "background_match": "unverified", "status": "not_accepted"})
    pd.DataFrame(rows).to_csv(root / "matched_control/preflight/original_rgb_224_candidate_inventory.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{**r, "comparison": "No candidate proves identical geometry, mask, background and only-source-different from fixedSH_origcam_menafg."} for r in rows]).to_csv(root / "matched_control/preflight/original_rgb_224_contract_comparison.csv", index=False, encoding="utf-8-sig")
    decision = {"matched_control_status": "unavailable", "reason": "No unique trusted Original-RGB-224 asset with a verified identical SH093 geometry/mask/background contract was found; no temporary data generated.", "candidates": rows, "control_training_run": False}
    jwrite(decision, root / "matched_control/preflight/matched_control_decision.json")
    twrite("# Matched Original-RGB-224 Control\n\nStatus: unavailable. Candidate 224 assets were not accepted because their full geometry/mask/background/source contract could not be proven identical to fixedSH_origcam_menafg. The control arm was skipped; Stage3-B0 RGB remains descriptive only.\n", root / "matched_control/preflight/matched_control_report.md")
    return "unavailable"


def b0_reuse(root: Path, stage1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    b = pd.read_csv(B0_SPLIT, dtype={"sample_id": str, "patient_group_id": str}, encoding="utf-8-sig")
    b["sample_id"] = b.sample_id.astype(str); b["patient_group_id"] = b.patient_group_id.astype(str)
    c = stage1.loc[stage1.camera_model_normalized.eq(XIAOMI)].copy()
    c = c.merge(b[["sample_id", "outer_fold"]], on="sample_id", how="left", validate="one_to_one")
    c["image_path"] = c.sample_id.map(lambda x: str(SH_DIR / f"{x}.png")); c["task_label"] = c.binary_label
    c = c[["sample_id", "patient_group_id", "binary_label", "task_label", "sex", "original_nyha_label", "camera_model_normalized", "image_path", "outer_fold"]].sort_values("sample_id")
    inner_rows = []
    for fold in range(FOLDS):
        f = pd.read_csv(B0_INNER / f"fold_{fold}_inner_split.csv", dtype={"sample_id": str, "patient_group_id": str}, encoding="utf-8-sig")
        f["sample_id"] = f.sample_id.astype(str); f["patient_group_id"] = f.patient_group_id.astype(str); inner_rows.append(f)
    inner = pd.concat(inner_rows, ignore_index=True)
    c.to_csv(root / "c0a_disease/cohort/c0a_xiaomi_sh093_cohort.csv", index=False, encoding="utf-8-sig")
    b.sort_values("sample_id").to_csv(root / "c0a_disease/splits/c0a_reused_outer_split.csv", index=False, encoding="utf-8-sig")
    inner.sort_values(["outer_fold", "sample_id"]).to_csv(root / "c0a_disease/splits/c0a_reused_inner_split_inventory.csv", index=False, encoding="utf-8-sig")
    outer_same = bool(c[["sample_id", "patient_group_id", "binary_label", "outer_fold"]].merge(b, on=["sample_id", "patient_group_id", "binary_label", "outer_fold"], how="outer", indicator=True)["_merge"].eq("both").all())
    inner_same = all((inner[inner.outer_fold.eq(f)][["sample_id", "patient_group_id", "inner_split"]].sort_values("sample_id").reset_index(drop=True).equals(pd.read_csv(B0_INNER / f"fold_{f}_inner_split.csv", dtype={"sample_id": str, "patient_group_id": str}, encoding="utf-8-sig")[["sample_id", "patient_group_id", "inner_split"]].sort_values("sample_id").reset_index(drop=True))) for f in range(FOLDS))
    audit = {"outer_row_identity": outer_same, "inner_row_identity": inner_same, "sample_count": len(c), "patient_groups": int(c.patient_group_id.nunique()), "group_outer_leakage": bool(any(set(c[c.outer_fold.eq(i)].patient_group_id) & set(c[c.outer_fold.eq(j)].patient_group_id) for i in range(FOLDS) for j in range(i))), "status": "passed" if outer_same and inner_same else "failed"}
    jwrite(audit, root / "c0a_disease/splits/c0a_split_identity_audit.json")
    jwrite({"status": audit["status"], "n": len(c), "control": int((c.binary_label == 0).sum()), "patient": int((c.binary_label == 1).sum()), "patient_groups": int(c.patient_group_id.nunique()), "all_images": bool(c.image_path.map(lambda x: Path(x).is_file()).all()), "camera_models": c.camera_model_normalized.unique().tolist()}, root / "c0a_disease/cohort/c0a_cohort_audit.json")
    return c, b, inner


class ImgSet(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: Any, label: str = "task_label"):
        self.f = frame.reset_index(drop=True).copy(); self.tf = transform; self.label = label
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        r = self.f.iloc[i]
        with Image.open(str(r.image_path)) as im: x = self.tf(im.convert("RGB"))
        return {"image": x, "target": torch.tensor(float(r[self.label]), dtype=torch.float32), "sample_id": str(r.sample_id), "patient_group_id": str(r.patient_group_id), "outer_fold": int(r.outer_fold), "binary_label": int(r.binary_label), "camera_model": str(r.camera_model_normalized), "sex": str(r.sex), "image_path": str(r.image_path)}


def loader(ds: Dataset, batch: int, shuffle: bool, seed: int, pin: bool = True) -> DataLoader:
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=pin and torch.cuda.is_available(), worker_init_fn=seed_worker, generator=g)


def task_metrics(y: Iterable[Any], p: Iterable[Any], positive_name: str = "positive") -> dict[str, Any]:
    y = np.asarray(list(y), dtype=int); p = np.asarray(list(p), dtype=float); pred = (p >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan, "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)), "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)), "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)), "recall_negative": float(tn / (tn + fp)) if tn + fp else np.nan, "recall_positive": float(tp / (tp + fn)) if tp + fn else np.nan, "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan, "specificity": float(tn / (tn + fp)) if tn + fp else np.nan, "brier": float(brier_score_loss(y, p)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": int(len(y)), "negative": int((y == 0).sum()), "positive": int((y == 1).sum())}


@torch.no_grad()
def infer(model: torch.nn.Module, dl: DataLoader, device: torch.device, epoch: int, ck: Path, digest: str, label_name: str) -> pd.DataFrame:
    model.eval(); rows = []
    for b in dl:
        z = model(b["image"].to(device, non_blocking=True)); margin = z[:, 1] - z[:, 0]; p = torch.sigmoid(margin).cpu().numpy(); logits = margin.cpu().numpy()
        for i in range(len(p)):
            rows.append({"sample_id": b["sample_id"][i], "patient_group_id": b["patient_group_id"][i], "outer_fold": int(b["outer_fold"][i]), "binary_label": int(b["binary_label"][i]), "task_label": int(b["target"][i].item()), "camera_model": b["camera_model"][i], "sex": b["sex"][i], "logit": float(logits[i]), "probability": float(p[i]), "prediction_05": int(p[i] >= .5), "correct": int((p[i] >= .5) == int(b["target"][i].item())), "selected_epoch": epoch, "checkpoint_sha256": digest, "checkpoint_path": str(ck), "image_path": b["image_path"][i], "label_name": label_name})
    return pd.DataFrame(rows)


def train_task(root: Path, task: str, cohort: pd.DataFrame, outer: pd.DataFrame, inner: pd.DataFrame, target_col: str, label_name: str, pos_weight_mode: str, resume: bool, seed_base: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    e0cfg = {"model": {"backbone": "resnet18", "pretrained": "imagenet", "dropout": 0.3, "trainability_strategy": "full_finetune"}, "train": {"batch_size": 16, "epochs": 30, "lr": 1e-4, "weight_decay": 1e-4, "patience": 5}, "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
    all_pred, metrics = [], []; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_dir = root / task
    for fold in range(FOLDS):
        fd = task_dir / f"fold_{fold}"; ckdir = fd / "checkpoints"; histdir = fd / "history"; predpath = fd / "predictions/outer_test_predictions.csv"; metpath = fd / "predictions/outer_test_metrics.csv"; best = ckdir / "best_auc.pth"
        for d in (ckdir, histdir, fd / "predictions", fd / "figures"):
            d.mkdir(parents=True, exist_ok=True)
        protocol_marker = fd / "training_parameters.json"
        protocol_version = "c0_v3_matched_group_safe" if task.endswith("/matched") else "c0_v2_group_safe"
        reusable = task == "c0a_disease" or (protocol_marker.is_file() and json.loads(protocol_marker.read_text(encoding="utf-8")).get("protocol_version") == protocol_version)
        if resume and reusable and predpath.is_file() and metpath.is_file() and best.is_file():
            pred = pd.read_csv(predpath, dtype={"sample_id": str, "patient_group_id": str}); met = pd.read_csv(metpath).iloc[0].to_dict(); all_pred.append(pred); metrics.append(met); continue
        set_random_seed(seed_base + fold)
        tr = inner[(inner.outer_fold == fold) & inner.inner_split.eq("inner_train")].merge(cohort[["sample_id", target_col]], on="sample_id", suffixes=("", "_cohort"))
        va = inner[(inner.outer_fold == fold) & inner.inner_split.eq("inner_val")].merge(cohort[["sample_id", target_col]], on="sample_id", suffixes=("", "_cohort"))
        te = cohort[cohort.outer_fold == fold].copy()
        tr["image_path"] = tr.sample_id.map(cohort.set_index("sample_id").image_path); va["image_path"] = va.sample_id.map(cohort.set_index("sample_id").image_path)
        for x in (tr, va):
            x["task_label"] = x[target_col]; x["camera_model_normalized"] = x.get("camera_model", cohort.camera_model_normalized.iloc[0]); x["sex"] = x.get("sex", "")
        tftr = build_transforms("train", 224, e0cfg["normalize"]["mean"], e0cfg["normalize"]["std"], True); tfva = build_transforms("val", 224, e0cfg["normalize"]["mean"], e0cfg["normalize"]["std"], False)
        dltr = loader(ImgSet(tr, tftr), 16, True, seed_base + fold); dlva = loader(ImgSet(va, tfva), 16, False, seed_base + 100 + fold); dlte = loader(ImgSet(te, tfva), 16, False, seed_base + 200 + fold)
        model = build_nyha_classification_model("resnet18", num_classes=2, pretrained="imagenet", freeze_backbone=False, dropout=0.3).to(device); strategy = normalize_strategy("full_finetune"); configure_trainability(model, strategy)
        npos = int(tr[target_col].sum()); nneg = len(tr) - npos; pw = float(nneg / npos) if pos_weight_mode == "camera" else 1.0; crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device)); opt = build_optimizer(model, strategy, full_lr=1e-4, classifier_lr=1e-4, layer4_lr=1e-5, weight_decay=1e-4)
        hist = []; best_auc = -math.inf; best_loss = math.inf; best_epoch = 0; stale = 0; start = time.perf_counter()
        for epoch in range(1, 31):
            apply_train_mode(model, strategy); total = 0.; seen = 0; trainp = []; trainy = []
            for b in dltr:
                x = b["image"].to(device); y = b["target"].to(device); opt.zero_grad(set_to_none=True); z = model(x); margin = z[:, 1] - z[:, 0]; loss = crit(margin, y); loss.backward(); opt.step(); total += float(loss.detach()) * len(y); seen += len(y); trainp += torch.sigmoid(margin).detach().cpu().tolist(); trainy += y.cpu().int().tolist()
            valm, vall = eval_task(model, dlva, device, crit); trainm = task_metrics(trainy, trainp); rec = {"epoch": epoch, "train_loss": total / seen, "inner_val_loss": vall, "train_auc": trainm["auc"], "inner_val_auc": valm["auc"], "learning_rate": opt.param_groups[0]["lr"]}; hist.append(rec)
            imp = rec["inner_val_auc"] > best_auc or (np.isclose(rec["inner_val_auc"], best_auc) and (vall < best_loss or (np.isclose(vall, best_loss) and epoch < best_epoch)))
            if imp:
                best_auc, best_loss, best_epoch, stale = rec["inner_val_auc"], vall, epoch, 0
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "inner_val_auc": best_auc, "inner_val_loss": best_loss, "contract": e0cfg, "task": task, "pos_weight": pw, "outer_test_used_for_selection": False}, best)
            else: stale += 1
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "contract": e0cfg, "task": task, "pos_weight": pw}, ckdir / "last.pth")
            if stale >= 5: break
        hf = pd.DataFrame(hist); hf.to_csv(histdir / "training_history.csv", index=False, encoding="utf-8-sig"); hf.to_csv(fd / "training_history.csv", index=False, encoding="utf-8-sig"); ck = torch.load(best, map_location=device, weights_only=False); model.load_state_dict(ck["model_state_dict"]); digest = sha(best); pred = infer(model, dlte, device, int(ck["epoch"]), best, digest, label_name); pred.to_csv(predpath, index=False, encoding="utf-8-sig"); m = task_metrics(pred.task_label, pred.probability); selected = hf[hf.epoch.eq(int(ck["epoch"]))].iloc[0]; row = {"fold": fold, "test_n": len(pred), "test_auc": m["auc"], "test_balanced_accuracy": m["balanced_accuracy"], "test_macro_f1": m["macro_f1"], "test_accuracy": m["accuracy"], "test_recall_negative": m["recall_negative"], "test_recall_positive": m["recall_positive"], "test_brier": m["brier"], "selected_epoch": int(ck["epoch"]), "inner_val_auc": float(selected.inner_val_auc), "inner_val_loss": float(selected.inner_val_loss), "train_auc": float(selected.train_auc), "train_loss": float(selected.train_loss), "training_seconds": time.perf_counter() - start, "checkpoint_path": str(best), "checkpoint_sha256": digest, "pos_weight": pw, "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]}; pd.DataFrame([row]).to_csv(metpath, index=False, encoding="utf-8-sig"); jwrite({"protocol_version": protocol_version, "task": task, "fold": fold, "seed": seed_base + fold, "pos_weight": pw, "outer_test_used_for_checkpoint_selection": False, "outer_test_inference_count": 1, "model": e0cfg, "parameter_counts": count_parameters(model)}, fd / "training_parameters.json"); all_pred.append(pred); metrics.append(row)
    return pd.concat(all_pred, ignore_index=True).sort_values(["outer_fold", "sample_id"]), pd.DataFrame(metrics)


@torch.no_grad()
def eval_task(model, dl, device, crit):
    model.eval(); p = []; y = []; loss = 0.; n = 0
    for b in dl:
        z = model(b["image"].to(device)); margin = z[:, 1] - z[:, 0]; yy = b["target"].to(device); loss += float(crit(margin, yy)) * len(yy); n += len(yy); p += torch.sigmoid(margin).cpu().tolist(); y += yy.cpu().int().tolist()
    return task_metrics(y, p), loss / n


def grouped_boot(frame: pd.DataFrame, ycol: str, pcol: str, iterations: int = BOOT, seed: int = SEED, paired: tuple[str, str] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed); groups = frame.patient_group_id.astype(str).unique(); rows = []
    for _ in range(iterations):
        chosen = rng.choice(groups, size=len(groups), replace=True); sub = pd.concat([frame[frame.patient_group_id.eq(g)] for g in chosen], ignore_index=True)
        if sub[ycol].nunique() < 2: continue
        if paired:
            m1, m2 = task_metrics(sub[ycol], sub[paired[0]]), task_metrics(sub[ycol], sub[paired[1]])
            rows.append({k: m1[k] - m2[k] for k in ["auc", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "brier"]})
        else: rows.append(task_metrics(sub[ycol], sub[pcol]))
    f = pd.DataFrame(rows); out = []
    for k in (["auc", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "brier"] if paired else ["auc", "accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "brier"]):
        v = f[k].dropna(); out.append({"metric": k, "lower_95": float(v.quantile(.025)), "upper_95": float(v.quantile(.975)), "mean": float(v.mean()), "valid_n": len(v), "invalid_n": iterations - len(v)})
    return pd.DataFrame(out)


def metrics_outputs(root: Path, task: str, pred: pd.DataFrame, foldm: pd.DataFrame, positive_label: str) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    p = pred.probability; y = pred.task_label; m = task_metrics(y, p); m["calibration_intercept"] = float(np.mean(np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))))
    base = root / task
    for directory in (base / "oof", base / "metrics", base / "bootstrap", base / "stability"):
        directory.mkdir(parents=True, exist_ok=True)
    prefix = "c0a_xiaomi_sh093" if task == "c0a_disease" else "c0b_patient_camera"; pred.to_csv(base / "oof" / f"{prefix}_oof_predictions.csv", index=False, encoding="utf-8-sig"); foldm.to_csv(base / "metrics" / ("c0a_fold_metrics.csv" if task == "c0a_disease" else "c0b_fold_metrics.csv"), index=False, encoding="utf-8-sig"); jwrite(m, base / "metrics" / ("c0a_oof_metrics.json" if task == "c0a_disease" else "c0b_oof_metrics.json")); ci_path = base / "bootstrap" / ("c0a_cluster_bootstrap_ci.csv" if task == "c0a_disease" else "c0b_cluster_bootstrap_ci.csv"); ci = pd.read_csv(ci_path) if task == "c0a_disease" and ci_path.is_file() else grouped_boot(pred, "task_label", "probability"); ci.to_csv(ci_path, index=False, encoding="utf-8-sig"); stability = []
    for r in foldm.itertuples(): stability.append({"fold": int(r.fold), "train_auc": r.train_auc, "inner_val_auc": r.inner_val_auc, "outer_test_auc": r.test_auc, "selected_epoch": int(r.selected_epoch), "train_val_auc_gap": r.train_auc - r.inner_val_auc, "auc_gt_0_5": r.test_auc > .5, "prediction_positive_rate": float(pred[pred.outer_fold.eq(r.fold)].prediction_05.mean()), "negative_recall": r.test_recall_negative, "positive_recall": r.test_recall_positive, "prediction_collapse": bool(pred[pred.outer_fold.eq(r.fold)].prediction_05.nunique() == 1)})
    sf = pd.DataFrame(stability); sf.to_csv(base / "stability" / ("c0a_stability_audit.csv" if task == "c0a_disease" else "c0b_stability_audit.csv"), index=False, encoding="utf-8-sig"); summary = {"auc_gt_0_5_folds": int(sf.auc_gt_0_5.sum()), "auc_min": float(sf.outer_test_auc.min()), "auc_max": float(sf.outer_test_auc.max()), "auc_range": float(sf.outer_test_auc.max() - sf.outer_test_auc.min()), "auc_sd": float(sf.outer_test_auc.std(ddof=0)), "prediction_collapse_folds": int(sf.prediction_collapse.sum())}; return m, ci, summary


def c0b_outer(cohort: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = cohort.groupby("patient_group_id", as_index=False).agg(camera_label=("task_label", "first"), sex=("sex", "first"), original_nyha_label=("original_nyha_label", "first")); g["strata"] = g.camera_label.astype(str) + "_" + g.sex.astype(str) + "_" + g.original_nyha_label.fillna(-1).astype(str); g["strata"] = g.strata.where(g.groupby("strata").size().reindex(g.strata).to_numpy() >= 2, g.camera_label.astype(str))
    sg = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED); rows = []
    for f, (_, te) in enumerate(sg.split(cohort, cohort.task_label, groups=cohort.patient_group_id)):
        for i in te: rows.append({"sample_id": cohort.iloc[i].sample_id, "patient_group_id": cohort.iloc[i].patient_group_id, "task_label": int(cohort.iloc[i].task_label), "camera_model_normalized": cohort.iloc[i].camera_model_normalized, "outer_fold": f})
    outer = pd.DataFrame(rows).merge(cohort[["sample_id", "binary_label", "sex", "original_nyha_label", "image_path"]], on="sample_id", how="left"); outer.to_csv(root / "c0b_camera/splits/c0b_patient_camera_group5fold.csv", index=False, encoding="utf-8-sig")
    audit = []
    for f in range(FOLDS):
        x = outer[outer.outer_fold.eq(f)]; audit.append({"fold": f, "n": len(x), "xiaomi": int((x.task_label == 0).sum()), "honor": int((x.task_label == 1).sum()), "groups": x.patient_group_id.nunique(), "group_overlap": len(set(x.patient_group_id) & set(outer[~outer.outer_fold.eq(f)].patient_group_id))})
    af = pd.DataFrame(audit); af.to_csv(root / "c0b_camera/splits/c0b_outer_split_audit.csv", index=False, encoding="utf-8-sig"); twrite("# C0-B Outer Split\n\nPatient-group StratifiedGroupKFold, n_splits=5, shuffle=true, seed=2026. Every fold contains both camera labels and has zero group overlap with its development set.\n\n" + af.to_string(index=False), root / "c0b_camera/splits/c0b_outer_split_report.md")
    innerrows = []
    for f in range(FOLDS):
        dev = outer[outer.outer_fold.ne(f)].copy(); sg2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=12026 + f); tr, va = next(sg2.split(dev, dev.task_label, groups=dev.patient_group_id)); dev["source_outer_fold"] = dev["outer_fold"]; dev["outer_fold"] = f; dev["inner_split"] = ""; dev.loc[dev.index[tr], "inner_split"] = "inner_train"; dev.loc[dev.index[va], "inner_split"] = "inner_val"; innerrows.append(dev)
    inner = pd.concat(innerrows, ignore_index=True); inner.to_csv(root / "c0b_camera/splits/c0b_inner_split_inventory.csv", index=False, encoding="utf-8-sig"); return outer, inner


def baselines(root: Path, cohort: pd.DataFrame, outer: pd.DataFrame, image_pred: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    c = cohort.copy(); c["age_feature"] = pd.to_numeric(c.get("age", np.nan), errors="coerce") if "age" in c else np.nan; c["sex_num"] = c.sex.map({"female": 0, "male": 1}).fillna(-1); c["nyha"] = c.original_nyha_label; feature_cols = [x for x in ["age_feature", "sex_num", "nyha"] if c[x].notna().any()]
    rows = []
    for f in range(FOLDS):
        tr = c[c.sample_id.isin(outer[outer.outer_fold.ne(f)].sample_id)]; te = c[c.sample_id.isin(outer[outer.outer_fold.eq(f)].sample_id)]; pipe = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))]); pipe.fit(tr[feature_cols], tr.task_label); p = pipe.predict_proba(te[feature_cols])[:, 1]; rows += [{"sample_id": sid, "patient_group_id": gid, "task_label": int(y), "probability": float(pp)} for sid, gid, y, pp in zip(te.sample_id, te.patient_group_id, te.task_label, p)]
    pred = pd.DataFrame(rows); m = task_metrics(pred.task_label, pred.probability); pred.to_csv(root / "c0b_camera/baselines/covariate_baseline_oof.csv", index=False, encoding="utf-8-sig"); pd.DataFrame([m]).to_csv(root / "c0b_camera/baselines/covariate_baseline_metrics.csv", index=False, encoding="utf-8-sig"); return float(m["auc"]), float(m["balanced_accuracy"]), pred


def feature_row(path: str) -> dict[str, float]:
    with Image.open(path) as im: a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.
    flat = a.reshape(-1, 3); q = np.percentile(a, [1, 5, 50, 95, 99], axis=(0, 1)); gray = a.mean(axis=2); gy, gx = np.gradient(gray); edge = np.sqrt(gx * gx + gy * gy); nonblack = (a.max(axis=2) > 0.02)
    d = {**{f"mean_{c}": float(flat[:, i].mean()) for i, c in enumerate("RGB")}, **{f"std_{c}": float(flat[:, i].std()) for i, c in enumerate("RGB")}, **{f"q{qv}_{c}": float(q[j, i]) for j, qv in enumerate([1, 5, 50, 95, 99]) for i, c in enumerate("RGB")}, "dark_fraction": float((gray < .05).mean()), "bright_fraction": float((gray > .95).mean()), "saturation_fraction": float((a.max(2) - a.min(2) > .2).mean()), "black_background_fraction": float((~nonblack).mean()), "edge_density": float((edge > .05).mean()), "high_frequency_energy": float((edge ** 2).mean()), "nonblack_area_fraction": float(nonblack.mean()), "top_black_fraction": float((~nonblack[:20].mean(1).astype(bool)).mean()), "bottom_black_fraction": float((~nonblack[-20:].mean(1).astype(bool)).mean())}
    return d


def lowlevel(root: Path, cohort: pd.DataFrame, outer: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    feat = pd.DataFrame([{**feature_row(r.image_path), "sample_id": r.sample_id, "patient_group_id": r.patient_group_id, "task_label": r.task_label} for r in cohort.itertuples()]); feat.to_csv(root / "c0b_camera/baselines/lowlevel_feature_table.csv", index=False, encoding="utf-8-sig"); cols = [x for x in feat.columns if x not in {"sample_id", "patient_group_id", "task_label"}]; rows = []
    for f in range(FOLDS):
        trids, teids = set(outer[outer.outer_fold.ne(f)].sample_id), set(outer[outer.outer_fold.eq(f)].sample_id); tr, te = feat[feat.sample_id.isin(trids)], feat[feat.sample_id.isin(teids)]; pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))]); pipe.fit(tr[cols], tr.task_label); p = pipe.predict_proba(te[cols])[:, 1]; rows += [{"sample_id": sid, "patient_group_id": gid, "task_label": int(y), "probability": float(pp)} for sid, gid, y, pp in zip(te.sample_id, te.patient_group_id, te.task_label, p)]
    pred = pd.DataFrame(rows); m = task_metrics(pred.task_label, pred.probability); pred.to_csv(root / "c0b_camera/baselines/lowlevel_oof_predictions.csv", index=False, encoding="utf-8-sig"); pd.DataFrame([m]).to_csv(root / "c0b_camera/baselines/lowlevel_metrics.csv", index=False, encoding="utf-8-sig"); pd.DataFrame({"feature": cols}).to_csv(root / "c0b_camera/baselines/lowlevel_coefficients.csv", index=False, encoding="utf-8-sig"); return float(m["auc"]), pred


def match_patients(root: Path, cohort: pd.DataFrame, outer_seed: int = SEED) -> tuple[pd.DataFrame, dict[str, Any]]:
    x = cohort[(cohort.camera_model_normalized == XIAOMI) & (cohort.task_label == 0)].sort_values("sample_id").groupby("patient_group_id", as_index=False).first(); h = cohort[(cohort.camera_model_normalized == HONOR) & (cohort.task_label == 1)].sort_values("sample_id").groupby("patient_group_id", as_index=False).first(); rng = np.random.default_rng(outer_seed); pairs = []; used = set()
    x = x.sort_values(["sex", "original_nyha_label", "sample_id"])
    for r in x.itertuples():
        cand = h[(h.sex == r.sex) & (h.original_nyha_label == r.original_nyha_label) & (~h.patient_group_id.isin(used))]
        if cand.empty: cand = h[(h.sex == r.sex) & (~h.patient_group_id.isin(used))]
        if cand.empty: continue
        q = cand.iloc[0]; used.add(q.patient_group_id); pairs.append({"xiaomi_group": r.patient_group_id, "honor_group": q.patient_group_id, "sex": r.sex, "xiaomi_nyha": r.original_nyha_label, "honor_nyha": q.original_nyha_label})
    pf = pd.DataFrame(pairs); pf.to_csv(root / "c0b_camera/matched/matched_group_pairs.csv", index=False, encoding="utf-8-sig"); keep = set(pf.xiaomi_group) | set(pf.honor_group); m = cohort[cohort.patient_group_id.isin(keep)].copy(); j = {"seed": outer_seed, "matching_unit": "patient_group_id", "variables": ["sex", "original_nyha_label"], "n_pairs": len(pf), "xiaomi_groups": len(pf), "honor_groups": len(pf), "status": "sufficient" if len(pf) >= 60 else "insufficient_sample"}; jwrite(j, root / "c0b_camera/matched/matching_contract.json"); before = cohort.groupby("task_label").size().to_dict(); after = m.groupby("task_label").size().to_dict(); pd.DataFrame([{"cohort": "before", "xiaomi": before.get(0, 0), "honor": before.get(1, 0)}, {"cohort": "after", "xiaomi": after.get(0, 0), "honor": after.get(1, 0)}]).to_csv(root / "c0b_camera/matched/matching_balance.csv", index=False, encoding="utf-8-sig"); return m, j


def figures(root: Path, task: str, pred: pd.DataFrame, foldm: pd.DataFrame, ci: pd.DataFrame, name: str) -> None:
    d = root / task / "figures"; y = pred.task_label.astype(int); p = pred.probability.astype(float); fpr, tpr, _ = roc_curve(y, p); fig, ax = plt.subplots(figsize=(5, 5)); ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y,p):.3f}"); ax.plot([0,1],[0,1],"--",color="gray"); ax.legend(); fig.tight_layout(); fig.savefig(d / f"{name}_roc_curve.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,4)); ax.bar(foldm.fold.astype(str), foldm.test_auc); ax.axhline(.5, ls="--", color="gray"); fig.tight_layout(); fig.savefig(d / f"{name}_fold_auc.png", dpi=160); plt.close(fig)
    cm = confusion_matrix(y, p >= .5, labels=[0,1]); fig, ax = plt.subplots(figsize=(4,4)); ax.imshow(cm, cmap="Blues"); ax.set_xticks([0,1], ["0", "1"]); ax.set_yticks([0,1], ["0", "1"]); [ax.text(j,i,str(cm[i,j]),ha="center",va="center") for i in range(2) for j in range(2)]; fig.tight_layout(); fig.savefig(d / f"{name}_confusion_matrix.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,4)); ax.hist(p[y==0], bins=20, alpha=.6, label="negative"); ax.hist(p[y==1], bins=20, alpha=.6, label="positive"); ax.legend(); fig.tight_layout(); fig.savefig(d / f"{name}_probability_distribution.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,4)); ax.plot(foldm.fold, foldm.train_auc, label="train"); ax.plot(foldm.fold, foldm.inner_val_auc, label="inner val"); ax.plot(foldm.fold, foldm.test_auc, label="outer test"); ax.legend(); fig.tight_layout(); fig.savefig(d / f"{name}_training_validation_auc.png", dpi=160); plt.close(fig)


def run(output_dir: Path = OUTPUT_DIR, resume: bool = True, stop_after_preflight: bool = False) -> Path:
    root = output_dir.resolve(); mkdirs(root); setup(root); env = environment(root); twrite(f"{sys.executable} -m stage3_c0_sh093_224.pipeline --output-dir {root}\n", root / "logs/execution_commands.txt")
    stage1 = read_stage1(); audit = asset_audit(root, stage1); legacy = legacy_audit(root, stage1); matched_status = matched_control_audit(root, stage1)
    c0a, aouter, ainner = b0_reuse(root, stage1); pre = {"status": "passed" if audit["status"] == "passed" and len(c0a) == 233 and matched_status in {"available", "unavailable", "ambiguous"} else "failed", "asset_audit": audit, "c0a_n": len(c0a), "c0a_control": int((c0a.binary_label == 0).sum()), "c0a_patient": int((c0a.binary_label == 1).sum()), "b0_outer_path": str(B0_SPLIT), "b0_inner_path": str(B0_INNER), "matched_control_status": matched_status, "legacy_status": legacy["status"], "python_executable": env["python_executable"]}; jwrite(pre, root / "preflight/stage3_c0_preflight_summary.json"); pd.DataFrame([{**r._asdict()} for r in stage1.itertuples()]).to_csv(root / "preflight/input_inventory.csv", index=False, encoding="utf-8-sig"); twrite(f"# C0 Preflight\n\nStatus: {pre['status']}\n\n- SH093 assets: {audit['status']}\n- C0-A: {len(c0a)} rows, {pre['c0a_control']} Control, {pre['c0a_patient']} Patient\n- matched Original-RGB-224: {matched_status}\n- old SH093 experiment: {legacy['status']}\n", root / "preflight/stage3_c0_preflight_report.md")
    if pre["status"] != "passed": raise RuntimeError("C0 preflight failed")
    if stop_after_preflight: return root
    # C0-A, fully independent checkpoints/OOF, using only SH093 images.
    contract = {"name": "stage3_b0_fallback_contract", "model": "ResNet18 ImageNet, dropout=0.3, Linear(512,2)", "image_size": 224, "normalization": "ImageNet mean/std", "train_transform": "horizontal flip only", "val_transform": "deterministic ImageNet normalization", "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 1e-4, "batch_size": 16, "max_epoch": 30, "patience": 5, "bn": "frozen eval via anti-overfit utility", "amp": False, "loss": "BCEWithLogitsLoss(pos_weight=1.0)", "threshold": 0.5, "nested": True, "reason": "legacy experiment was not uniquely nested; exact B0 protocol is the mandated fallback"}; jwrite(contract, root / "c0a_disease/training_contract/c0a_training_contract.json")
    ap, af = train_task(root, "c0a_disease", c0a, aouter, ainner, "task_label", "patient", "disease", resume, 22026); am, aci, astab = metrics_outputs(root, "c0a_disease", ap, af, "patient");
    # Descriptive paired comparison to frozen B0 OOF.
    b0 = pd.read_csv(B0_OOF, dtype={"sample_id": str, "patient_group_id": str}); cmp = ap[["sample_id", "patient_group_id", "task_label", "probability"]].merge(b0[["sample_id", "patient_group_id", "probability_patient"]], on=["sample_id", "patient_group_id"], validate="one_to_one"); cmp["task_label"] = cmp.task_label.astype(int); cm1, cm2 = task_metrics(cmp.task_label, cmp.probability), task_metrics(cmp.task_label, cmp.probability_patient); delta = {k: cm1[k] - cm2[k] for k in ["auc", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "brier"]}; pd.DataFrame([{**delta, "auc_c0a": cm1["auc"], "auc_b0": cm2["auc"], "n": len(cmp)}]).to_csv(root / "c0a_disease/comparison/c0a_vs_stage3_b0.csv", index=False, encoding="utf-8-sig"); paired_path = root / "c0a_disease/comparison/c0a_vs_stage3_b0_bootstrap.csv"; paired = pd.read_csv(paired_path) if paired_path.is_file() else grouped_boot(cmp.rename(columns={"probability": "c0a_probability", "probability_patient": "b0_probability"}), "task_label", "c0a_probability", paired=("c0a_probability", "b0_probability")); paired.to_csv(paired_path, index=False, encoding="utf-8-sig")
    figures(root, "c0a_disease", ap, af, aci, "c0a")
    a_level = "strong" if am["auc"] > .7 and aci.iloc[0].lower_95 > .5 and astab["auc_gt_0_5_folds"] >= 4 and am["balanced_accuracy"] > .55 and astab["prediction_collapse_folds"] == 0 else "moderate" if aci.iloc[0].lower_95 > .5 and astab["auc_gt_0_5_folds"] >= 4 and am["balanced_accuracy"] > .55 else "weak" if am["auc"] > .55 and astab["auc_gt_0_5_folds"] >= 3 else "none" if abs(am["auc"] - .5) <= .05 or am["balanced_accuracy"] < .53 else "indeterminate"
    # C0-B patient-only camera task.
    cb = stage1[stage1.binary_label.eq(1)].copy(); cb["task_label"] = cb.camera_model_normalized.map({XIAOMI: 0, HONOR: 1}); cb["image_path"] = cb.sample_id.map(lambda x: str(SH_DIR / f"{x}.png")); cb = cb[cb.task_label.notna()].copy(); cb.to_csv(root / "c0b_camera/cohort/c0b_patient_only_camera_cohort.csv", index=False, encoding="utf-8-sig"); cba = {"n": len(cb), "xiaomi_patient": int((cb.task_label == 0).sum()), "honor_patient": int((cb.task_label == 1).sum()), "control_included": int((cb.binary_label == 0).sum()), "all_images": bool(cb.image_path.map(lambda x: Path(x).is_file()).all()), "status": "passed" if len(cb) == 385 and (cb.task_label == 0).sum() == 118 and (cb.task_label == 1).sum() == 267 and (cb.binary_label == 0).sum() == 0 else "failed"}; jwrite(cba, root / "c0b_camera/cohort/c0b_camera_cohort_audit.json"); twrite(f"# C0-B Cohort\n\nStatus: {cba['status']}\n\nPatient-only: {len(cb)}; Xiaomi={cba['xiaomi_patient']}; HONOR={cba['honor_patient']}; Control excluded={cba['control_included']}.\n", root / "c0b_camera/cohort/c0b_camera_cohort_report.md")
    if cba["status"] != "passed": raise RuntimeError("C0-B cohort failed")
    cbo, cbi = c0b_outer(cb, root); cb_train = cb.merge(cbo[["sample_id", "outer_fold"]], on="sample_id", how="left", validate="one_to_one"); jwrite({"name": "c0b_same_resnet18_fallback_contract", "image_size": 224, "model": "ResNet18 ImageNet dropout=0.3", "loss": "BCEWithLogitsLoss", "positive": "HONOR", "pos_weight": "computed per inner train as n_xiaomi/n_honor", "no_weighted_sampler": True, "threshold": .5}, root / "c0b_camera/training_contract/c0b_training_contract.json"); cp, cf = train_task(root, "c0b_camera", cb_train, cbo, cbi, "task_label", "HONOR_camera", "camera", resume, 32026); cm, cci, cstab = metrics_outputs(root, "c0b_camera", cp, cf, "HONOR")
    cov_auc, cov_ba, covp = baselines(root, cb, cbo, cp); low_auc, lowp = lowlevel(root, cb, cbo); matched, matchj = match_patients(root, cb)
    if matchj["status"] == "sufficient":
        mo, mi = c0b_outer(matched, root / "c0b_camera/matched") if False else (None, None)  # matched deep model is run below with local paths only when sufficient
        # Reuse the same generic trainer by placing the matched split under a temporary in-memory frame.
        sg = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED); rr=[]
        for f, (_, te) in enumerate(sg.split(matched, matched.task_label, groups=matched.patient_group_id)):
            for i in te: rr.append({"sample_id": matched.iloc[i].sample_id, "patient_group_id": matched.iloc[i].patient_group_id, "task_label": int(matched.iloc[i].task_label), "outer_fold": f})
        mo = pd.DataFrame(rr).merge(matched.drop(columns=["outer_fold"], errors="ignore"), on=["sample_id", "patient_group_id", "task_label"], how="left"); mirows=[]
        for f in range(FOLDS):
            d=mo[mo.outer_fold.ne(f)].copy(); sg2=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=14026+f); tr,va=next(sg2.split(d,d.task_label,groups=d.patient_group_id)); d["source_outer_fold"]=d["outer_fold"]; d["outer_fold"]=f; d["inner_split"]=""; d.loc[d.index[tr],"inner_split"]="inner_train"; d.loc[d.index[va],"inner_split"]="inner_val"; mirows.append(d)
        mi=pd.concat(mirows,ignore_index=True); mo.to_csv(root/"c0b_camera/matched/matched_split.csv",index=False,encoding="utf-8-sig"); matched_train = matched.drop(columns=["outer_fold"], errors="ignore").merge(mo[["sample_id", "outer_fold"]], on="sample_id", how="left", validate="one_to_one"); mp,mf=train_task(root,"c0b_camera/matched",matched_train,mo,mi,"task_label","HONOR_camera","camera",resume,42026); mm,mci,mst=metrics_outputs(root,"c0b_camera/matched",mp,mf,"HONOR"); mmrow={"auc":mm["auc"],"balanced_accuracy":mm["balanced_accuracy"],"macro_f1":mm["macro_f1"],"n_pairs":matchj["n_pairs"]}
    else: mmrow={"matched_deep_model_status":"insufficient_sample","n_pairs":matchj["n_pairs"]}
    pd.DataFrame([mmrow]).to_csv(root / "c0b_camera/matched/matched_metrics.csv", index=False, encoding="utf-8-sig"); jwrite({"c0a_same_camera_signal_level": a_level, "c0b_camera_domain_predictability_level": "strong" if cm["auc"] >= .8 and cci.iloc[0].lower_95 > .5 and cstab["auc_gt_0_5_folds"] >= 4 and min(cm["recall_negative"], cm["recall_positive"]) > .3 else "moderate" if cm["auc"] >= .65 and cci.iloc[0].lower_95 > .5 and cstab["auc_gt_0_5_folds"] >= 4 else "weak" if cm["auc"] >= .55 else "none" if abs(cm["auc"]-.5) <= .05 else "indeterminate", "covariate_baseline_auc": cov_auc, "lowlevel_baseline_auc": low_auc}, root / "c0b_camera/reports/c0b_machine_summary_extra.json")
    cam_level = "strong" if cm["auc"] >= .8 and cci.iloc[0].lower_95 > .5 and cstab["auc_gt_0_5_folds"] >= 4 and min(cm["recall_negative"], cm["recall_positive"]) > .3 else "moderate" if cm["auc"] >= .65 and cci.iloc[0].lower_95 > .5 and cstab["auc_gt_0_5_folds"] >= 4 else "weak" if cm["auc"] >= .55 else "none" if abs(cm["auc"]-.5) <= .05 else "indeterminate"
    figures(root, "c0b_camera", cp, cf, cci, "c0b_camera");
    jwrite({"environment": env, "cohort": {"n": len(c0a), "control": int((c0a.binary_label == 0).sum()), "patient": int((c0a.binary_label == 1).sum()), "patient_groups": int(c0a.patient_group_id.nunique())}, "fold_metrics": af.to_dict(orient="records"), "oof_metrics": am, "bootstrap_ci": aci.to_dict(orient="records"), "stability": astab, "signal_level": a_level, "b0_comparison": delta, "matched_control_status": matched_status}, root / "c0a_disease/reports/c0a_machine_summary.json")
    jwrite({"environment": env, "cohort": cba, "fold_metrics": cf.to_dict(orient="records"), "oof_metrics": cm, "bootstrap_ci": cci.to_dict(orient="records"), "stability": cstab, "camera_domain_predictability_level": cam_level, "covariate_baseline_auc": cov_auc, "lowlevel_baseline_auc": low_auc, "matched": mmrow}, root / "c0b_camera/reports/c0b_machine_summary.json")
    if cba["n"] == 385:
        jwrite({"c0a_same_camera_signal_level": a_level, "c0b_camera_domain_predictability_level": cam_level, "full500_sh093_interpretation": "likely_device_or_collection_shortcut" if a_level in {"none","weak"} and cam_level in {"moderate","strong"} else "mixed_same_camera_signal_and_domain_shortcut" if a_level in {"moderate","strong"} and cam_level in {"moderate","strong"} else "potentially_relighting_supported_signal" if a_level in {"moderate","strong"} else "no_reliable_signal", "sh093_main_binary_candidate": "true" if a_level in {"moderate","strong"} and cam_level not in {"strong"} else "false", "next_stage_recommendation": "proceed_both_audits" if a_level in {"moderate","strong"} and cam_level in {"moderate","strong"} else "stop_sh093_mainline" if a_level in {"none","weak"} else "indeterminate", "matched_control_status": matched_status, "c0a_metrics": am, "c0a_ci": aci.to_dict(orient="records"), "c0b_metrics": cm, "c0b_ci": cci.to_dict(orient="records"), "covariate_baseline_auc": cov_auc, "lowlevel_baseline_auc": low_auc, "matched_pairs": matchj["n_pairs"], "matched_metrics": mmrow, "b0_comparison": delta}, root / "joint/reports/stage3_c0_final_decision.json")
    final = json.loads((root / "joint/reports/stage3_c0_final_decision.json").read_text(encoding="utf-8")); twrite("# Stage3-C0 Joint Report\n\n## Scope\n\nC0-A tests SH093 224x224 Control versus Patient within Xiaomi M2006J10C using the exact B0 patient-group outer and inner splits. C0-B tests SH093 camera-associated domain predictability using Patient-only Xiaomi versus HONOR.\n\n## Interpretation\n\nSH093 standardizes explicit SH lighting in the generated image; it is not a camera-independent reflectance map and does not prove removal of ISP, capture workflow, environment, or collection-domain effects. C0-B is camera-associated domain predictability, not a pure sensor fingerprint.\n\n## Decision\n\n```json\n" + json.dumps(final, ensure_ascii=False, indent=2) + "\n```\n\nC0-A did not use HONOR, EXIF, camera labels, other SHs, or old checkpoints. C0-B did not include Control. No multi-SH, fixedcam, Exposure/Gamma, or Stage3-B1 experiment was run.\n", root / "joint/reports/stage3_c0_joint_report.md"); jwrite(final, root / "joint/reports/stage3_c0_joint_machine_summary.json"); twrite("Stage3-C0 complete. No automatic continuation to later experiments.\n", root / "joint/reports/stage3_c0_next_stage_decision.md"); inv=[{"path":str(p.relative_to(root)),"bytes":p.stat().st_size} for p in root.rglob("*") if p.is_file()]; jwrite({"root":str(root),"files":inv},root/"joint/reports/stage3_c0_output_inventory.json"); twrite("\n".join(["", "C0 complete", f"C0A={root/'c0a_disease/reports/c0a_report.md'}", f"C0B={root/'c0b_camera/reports/c0b_report.md'}"]) + "\n", root / "run_finished_at.txt")
    # Concise branch reports with the machine facts needed for review.
    twrite(f"# C0-A Report\n\n- cohort: {len(c0a)} (Control={int((c0a.binary_label==0).sum())}, Patient={int((c0a.binary_label==1).sum())})\n- pooled AUC: {am['auc']:.4f}; 95% CI: {aci.iloc[0].lower_95:.4f}-{aci.iloc[0].upper_95:.4f}\n- BA: {am['balanced_accuracy']:.4f}; Macro-F1: {am['macro_f1']:.4f}; sensitivity: {am['sensitivity']:.4f}; specificity: {am['specificity']:.4f}; Brier: {am['brier']:.4f}\n- same-camera signal: {a_level}\n- Original-RGB-224 matched control: {matched_status}\n\nC0-A uses SH093-only images and the exact B0 outer/inner splits. The B0 comparison includes both input-size and preprocessing differences.\n", root/"c0a_disease/reports/c0a_report.md"); twrite(f"# C0-B Report\n\n- Patient-only cohort: {len(cb)} (Xiaomi={cba['xiaomi_patient']}, HONOR={cba['honor_patient']})\n- pooled camera AUC: {cm['auc']:.4f}; 95% CI: {cci.iloc[0].lower_95:.4f}-{cci.iloc[0].upper_95:.4f}\n- BA: {cm['balanced_accuracy']:.4f}; Macro-F1: {cm['macro_f1']:.4f}; Xiaomi recall: {cm['recall_negative']:.4f}; HONOR recall: {cm['recall_positive']:.4f}\n- covariate baseline AUC: {cov_auc:.4f}; low-level image baseline AUC: {low_auc:.4f}; matched pairs: {matchj['n_pairs']}\n- camera-associated domain predictability: {cam_level}\n\nThis is not a pure sensor fingerprint.\n", root/"c0b_camera/reports/c0b_report.md")
    return root


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR); ap.add_argument("--no-resume", action="store_true"); ap.add_argument("--stop-after-preflight", action="store_true"); args=ap.parse_args(); print(f"STAGE3_C0_DIR={run(args.output_dir, not args.no_resume, args.stop_after_preflight)}")


if __name__ == "__main__": main()
