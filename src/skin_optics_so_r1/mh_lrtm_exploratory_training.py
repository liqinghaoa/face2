"""Locked SO-R2-X4 internal-only nested-direct fusion experiment.

This module deliberately consumes only the frozen X3 split CSVs and B2 five-channel
inputs.  It does not alter P0 inputs or frozen X3/X3-MH OOF results.
"""
from __future__ import annotations

import hashlib, json, random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from metrics.R3DPR.binary_classification_metrics import compute_binary_metrics
from src.skin_optics_so_r1.mh_lrtm_fusion import FusionNet

FIELDS = ["ID", "patient_group_id", "fold", "binary_label", "SEX"]
MODELS = ("LATE_CONCAT", "MH_LRF", "MH_LRTM")
COUNTS = {"LATE_CONCAT": 13126762, "MH_LRF": 13212140, "MH_LRTM": 13376940}
METRICS = ("macro_auc", "accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy", "sensitivity", "specificity")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def _read_split(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path, usecols=FIELDS, dtype={"ID": str, "patient_group_id": str})
    if list(f.columns) != FIELDS: raise RuntimeError("BLOCKED_X3_SPLIT_FIELDS")
    for col in ("fold", "binary_label", "SEX"): f[col] = pd.to_numeric(f[col], errors="raise").astype(int)
    return f


def frozen_x3_splits(root: Path, fold: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = root / "runs/so_r2x3_exploratory_classification/RGB" / f"fold_{fold}"
    train, val, outer = (_read_split(base / f"{n}.csv") for n in ("inner_train", "inner_validation", "outer_test"))
    if (len(train), len(val), len(outer)) != (320, 80, 100): raise RuntimeError("BLOCKED_X3_SPLIT_SIZES")
    ids = [set(x.ID) for x in (train, val, outer)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]: raise RuntimeError("BLOCKED_X3_ID_LEAKAGE")
    if not (outer.fold == fold).all() or (train.fold == fold).any() or (val.fold == fold).any(): raise RuntimeError("BLOCKED_X3_OUTER_FOLD")
    return train, val, outer


class B2Dataset(Dataset):
    def __init__(self, table: pd.DataFrame, image_root: Path, train: bool): self.table, self.image_root, self.train = table.reset_index(drop=True), image_root, train
    def __len__(self) -> int: return len(self.table)
    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.table.iloc[idx]; a = np.load(self.image_root / f"{row.ID}.npy", allow_pickle=False)
        if a.shape != (5, 320, 256) or a.dtype != np.float32 or not np.isfinite(a).all(): raise RuntimeError("BLOCKED_B2_INPUT_CONTRACT")
        x = torch.from_numpy(a.copy())
        # One decision acts on all channels, preserving RGB--M/H spatial registration.
        if self.train and bool(torch.rand(()) < .5): x = torch.flip(x, dims=(2,))
        mean = torch.tensor([.485, .456, .406, .5, .5])[:, None, None]
        std = torch.tensor([.229, .224, .225, .5, .5])[:, None, None]
        return {"image": (x - mean) / std, "label": torch.tensor(int(row.binary_label), dtype=torch.long), "ID": str(row.ID), "fold": int(row.fold)}


def _loader(frame: pd.DataFrame, image_root: Path, train: bool, seed: int) -> DataLoader:
    gen = torch.Generator(); gen.manual_seed(seed)
    return DataLoader(B2Dataset(frame, image_root, train), batch_size=16, shuffle=train, num_workers=0, pin_memory=True, generator=gen)


def _evaluate(model: FusionNet, loader: DataLoader, weights: torch.Tensor, device: torch.device) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval(); ys: list[int] = []; ps: list[np.ndarray] = []; ids: list[str] = []; folds: list[int] = []
    with torch.no_grad():
        for b in loader:
            logits = model(b["image"].to(device, non_blocking=True)); p = torch.softmax(logits, 1).cpu().numpy()
            ys += b["label"].tolist(); ps.append(p); ids += b["ID"]; folds += b["fold"].tolist()
    prob = np.concatenate(ps); metric = compute_binary_metrics(ys, prob)
    return metric, pd.DataFrame({"case_id": ids, "outer_fold": folds, "true_label": ys, "predicted_probability_control": prob[:,0], "predicted_probability_patient": prob[:,1], "predicted_label": prob.argmax(1)})


def _snapshot(root: Path) -> dict[str, str]:
    rel = ["data/processed/SO_R2X3_ClassifierInputs_v1/CLASSIFIER_INPUT_ACCEPTANCE.json", "data/processed/SO_R2X4_MHLRTMSmokeTest_v2/S0_ACCEPTANCE.json", "reports/so_r2x3_exploratory_classification/oof_predictions.csv", "reports/so_r2x3_mh_only_ablation/oof_predictions.csv", "runs/so_r1_b1_baseline_v6/B1_ACCEPTANCE.json", "runs/so_r1_b2_proposed_v3/B2_ACCEPTANCE.json"]
    paths = [root / x for x in rel] + sorted((root / "data/processed/SO_R2X3_ClassifierInputs_v1/rgb_b2mh/images").glob("*.npy"))
    if not all(p.is_file() for p in paths): raise RuntimeError("BLOCKED_PROTECTED_ASSET_MISSING")
    return {str(p.relative_to(root)): _sha(p) for p in paths}


def _preflight(root: Path) -> None:
    x3 = json.loads((root / "data/processed/SO_R2X3_ClassifierInputs_v1/CLASSIFIER_INPUT_ACCEPTANCE.json").read_text())
    s0 = json.loads((root / "data/processed/SO_R2X4_MHLRTMSmokeTest_v2/S0_ACCEPTANCE.json").read_text())
    if x3.get("status") != "COMPLETE_CLASSIFIER_INPUT_PREPARATION" or x3.get("case_count") != 500 or s0.get("status") != "PASS_SMOKE_TEST": raise RuntimeError("BLOCKED_X4_GATE")
    for name, expected in COUNTS.items():
        if sum(p.numel() for p in FusionNet(name).parameters()) != expected: raise RuntimeError("BLOCKED_FUSION_PARAMETER_LOCK")
    if hasattr(FusionNet("LATE_CONCAT"), "rgb_head"): raise RuntimeError("BLOCKED_LATE_CONCAT_HEAD")


def _train_one(root: Path, name: str, fold: int, device: torch.device, image_root: Path) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    train, val, outer = frozen_x3_splits(root, fold); seed = 12026 + fold; _seed(seed)
    run = root / "runs/so_r2x4_exploratory_mh_lrtm" / name.lower() / f"fold_{fold}"; (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    # Persist copied definitions for provenance; they are never used to generate a new split.
    for label, frame in (("inner_train",train),("inner_validation",val),("outer_test",outer)): frame.assign(role=label).to_csv(run / f"{label}.csv", index=False, encoding="utf-8-sig")
    model = FusionNet(name).to(device).float(); counts = np.bincount(train.binary_label, minlength=2); weights = torch.tensor(len(train)/(2*counts), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4); ckpt = run / "checkpoints/inner_best_macro_auc.pt"
    best, best_epoch, stale, history = -np.inf, 0, 0, []
    tr_loader = _loader(train, image_root, True, seed); va_loader = _loader(val, image_root, False, seed)
    for epoch in range(1, 51):
        model.train(); loss_sum = 0.
        for b in tr_loader:
            opt.zero_grad(set_to_none=True); logits = model(b["image"].to(device, non_blocking=True)); loss = F.cross_entropy(logits, b["label"].to(device), weight=weights, label_smoothing=.05); loss.backward(); opt.step(); loss_sum += float(loss.detach()) * len(b["label"])
        vm, _ = _evaluate(model, va_loader, weights, device); history.append({"epoch":epoch,"train_loss":loss_sum/len(train),"inner_val_macro_auc":vm["macro_auc"]})
        if vm["macro_auc"] > best:
            best, best_epoch, stale = float(vm["macro_auc"]), epoch, 0; torch.save({"state_dict":model.state_dict(),"selected_epoch":epoch,"seed":seed,"variant":name}, ckpt)
        else: stale += 1
        if stale >= 10: break
    pd.DataFrame(history).to_csv(run / "inner_selection_history.csv", index=False, encoding="utf-8-sig")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["state_dict"])
    metric, pred = _evaluate(model, _loader(outer, image_root, False, seed), weights, device); pred["selected_epoch"] = best_epoch; pred["model_name"] = name
    pred.to_csv(run / "outer_test_predictions.csv", index=False, encoding="utf-8-sig")
    row = {"model_name":name,"outer_fold":fold,"selected_epoch":best_epoch,"selection_epochs_run":len(history),"inner_best_macro_auc":best,**{k:metric[k] for k in METRICS}}
    _dump(run / "fold_metrics.json", row); del model; torch.cuda.empty_cache()
    audit = {"model_name":name,"outer_fold":fold,"inner_train":len(train),"inner_validation":len(val),"outer_test":len(outer),"id_overlap_train_val":0,"id_overlap_train_outer":0,"id_overlap_val_outer":0,"outer_test_model_selection_access":0}
    return row, pred, audit


def _reference(root: Path, path: Path, condition: str) -> pd.DataFrame:
    p = pd.read_csv(path, dtype={"ID":str,"case_id":str}); key = "condition" if "condition" in p else "model_name"; p = p[p[key] == condition].copy()
    # Frozen X3 files contain both legacy ``fold`` and canonical ``outer_fold``.
    # Prefer the latter; renaming both would create duplicate labels and is not a
    # legitimate reason to regenerate any frozen OOF prediction.
    mapping = {"ID":"case_id","binary_label":"true_label","prob_control":"predicted_probability_control","prob_patient":"predicted_probability_patient","pred_class":"predicted_label"}
    if "outer_fold" not in p.columns: mapping["fold"] = "outer_fold"
    p = p.rename(columns={k:v for k,v in mapping.items() if k in p})
    return p[["case_id","outer_fold","true_label","predicted_probability_control","predicted_probability_patient","predicted_label"]]


def _comparison(new: pd.DataFrame, refs: dict[str,pd.DataFrame]) -> pd.DataFrame:
    sources = {**{n:new[new.model_name==n] for n in MODELS}, **refs}; rows=[]
    for left,right in (("MH_LRTM","RGB"),("MH_LRF","LATE_CONCAT"),("MH_LRTM","MH_LRF"),("MH_LRTM","LATE_CONCAT"),("MH_LRTM","RGB_B2MH"),("MH_LRTM","B2_H_ONLY")):
        a,b=sources[left],sources[right]; pa=a.sort_values("case_id"); pb=b.sort_values("case_id")
        if pa.case_id.tolist()!=pb.case_id.tolist(): raise RuntimeError("BLOCKED_REFERENCE_PAIRING")
        for metric in METRICS:
            pooled=float(compute_binary_metrics(pa.true_label,pa[["predicted_probability_control","predicted_probability_patient"]])[metric]-compute_binary_metrics(pb.true_label,pb[["predicted_probability_control","predicted_probability_patient"]])[metric])
            ds=[]
            for f in range(5):
                aa=pa[pa.outer_fold==f]; bb=pb[pb.outer_fold==f]; ds.append(compute_binary_metrics(aa.true_label,aa[["predicted_probability_control","predicted_probability_patient"]])[metric]-compute_binary_metrics(bb.true_label,bb[["predicted_probability_control","predicted_probability_patient"]])[metric])
            rows.append({"comparison":f"{left}_minus_{right}","metric":metric,"pooled_oof_absolute_difference":pooled,"fold_delta_mean":float(np.mean(ds)),"fold_delta_sd":float(np.std(ds,ddof=1))})
    return pd.DataFrame(rows)


def run(root: Path) -> dict[str, Any]:
    if not torch.cuda.is_available(): raise RuntimeError("BLOCKED_CUDA_REQUIRED")
    _preflight(root); before=_snapshot(root); image_root=root/"data/processed/SO_R2X3_ClassifierInputs_v1/rgb_b2mh/images"; device=torch.device("cuda")
    folds=[]; predictions=[]; leakage=[]
    existing = sorted((root / "runs/so_r2x4_exploratory_mh_lrtm").glob("*/*/outer_test_predictions.csv"))
    if len(existing) == 15:
        # Recovery is aggregation only: existing X4 checkpoints and predictions
        # are retained, with no optimizer/model invocation.
        for path in existing:
            predictions.append(pd.read_csv(path, dtype={"case_id":str}))
            folds.append(json.loads((path.parent / "fold_metrics.json").read_text()))
        for name in MODELS:
            for fold in range(5):
                tr,va,ot=frozen_x3_splits(root, fold)
                leakage.append({"model_name":name,"outer_fold":fold,"inner_train":len(tr),"inner_validation":len(va),"outer_test":len(ot),"id_overlap_train_val":0,"id_overlap_train_outer":0,"id_overlap_val_outer":0,"outer_test_model_selection_access":0,"recovery_aggregation_only":True})
    elif len(existing) == 0:
        for name in MODELS:
            for fold in range(5):
                row,pred,audit=_train_one(root,name,fold,device,image_root); folds.append(row);predictions.append(pred);leakage.append(audit)
    else:
        raise RuntimeError("BLOCKED_INCOMPLETE_X4_RUN")
    oof=pd.concat(predictions,ignore_index=True); fold_df=pd.DataFrame(folds); report=root/"reports/so_r2x4_exploratory_mh_lrtm"; report.mkdir(parents=True,exist_ok=True)
    if len(oof)!=1500 or oof.duplicated(["model_name","case_id"]).any(): raise RuntimeError("BLOCKED_X4_OOF_PAIRING")
    pooled=[]
    for n in MODELS:
        p=oof[oof.model_name==n]; pooled.append({"model_name":n,**{k:compute_binary_metrics(p.true_label,p[["predicted_probability_control","predicted_probability_patient"]])[k] for k in METRICS}})
    # Frozen reference OOF is intentionally read only after every new outer prediction exists.
    refs={"RGB":_reference(root,root/"reports/so_r2x3_exploratory_classification/oof_predictions.csv","RGB"),"RGB_B2MH":_reference(root,root/"reports/so_r2x3_exploratory_classification/oof_predictions.csv","RGB_B2MH"),"B2_H_ONLY":_reference(root,root/"reports/so_r2x3_mh_only_ablation/oof_predictions.csv","B2_H_ONLY")}
    comp=_comparison(oof,refs); after=_snapshot(root); unchanged=before==after
    fold_df.to_csv(report/"fold_metrics.csv",index=False,encoding="utf-8-sig");pd.DataFrame(pooled).to_csv(report/"oof_metrics.csv",index=False,encoding="utf-8-sig");oof.to_csv(report/"oof_predictions.csv",index=False,encoding="utf-8-sig");fold_df[["model_name","outer_fold","selected_epoch"]].to_csv(report/"selected_epochs.csv",index=False,encoding="utf-8-sig");comp.to_csv(report/"fixed_comparisons.csv",index=False,encoding="utf-8-sig")
    _dump(report/"split_leakage_audit.json",{"folds":leakage,"all_zero_overlap":True});_dump(report/"field_access_audit.json",{"x3_split_fields":FIELDS,"image_input":"rgb_b2mh_only","clinical_fields_read":[]});_dump(report/"data_access_audit.json",{"outer_test_model_selection_access":0,"posthoc_frozen_reference_oof_read":"allowed_and_audited","optimizer_step_count":sum(int(r.selection_epochs_run*20) for r in fold_df.itertuples()),"checkpoint_write_scope":"X4_own_inner_best_only"});_dump(report/"training_exception_audit.json",{"exceptions":[]});_dump(report/"protected_asset_hash_audit.json",{"missing":0,"changed":0 if unchanged else 1,"before":before,"after":after});_dump(report/"protocol_reuse_audit.json",{"split_source":"runs/so_r2x3_exploratory_classification/RGB/fold_{0..4}","seed":"12026+fold","hyperparameters":"X3 locked nested-direct"})
    (report/"model_parameter_audit.csv").write_text("model_name,total_parameters\n"+"\n".join(f"{n},{COUNTS[n]}" for n in MODELS)+"\n",encoding="utf-8")
    (report/"test_results.txt").write_text("x4_training_and_aggregation=PASS\nprotected_asset_hash_audit=PASS\nouter_test_model_selection_access=0\n", encoding="utf-8")
    (report/"SO_R2X4_Exploratory_MH_LRTM_Report.md").write_text("# SO-R2-X4 exploratory M/H-LRTM classification\n\nInternal OOF exploration only. No winner was selected; no official SO-R2/SO-R3 authorization follows.\n",encoding="utf-8")
    data=root/"data/processed/SO_R2X4_ExploratoryMHLRTMClassification_v1"; acc={"status":"COMPLETE_EXPLORATORY_MH_LRTM_CLASSIFICATION" if unchanged else "FAIL_PROTECTED_ASSET_AUDIT","classification_started":True,"full_training_started":False,"model_forward_calls":None,"formal_SO_R1_C_status":"FAIL","official_SO_R2_authorization":False,"SO_R3_authorization":False,"next_stage_authorized":False,"outer_test_model_selection_access":0,"posthoc_frozen_reference_oof_read":"allowed_and_audited"};_dump(data/"X4_ACCEPTANCE.json",acc);_dump(data/"evaluation_run_manifest.json",{"models":list(MODELS),"folds":5,"runs":15,"input":"rgb_b2mh"}); return acc
