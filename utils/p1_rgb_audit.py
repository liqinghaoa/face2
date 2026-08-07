"""Non-mutating data, historical-equivalence, and OOF helpers for P1-RGB."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, spearmanr

EXPECTED_SPLIT_SHA = "d5a20fb56c96e657dd7902b6d829bed78b6d43d3e58ec47e6bd3542ec34378cb"
ROOT = Path(__file__).resolve().parents[1]

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def local_path(value: str | Path) -> Path:
    text = str(value).replace("\\", "/")
    if text.lower().startswith("/mnt/e/"): return Path("E:/" + text[7:])
    return Path(text)

def _require(frame: pd.DataFrame, cols: set[str], name: str) -> None:
    miss = cols.difference(frame.columns)
    if miss: raise ValueError(f"{name} missing required columns: {sorted(miss)}")

def load_p1_frame(manifest: Path, split: Path) -> pd.DataFrame:
    """Reconcile the sole P1 manifest with the sole immutable P0 split."""
    if sha(split) != EXPECTED_SPLIT_SHA: raise ValueError("fixed split SHA256 mismatch")
    master = pd.read_csv(manifest, dtype={"case_id": str, "patient_id": str, "group_id": str}, encoding="utf-8-sig")
    _require(master, {"case_id", "patient_id", "group_id", "fold", "label_original", "label_3class", "label_binary", "rgb_path", "p0_quality_status"}, "P1 master manifest")
    fixed = pd.read_csv(split, dtype={"ID": str, "patient_group_id": str}, encoding="utf-8-sig")
    _require(fixed, {"ID", "patient_group_id", "fold", "NYHA", "label_3class", "binary_label"}, "fixed split")
    if len(master) != 500 or master.case_id.nunique() != 500: raise ValueError("P1 master must have 500 unique cases")
    if len(fixed) != 500 or fixed.ID.nunique() != 500: raise ValueError("fixed split must have 500 unique cases")
    merged = master.merge(fixed.rename(columns={"ID": "case_id", "patient_group_id": "split_patient_group_id", "NYHA": "split_label_original", "label_3class": "split_label_3class", "binary_label": "split_label_binary", "fold": "split_fold"}), on="case_id", how="outer", indicator=True)
    if not (merged["_merge"] == "both").all(): raise ValueError("P1 master and fixed split case IDs differ")
    if not (merged["group_id"].astype(str) == merged["split_patient_group_id"].astype(str)).all():
        raise ValueError("manifest group_id disagrees with authoritative fixed split patient_group_id")
    merged["patient_group_id"] = merged["split_patient_group_id"].astype(str)
    for actual, expected in (("fold", "split_fold"), ("label_original", "split_label_original"), ("label_3class", "split_label_3class"), ("label_binary", "split_label_binary")):
        a = pd.to_numeric(merged[actual], errors="coerce"); b = pd.to_numeric(merged[expected], errors="coerce")
        if a.isna().any() or b.isna().any() or not (a.astype(int) == b.astype(int)).all(): raise ValueError(f"manifest/split mismatch: {actual}")
    labels = merged[["label_original", "label_3class", "label_binary"]].astype(int)
    expected_three = labels.label_original.map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2})
    expected_binary = (labels.label_original > 0).astype(int)
    if not (labels.label_3class == expected_three).all() or not (labels.label_binary == expected_binary).all(): raise ValueError("invalid NYHA/three-class/binary mapping")
    aligned_root = (ROOT / "data/processed/P0_Physics_Audit_v1/images/aligned_scene_224").resolve()
    resolved_rgb = merged.rgb_path.map(local_path).map(lambda p: p.resolve())
    if not resolved_rgb.map(lambda p: aligned_root in p.parents and p.is_file()).all():
        raise FileNotFoundError("rgb_path must be an existing P0 aligned_scene_224 image")
    merged["rgb_path"] = resolved_rgb.map(str)
    groups = merged.groupby("patient_group_id")
    if groups.fold.nunique().max() != 1: raise ValueError("patient group crosses folds")
    # Longitudinal patient groups may change labels across visits; only fold leakage is invalid.
    for fold in range(5):
        val = merged[merged.fold == fold]; train = merged[merged.fold != fold]
        if len(val) != 100 or len(train) != 400 or (val.label_binary == 0).sum() != 23 or (val.label_binary == 1).sum() != 77 or (train.label_binary == 0).sum() != 92 or (train.label_binary == 1).sum() != 308: raise ValueError(f"unexpected fixed fold distribution: {fold}")
    return merged

def split_equivalence(frame: pd.DataFrame, old_dir: Path, metadata: Path) -> str:
    rows=[]; status="EXACT_MATCH"
    if not old_dir.is_dir():
        status="HISTORICAL_SPLIT_UNAVAILABLE"; pd.DataFrame([{ "status": status }]).to_csv(metadata/'split_equivalence_audit.csv',index=False); (metadata/'split_equivalence_summary.json').write_text(json.dumps({"status":status},indent=2)); return status
    for fold in range(5):
        expected_val=frame.loc[frame.fold==fold,['case_id','patient_group_id','fold','label_original','label_3class','label_binary']].copy(); expected_train=frame.loc[frame.fold!=fold,['case_id','patient_group_id','fold','label_original','label_3class','label_binary']].copy()
        for role, expected_frame in (("val",expected_val),("train",expected_train)):
            path=old_dir/f"fold_{fold}_{role}.csv"
            if not path.is_file(): status="HISTORICAL_SPLIT_UNAVAILABLE"; rows.append({"fold":fold,"role":role,"status":"MISSING"}); continue
            old=pd.read_csv(path,dtype={'ID':str,'patient_group_id':str}); observed=set(old.ID.astype(str)); expected=set(expected_frame.case_id.astype(str)); exact=observed==expected and len(old)==len(expected)
            fields_ok=False
            if exact and {'ID','patient_group_id','fold','NYHA','label_3class'}.issubset(old.columns):
                old = old.copy(); old['old_label_binary'] = (pd.to_numeric(old['NYHA'], errors='coerce') > 0).astype(int)
                check=expected_frame.merge(old[['ID','patient_group_id','fold','NYHA','label_3class','old_label_binary']],left_on='case_id',right_on='ID',how='inner',suffixes=('_new','_old'))
                fields_ok=len(check)==len(expected_frame) and (check.patient_group_id_new.astype(str)==check.patient_group_id_old.astype(str)).all() and (check.fold_new.astype(int)==check.fold_old.astype(int)).all() and (check.label_original.astype(int)==check.NYHA.astype(int)).all() and (check.label_3class_new.astype(int)==check.label_3class_old.astype(int)).all() and (check.label_binary.astype(int)==check.old_label_binary.astype(int)).all()
            exact = exact and fields_ok
            rows.append({"fold":fold,"role":role,"expected_n":len(expected),"observed_n":len(old),"exact_case_set":observed==expected,"patient_group_fold_nyha_label_binary_match":fields_ok,"status":"EXACT_MATCH" if exact else "DIFFERENT"})
            if not exact and status != "HISTORICAL_SPLIT_UNAVAILABLE": status="DIFFERENT_BUT_P0_FIXED_SPLIT_USED"
    pd.DataFrame(rows).to_csv(metadata/'split_equivalence_audit.csv',index=False,encoding='utf-8-sig'); (metadata/'split_equivalence_summary.json').write_text(json.dumps({"status":status},indent=2)); return status

def image_equivalence(frame: pd.DataFrame, old_root: Path, metadata: Path) -> str:
    rows=[]
    if not old_root.is_dir():
        status="HISTORICAL_RGB_UNAVAILABLE"; pd.DataFrame([{ "status": status }]).to_csv(metadata/'image_equivalence_audit.csv',index=False); (metadata/'image_equivalence_summary.json').write_text(json.dumps({"status":status},indent=2)); return status
    for row in frame.itertuples():
        current=Path(row.rgb_path); old=old_root/f"{row.case_id}.png"; record={"case_id":row.case_id,"old_exists":old.is_file(),"current_exists":current.is_file()}
        if old.is_file() and current.is_file():
            with Image.open(old) as im: a=np.asarray(im.convert('RGB'))
            with Image.open(current) as im: b=np.asarray(im.convert('RGB'))
            record.update({"old_sha256":sha(old),"current_sha256":sha(current),"old_decoded_rgb_sha256":hashlib.sha256(a.tobytes()).hexdigest(),"current_decoded_rgb_sha256":hashlib.sha256(b.tobytes()).hexdigest(),"old_shape":str(a.shape),"current_shape":str(b.shape),"dtype":str(b.dtype),"mae":float(np.abs(a.astype(float)-b.astype(float)).mean()) if a.shape==b.shape else np.nan,"max_abs":int(np.abs(a.astype(int)-b.astype(int)).max()) if a.shape==b.shape else np.nan,"exact_equal":bool(a.shape==b.shape and np.array_equal(a,b))})
        rows.append(record)
    result=pd.DataFrame(rows); status="EXACT_MATCH_500_OF_500" if len(result)==500 and result.old_exists.all() and result.exact_equal.all() else "DIFFERENT_INPUT_REPRESENTATION"
    result.to_csv(metadata/'image_equivalence_audit.csv',index=False,encoding='utf-8-sig'); (metadata/'image_equivalence_summary.json').write_text(json.dumps({"status":status,"exact_count":int(result.exact_equal.fillna(False).sum())},indent=2)); return status

def write_source_metadata(frame: pd.DataFrame, manifest: Path, split: Path, metadata: Path, split_status: str, image_status: str) -> str:
    metadata.mkdir(parents=True,exist_ok=True); protocol="E0B_STRICT_INPUT_PROTOCOL_REPRODUCTION" if split_status=="EXACT_MATCH" and image_status=="EXACT_MATCH_500_OF_500" else "P0_UNIFIED_RGB_REFRESHED_BASELINE"
    data={"master_manifest":str(manifest),"master_manifest_sha256":sha(manifest),"fixed_split":str(split),"fixed_split_sha256":sha(split),"aligned_rgb_cases":len(frame),"patient_groups":int(frame.patient_group_id.nunique())}
    (metadata/'source_hashes.json').write_text(json.dumps(data,indent=2)); (metadata/'protocol_identity.json').write_text(json.dumps({"protocol_identity":protocol,"split_equivalence_status":split_status,"image_equivalence_status":image_status},indent=2)); return protocol

def aggregate_groups(case: pd.DataFrame) -> pd.DataFrame:
    records=[]
    for gid,g in case.groupby('patient_group_id',sort=False):
        if g.label_binary.nunique()!=1 or g.fold.nunique()!=1: raise ValueError(f"invalid group aggregation: {gid}")
        p=float(g.prob_patient.mean()); records.append({"patient_group_id":str(gid),"n_cases":len(g),"case_ids":";".join(g.case_id.astype(str)),"fold":int(g.fold.iloc[0]),"label_binary":int(g.label_binary.iloc[0]),"prob_control":1-p,"prob_patient":p,"pred_binary":int(p>=.5)})
    out=pd.DataFrame(records)
    if len(out)!=out.patient_group_id.nunique(): raise ValueError("duplicate group output")
    return out

def historical_prediction_comparison(case: pd.DataFrame, historical: Path, summary: Path) -> dict[str,Any]:
    if not historical.is_file(): return {"historical_e0b_prediction_comparison_status":"unavailable"}
    old=pd.read_csv(historical,dtype={'sample_id':str,'patient_group_id':str}); merged=case.merge(old.rename(columns={'sample_id':'case_id','prob_patient':'historical_prob_patient','pred_class':'historical_pred_binary'}),on='case_id',how='inner')
    if len(merged)!=len(case): return {"historical_e0b_prediction_comparison_status":"unavailable_case_set_mismatch"}
    d=merged.prob_patient-merged.historical_prob_patient
    result={"historical_e0b_prediction_comparison_status":"available","probability_mae":float(np.abs(d).mean()),"probability_rmse":float(np.sqrt(np.mean(d*d))),"pearson":float(pearsonr(merged.prob_patient,merged.historical_prob_patient).statistic),"spearman":float(spearmanr(merged.prob_patient,merged.historical_prob_patient).statistic),"hard_prediction_agreement":float((merged.pred_binary==merged.historical_pred_binary).mean()),"changed_prediction_count":int((merged.pred_binary!=merged.historical_pred_binary).sum())}
    (summary/'prediction_agreement_with_e0b.json').write_text(json.dumps(result,indent=2)); return result
