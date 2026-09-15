"""SO-R2-X2: read-only, label-blind audit of X1 fused representation maps."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import yaml

ROOT_DATA = Path("data/processed/SO_R2X1_RealFaceFrozenInference_v1")
REPORT = Path("reports/so_r2x2_realface_output_quality_audit")
OUT = Path("data/processed/SO_R2X2_RealFaceOutputAudit_v1")
REAL = Path("data/processed/skin_optics/skinoptics_realface_979x1220_blackbg_v1")
SEAM_X, SEAM_Y, OFFSET = (241,482,723), (241,482,723,964), 16
QUAL_INDICES = (1,46,91,137,182,228,273,319,364,410,455,500)
MODELS = (("B1_baseline", "B1"), ("B2_proposed_lambda_0p50", "B2_lambda_0p50"))

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def dump(p: Path, x: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")

def _stats(a: np.ndarray) -> dict[str,float]:
    q=np.percentile(a,[1,5,25,50,75,95,99])
    return {"min":float(a.min()),"max":float(a.max()),"mean":float(a.mean()),"median":float(q[3]),"std":float(a.std()),"p01":float(q[0]),"p05":float(q[1]),"p25":float(q[2]),"p75":float(q[4]),"p95":float(q[5]),"p99":float(q[6]),"iqr":float(q[4]-q[2]),"p99_p01_range":float(q[6]-q[0]),"zero_fraction":float((a==0).mean()),"one_fraction":float((a==1).mean()),"near_zero_fraction":float((a<=1e-4).mean()),"near_one_fraction":float((a>=1-1e-4).mean())}

def _seam(a: np.ndarray, valid: np.ndarray, axis: str, position: int) -> dict[str,float]:
    if axis=="x":
        seam=np.abs(a[:,position]-a[:,position-1]); ok=valid[:,position]&valid[:,position-1]
        left=np.abs(a[:,position-OFFSET]-a[:,position-OFFSET-1]); lo=valid[:,position-OFFSET]&valid[:,position-OFFSET-1]
        right=np.abs(a[:,position+OFFSET+1]-a[:,position+OFFSET]); ro=valid[:,position+OFFSET+1]&valid[:,position+OFFSET]
    else:
        seam=np.abs(a[position,:]-a[position-1,:]); ok=valid[position,:]&valid[position-1,:]
        left=np.abs(a[position-OFFSET,:]-a[position-OFFSET-1,:]); lo=valid[position-OFFSET,:]&valid[position-OFFSET-1,:]
        right=np.abs(a[position+OFFSET+1,:]-a[position+OFFSET,:]); ro=valid[position+OFFSET+1,:]&valid[position+OFFSET,:]
    s=seam[ok]; refs=np.concatenate((left[lo],right[ro]))
    ref=float(np.median(refs)) if refs.size else np.nan
    return {"seam_abs_first_difference":float(np.median(s)) if s.size else np.nan,"matched_local_reference_difference":ref,"seam_amplification_ratio":float(np.median(s)/(ref+1e-12)) if s.size and np.isfinite(ref) else np.nan,"valid_pair_count":int(s.size)}

def _corr(x: np.ndarray,y: np.ndarray) -> tuple[float,float,int]:
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3 or np.std(x[ok])==0 or np.std(y[ok])==0:return np.nan,np.nan,int(ok.sum())
    return float(spearmanr(x[ok],y[ok]).statistic),float(pearsonr(x[ok],y[ok]).statistic),int(ok.sum())

def _protocol() -> dict[str,Any]:
    return {"stage":"SO-R2-X2","source_stage":"SO-R2-X1","map_statistics":"valid-mask only","channels":["M_sensitive","H_sensitive"],"models":["B1_baseline","B2_proposed_lambda_0p50"],"seam_x":list(SEAM_X),"seam_y":list(SEAM_Y),"reference_offset":OFFSET,"qualitative_selection":"canonical full_ids fixed quantiles","qualitative_indices":list(QUAL_INDICES),"forward_calls_allowed":0,"label_access_allowed":False,"visualization":{"map_display":"per-board valid-mask percentile 1-99 only; source arrays unchanged"}}

def _preflight(root: Path) -> tuple[list[str],dict[str,str]]:
    x1=root/ROOT_DATA; acc=json.loads((x1/"R2X1_ACCEPTANCE.json").read_text())
    if acc.get("status")!="COMPLETE_EXPLORATORY_FROZEN_INFERENCE":raise RuntimeError("BLOCKED:X1_ACCEPTANCE")
    ids=[r["case_id"] for r in csv.DictReader((x1/"inference_manifest.csv").open(encoding="utf-8"))]
    if len(ids)!=500 or len(set(ids))!=500:raise RuntimeError("BLOCKED:X1_MANIFEST")
    for _,d in MODELS:
        if len(list((x1/d).glob("*.npz")))!=500:raise RuntimeError("BLOCKED:MAP_COUNT")
    if len(list((x1/"valid_masks").glob("*.png")))!=500:raise RuntimeError("BLOCKED:MASK_COUNT")
    h={"B1_checkpoint_hash":sha(root/"runs/so_r1_b1_baseline_v6/checkpoints/best_val_masked_smoothl1.pt"),"B2_checkpoint_hash":sha(root/"runs/so_r1_b2_proposed_v3/lambda_0p50/checkpoints/best_val_masked_smoothl1.pt"),"source_R2X1_acceptance_hash":sha(x1/"R2X1_ACCEPTANCE.json")}
    if h["B1_checkpoint_hash"]!="5b1dd85f840d4d00a7c1c1f3b7c0a9306031408ecf0c5a557f53ca0810c0265b" or h["B2_checkpoint_hash"]!="5cd86ceee6c9ad7f448676b59b3f1f5e8d80dc28bce106101b19d1286b6562a0":raise RuntimeError("BLOCKED:CHECKPOINT_HASH")
    return ids,h

def _protected(root:Path)->dict[str,str]:
    files=[root/ROOT_DATA/"R2X1_ACCEPTANCE.json",root/ROOT_DATA/"inference_manifest.csv",root/"reports/so_r2x1_realface_frozen_inference/input_index_lineage_addendum.json",root/"data/processed/SO_R1_C_SyntheticEvaluation_v1/C_ACCEPTANCE.json",root/"runs/so_r1_b1_baseline_v6/B1_ACCEPTANCE.json",root/"runs/so_r1_b2_proposed_v3/B2_ACCEPTANCE.json",root/"runs/so_r2x0_realface_input_audit/R2X0_ACCEPTANCE.json",root/REAL/"COMPLETED.json",root/REAL/"manifest.csv",root/REAL/"full_ids.txt",root/REAL/"config/resolved_config.yaml"]
    files += sorted((root/ROOT_DATA/"B1").glob("*.npz"))+sorted((root/ROOT_DATA/"B2_lambda_0p50").glob("*.npz"))+sorted((root/ROOT_DATA/"valid_masks").glob("*.png"))
    if not all(p.is_file() for p in files):raise RuntimeError("BLOCKED:PROTECTED_MISSING")
    return {str(p.relative_to(root)):sha(p) for p in files}

def _board(path:Path, rgb:np.ndarray, valid:np.ndarray, maps:list[np.ndarray], case:str)->None:
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    axs[0,0].imshow(np.clip(rgb,0,1)); axs[0,0].set_title("linear-RGB display")
    axs[0,1].imshow(valid,cmap="gray"); axs[0,1].set_title("valid mask")
    labels=["B1 M-sensitive","B1 H-sensitive","B2 M-sensitive","B2 H-sensitive"]
    for ax,a,label in zip((axs[0,2],axs[1,0],axs[1,1],axs[1,2]),maps,labels):
        ma=np.ma.masked_where(~valid,a); lo,hi=np.percentile(a[valid],[1,99]); im=ax.imshow(ma,cmap="magma",vmin=lo,vmax=hi); ax.set_title(label); fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
    for a in axs.flat:a.set_axis_off()
    fig.suptitle(f"QUALITATIVE_AUDIT_ONLY · NOT_PHYSIOLOGICAL_GROUND_TRUTH · case {case}")
    fig.savefig(path,dpi=120); plt.close(fig)

def run(root:Path)->dict[str,Any]:
    root=root.resolve()
    if (root/OUT).exists() or (root/REPORT).exists():raise RuntimeError("REFUSE_OVERWRITE_X2")
    ids,h=_preflight(root); before=_protected(root); protocol=_protocol(); (root/"config/so_r1").mkdir(exist_ok=True); (root/"config/so_r1/so_r2x2_realface_output_quality_protocol_v1.yaml").write_text(yaml.safe_dump(protocol,sort_keys=False),encoding="utf-8"); dump(root/"config/so_r1/SO_R2_X2_OUTPUT_AUDIT_LOCK.json",{"status":"FROZEN","protocol_sha256":sha(root/"config/so_r1/so_r2x2_realface_output_quality_protocol_v1.yaml"),"protocol":protocol})
    (root/OUT).mkdir(); (root/REPORT).mkdir(); (root/REPORT/"fixed_qualitative_boards").mkdir()
    dist=[]; seam=[]; agree=[]; tech=[]; hard=[]
    for case in ids:
        valid=cv2.imread(str(root/ROOT_DATA/"valid_masks"/f"{case}.png"),cv2.IMREAD_GRAYSCALE)>0
        arrays={}
        for label,d in MODELS:
            z=np.load(root/ROOT_DATA/d/f"{case}.npz",allow_pickle=False); a=z["mh_sensitive_map"]
            if a.shape!=(2,1220,979) or a.dtype!=np.float32 or z["valid_mask"].shape!=(1220,979) or not np.array_equal(valid,z["valid_mask"]>0):hard.append(f"{case}:{label}:shape_or_mask");continue
            arrays[label]=a
            if not np.isfinite(a[:,valid]).all() or np.any(a[:,~valid]!=0):hard.append(f"{case}:{label}:finite_or_outside")
            for ch,name in enumerate(("M_sensitive","H_sensitive")):
                v=a[ch,valid]; s=_stats(v); zero=bool(np.all(v==0)); const=bool(np.all(v==v[0]));
                if zero or const:hard.append(f"{case}:{label}:{name}:zero_or_constant")
                dist.append({"case_id":case,"model":label,"channel":name,"finite_fraction":float(np.isfinite(v).mean()),"all_zero":zero,"constant":const,**s})
                for pos in SEAM_X:seam.append({"case_id":case,"model":label,"channel":name,"orientation":"vertical","position":pos,**_seam(a[ch],valid,"x",pos)})
                for pos in SEAM_Y:seam.append({"case_id":case,"model":label,"channel":name,"orientation":"horizontal","position":pos,**_seam(a[ch],valid,"y",pos)})
                tech.append({"case_id":case,"model":label,"channel":name,"map_mean":s["mean"],"map_median":s["median"],"map_std":s["std"],"map_dynamic_range":s["p99_p01_range"],"input_luminance_mean":float((np.load(root/REAL/"blackbg_linear_rgb"/f"{case}.npy",allow_pickle=False).mean(axis=2)[valid]).mean())})
        if len(arrays)==2:
            for ch,name in enumerate(("M_sensitive","H_sensitive")):
                x,y=arrays["B1_baseline"][ch,valid],arrays["B2_proposed_lambda_0p50"][ch,valid]; sp,pe,n=_corr(x,y); agree.append({"case_id":case,"channel":name,"masked_mae":float(np.abs(x-y).mean()),"masked_rmse":float(np.sqrt(((x-y)**2).mean())),"pearson":pe,"spearman":sp,"mean_difference":float((x-y).mean()),"std_ratio":float(x.std()/(y.std()+1e-12)),"valid_pixel_count":n})
    D,S,A,T=map(pd.DataFrame,(dist,seam,agree,tech));
    def summary(df,group,cols):return df.groupby(group)[cols].agg(["mean","median",lambda x:np.percentile(x,25),lambda x:np.percentile(x,75),lambda x:np.percentile(x,95)]).reset_index()
    D.to_csv(root/REPORT/"map_distribution_case_metrics.csv",index=False); summary(D,["model","channel"],["std","p99_p01_range","near_zero_fraction","near_one_fraction"]).to_csv(root/REPORT/"map_distribution_summary.csv",index=False)
    S.to_csv(root/REPORT/"seam_case_metrics.csv",index=False); summary(S,["model","channel","orientation","position"],["seam_abs_first_difference","matched_local_reference_difference","seam_amplification_ratio"]).to_csv(root/REPORT/"seam_summary.csv",index=False)
    A.to_csv(root/REPORT/"b1_b2_case_agreement_metrics.csv",index=False); summary(A,["channel"],["masked_mae","masked_rmse","pearson","spearman","mean_difference","std_ratio"]).to_csv(root/REPORT/"b1_b2_agreement_summary.csv",index=False)
    T.to_csv(root/REPORT/"technical_factor_case_metrics.csv",index=False)
    assoc=[]
    for (m,c),g in T.groupby(["model","channel"]):
        for col in ("map_mean","map_median","map_std","map_dynamic_range"):
            sp,pe,n=_corr(g.input_luminance_mean.to_numpy(),g[col].to_numpy());assoc.append({"factor":"input_fullface_validmask_linear_luminance_mean","model":m,"channel":c,"map_summary":col,"spearman_rho":sp,"pearson_r":pe,"n":n,"interpretation":"descriptive technical association; not causal, clinical, or disease-discriminative evidence"})
    pd.DataFrame(assoc).to_csv(root/REPORT/"technical_factor_association_summary.csv",index=False)
    dump(root/REPORT/"output_integrity_audit.json",{"status":"PASS" if not hard else "FAIL","hard_failure_count":len(hard),"hard_failures":hard[:50],"output_boundary":"[0,1] sigmoid; near-boundary epsilon=1e-4"}); dump(root/REPORT/"seam_audit.json",{"status":"SEAM_AUDIT_DESCRIPTIVE","warning_rule":"manual review if repeated high ratios; no pre-frozen artifact threshold","offset":OFFSET}); dump(root/REPORT/"b1_b2_agreement_audit.json",{"status":"DESCRIPTIVE_ONLY","model_selection_performed":False}); dump(root/REPORT/"technical_metadata_availability.json",{"camera_model":"NOT_AVAILABLE_NO_SAFE_WHITELIST_TABLE","BrightnessValue":"NOT_AVAILABLE_NO_SAFE_WHITELIST_TABLE","ExposureTime":"NOT_AVAILABLE_NO_SAFE_WHITELIST_TABLE","ISO":"NOT_AVAILABLE_NO_SAFE_WHITELIST_TABLE","FNumber":"NOT_AVAILABLE_NO_SAFE_WHITELIST_TABLE","input_luminance":"AVAILABLE"}); dump(root/REPORT/"technical_factor_audit.json",{"status":"DESCRIPTIVE_ONLY","technical_metadata_status":"PARTIAL_INPUT_LUMINANCE_ONLY","no_clinical_or_label_fields_read":True})
    boards=[]
    for i in QUAL_INDICES:
        case=ids[i-1]; valid=cv2.imread(str(root/ROOT_DATA/"valid_masks"/f"{case}.png"),cv2.IMREAD_GRAYSCALE)>0; b1=np.load(root/ROOT_DATA/"B1"/f"{case}.npz",allow_pickle=False)["mh_sensitive_map"]; b2=np.load(root/ROOT_DATA/"B2_lambda_0p50"/f"{case}.npz",allow_pickle=False)["mh_sensitive_map"]; rgb=np.load(root/REAL/"blackbg_linear_rgb"/f"{case}.npy",allow_pickle=False); p=root/REPORT/"fixed_qualitative_boards"/f"{case}.png";_board(p,rgb,valid,[b1[0],b1[1],b2[0],b2[1]],case);boards.append({"canonical_index":i,"case_id":case,"board":str(p.relative_to(root))})
    pd.DataFrame(boards).to_csv(root/REPORT/"fixed_qualitative_case_manifest.csv",index=False)
    after=_protected(root); unchanged=before==after; dump(root/REPORT/"protected_asset_hash_audit.json",{"changed":0 if unchanged else 1,"missing":0,"unchanged":unchanged,"before":before,"after":after}); dump(root/REPORT/"field_access_audit.json",{"model_forward_calls":0,"training_calls":0,"checkpoint_writes":0,"NYHA_read_count":0,"label_read_count":0,"fold_read_count":0,"sex_read_count":0,"clinical_variable_read_count":0})
    status="COMPLETE_EXPLORATORY_OUTPUT_AUDIT" if not hard and unchanged else "FAIL_TECHNICAL_OUTPUT_INTEGRITY"; acceptance={"status":status,"technical_output_integrity":"PASS" if not hard else "FAIL","formal_SO_R1_C_status":"FAIL","official_SO_R2_authorization":False,"SO_R3_authorization":False,"classification_started":False,"next_stage_authorized":False,**h,"lambda_pair":0.50,"case_count":500,"map_count_B1":500,"map_count_B2":500,"map_shape":[2,1220,979],"model_forward_calls":0,"training_calls":0,"checkpoint_writes":0,"clinical_or_label_fields_read":False,"seam_audit_status":"SEAM_AUDIT_DESCRIPTIVE","technical_metadata_status":"PARTIAL_INPUT_LUMINANCE_ONLY","protected_assets_unchanged":unchanged,"input_index_lineage_status":"DOCUMENTED_WITH_UNRECORDED_INITIAL_HASH","manual_review_recommended":True}
    dump(root/OUT/"R2X2_ACCEPTANCE.json",acceptance);dump(root/OUT/"evaluation_run_manifest.json",acceptance);dump(root/REPORT/"R2X2_ACCEPTANCE.json",acceptance);(root/REPORT/"SO_R2X2_RealFace_Output_Quality_Audit_Report.md").write_text("# SO-R2-X2\n\nExploratory, label-blind technical audit only; not physiological ground truth or an official SO-R2 gate.\n\n"+json.dumps(acceptance,indent=2),encoding="utf-8");(root/REPORT/"test_results.txt").write_text("pytest and git diff results are recorded after execution.\n",encoding="utf-8")
    return acceptance
