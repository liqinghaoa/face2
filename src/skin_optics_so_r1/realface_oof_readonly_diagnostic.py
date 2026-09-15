"""SO-R2-X3-D: strictly read-only descriptive diagnostics of frozen OOF outputs."""
from __future__ import annotations

import hashlib, json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from metrics.R3DPR.binary_classification_metrics import compute_binary_metrics

ROOT = Path(__file__).resolve().parents[2]
INPUTS = [
    "data/processed/SO_R2X3_ExploratoryClassification_v1/R2X3_ACCEPTANCE.json",
    "reports/so_r2x3_exploratory_classification/oof_predictions.csv",
    "reports/so_r2x3_exploratory_classification/fold_metrics.csv",
    "reports/so_r2x3_exploratory_classification/oof_metrics.csv",
    "reports/so_r2x3_exploratory_classification/oof_confusion_matrix.csv",
]
OOF_COLUMNS = ["ID", "fold", "binary_label", "prob_control", "prob_patient", "pred_class", "condition", "selected_epoch"]
CONDITIONS = ("RGB", "RGB_B1MH", "RGB_B2MH")


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, default=str)+"\n", encoding="utf-8")


def _snapshot(root: Path) -> dict[str,str]:
    paths=[root / item for item in INPUTS]
    if not all(p.is_file() for p in paths): raise RuntimeError("BLOCKED_REQUIRED_OOF_INPUT_MISSING")
    return {str(p.relative_to(root)):_sha(p) for p in paths}


def _load(root: Path) -> pd.DataFrame:
    acc=json.loads((root / INPUTS[0]).read_text(encoding="utf-8"))
    if acc.get("status") != "COMPLETE_EXPLORATORY_NESTED_DIRECT_CLASSIFICATION": raise RuntimeError("BLOCKED_X3_ACCEPTANCE")
    # Explicit white-list prevents patient_group_id and any other clinical/source column from being loaded.
    frame=pd.read_csv(root / INPUTS[1], usecols=OOF_COLUMNS, dtype={"ID":str})
    if set(frame.columns)!=set(OOF_COLUMNS) or len(frame)!=1500: raise RuntimeError("BLOCKED_OOF_SCHEMA")
    frame=frame.loc[:,OOF_COLUMNS].copy()
    # Historical DataLoader collation serialized fold as e.g. ``tensor(0)``;
    # parse that already-stored outer-fold representation in memory only.
    frame["fold"] = frame["fold"].astype(str).str.extract(r"(-?\d+)", expand=False)
    for c in ("fold","binary_label","pred_class","selected_epoch"): frame[c]=pd.to_numeric(frame[c],errors="raise").astype(int)
    for c in ("prob_control","prob_patient"): frame[c]=pd.to_numeric(frame[c],errors="raise").astype(float)
    if set(frame.condition)!=set(CONDITIONS) or not frame.binary_label.isin([0,1]).all() or not frame.pred_class.isin([0,1]).all(): raise RuntimeError("BLOCKED_OOF_VALUES")
    if not np.isfinite(frame[["prob_control","prob_patient"]].to_numpy()).all() or not np.allclose(frame.prob_control+frame.prob_patient,1.0,atol=1e-6): raise RuntimeError("BLOCKED_OOF_PROBABILITIES")
    if frame.duplicated(["condition","ID"]).any() or frame.groupby("condition").size().to_dict()!={c:500 for c in CONDITIONS}: raise RuntimeError("BLOCKED_OOF_PAIRING")
    pivot=frame.pivot(index="ID",columns="condition",values=["fold","binary_label"])
    if not (pivot["fold"].nunique(axis=1)==1).all() or not (pivot["binary_label"].nunique(axis=1)==1).all(): raise RuntimeError("BLOCKED_CROSS_MODEL_LABEL_OR_FOLD_MISMATCH")
    return frame


def _truth_name(label: int | str) -> str: return "Patient" if int(label)==1 else "Control"


def correctness_transitions(frame: pd.DataFrame, mh: str) -> list[dict[str,Any]]:
    rgb=frame[frame.condition=="RGB"].set_index("ID"); other=frame[frame.condition==mh].set_index("ID").reindex(rgb.index)
    a=rgb.pred_class.eq(rgb.binary_label); b=other.pred_class.eq(rgb.binary_label)
    category=np.select([a&b,a&~b,~a&b],["both_correct","RGB_only_correct",f"{mh}_only_correct"],default="both_wrong")
    out=[]
    for group, mask in [("all cases",np.ones(len(rgb),dtype=bool)),("Control",rgb.binary_label.eq(0).to_numpy()),("Patient",rgb.binary_label.eq(1).to_numpy())]:
        for c in ("both_correct","RGB_only_correct",f"{mh}_only_correct","both_wrong"):
            out.append({"comparison":f"{mh}_vs_RGB","true_class":group,"transition":c,"count":int(((category==c)&mask).sum())})
    return out


def probability_summary(frame: pd.DataFrame, mh: str) -> list[dict[str,Any]]:
    rgb=frame[frame.condition=="RGB"].set_index("ID"); other=frame[frame.condition==mh].set_index("ID").reindex(rgb.index); delta=other.prob_patient-rgb.prob_patient
    rows=[]
    for name, source in [("RGB",rgb),(mh,other)]:
        for truth, mask in [("all cases",np.ones(len(source),dtype=bool)),("Control",source.binary_label.eq(0).to_numpy()),("Patient",source.binary_label.eq(1).to_numpy())]:
            values=source.prob_patient.to_numpy()[mask]
            rows.append({"record_type":"prob_patient_distribution","comparison":f"{mh}_vs_RGB","condition":name,"true_class":truth,"count":len(values),"mean":float(np.mean(values)),"median":float(np.median(values)),"p05":float(np.quantile(values,.05)),"p25":float(np.quantile(values,.25)),"p75":float(np.quantile(values,.75)),"p95":float(np.quantile(values,.95))})
    for truth, mask in [("all cases",np.ones(len(rgb),dtype=bool)),("Control",rgb.binary_label.eq(0).to_numpy()),("Patient",rgb.binary_label.eq(1).to_numpy())]:
        d=delta.to_numpy()[mask]; r=rgb.prob_patient.to_numpy()[mask]; o=other.prob_patient.to_numpy()[mask]
        rows.append({"record_type":"delta_and_threshold_crossing","comparison":f"{mh}_vs_RGB","condition":mh,"true_class":truth,"count":len(d),"mean_delta":float(np.mean(d)),"median_delta":float(np.median(d)),"p05_delta":float(np.quantile(d,.05)),"p25_delta":float(np.quantile(d,.25)),"p75_delta":float(np.quantile(d,.75)),"p95_delta":float(np.quantile(d,.95)),"threshold_cross_up_count":int(((r<.5)&(o>=.5)).sum()),"threshold_cross_up_fraction":float(((r<.5)&(o>=.5)).mean()),"threshold_cross_down_count":int(((r>=.5)&(o<.5)).sum()),"threshold_cross_down_fraction":float(((r>=.5)&(o<.5)).mean())})
    return rows


def fold_metrics(frame: pd.DataFrame) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    rows=[]
    for condition in CONDITIONS:
        for scope, sub in [("pooled_oof",frame[frame.condition==condition])]+[(f"outer_fold_{k}",frame[(frame.condition==condition)&(frame.fold==k)]) for k in sorted(frame.fold.unique())]:
            m=compute_binary_metrics(sub.binary_label,sub[["prob_control","prob_patient"]].to_numpy())
            rows.append({"condition":condition,"scope":scope,**{x:float(m[x]) for x in ("macro_auc","macro_f1","balanced_accuracy","sensitivity","specificity")}})
    result=pd.DataFrame(rows); deltas=[]
    for scope in result.scope.drop_duplicates():
        ref=float(result[(result.condition=="RGB")&(result.scope==scope)].macro_auc.iloc[0])
        for condition in ("RGB_B1MH","RGB_B2MH"): deltas.append({"condition":condition,"scope":scope,"macro_auc_delta_vs_RGB":float(result[(result.condition==condition)&(result.scope==scope)].macro_auc.iloc[0]-ref)})
    return rows,deltas


def calibration(frame: pd.DataFrame) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    summary=[]; bins=[]; edges=np.linspace(0,1,11)
    for condition in CONDITIONS:
        sub=frame[frame.condition==condition]; y=sub.binary_label.to_numpy(); p=sub.prob_patient.to_numpy(); ids=np.minimum((p*10).astype(int),9); ece=0.
        for i in range(10):
            mask=ids==i; n=int(mask.sum()); mean=float(p[mask].mean()) if n else np.nan; observed=float(y[mask].mean()) if n else np.nan
            if n: ece+=n/len(sub)*abs(mean-observed)
            bins.append({"condition":condition,"bin_index":i,"bin_range":f"[{edges[i]:.1f}, {edges[i+1]:.1f}{']' if i==9 else ')'}","count":n,"mean_predicted_probability":mean,"observed_patient_fraction":observed})
        summary.append({"condition":condition,"brier_score":float(brier_score_loss(y,p)),"ece_10_equal_width":float(ece)})
    return summary,bins


def misclassification(frame: pd.DataFrame) -> list[dict[str,Any]]:
    out=[]; errors={}
    for condition in CONDITIONS:
        sub=frame[frame.condition==condition].set_index("ID"); fp=set(sub[(sub.binary_label==0)&(sub.pred_class==1)].index); fn=set(sub[(sub.binary_label==1)&(sub.pred_class==0)].index); errors[condition]=(fp,fn); out += [{"comparison":condition,"error_type":"false_positive","category":"count","count":len(fp)},{"comparison":condition,"error_type":"false_negative","category":"count","count":len(fn)}]
    for kind,index in [("false_positive",0),("false_negative",1)]:
        rgb,b2=errors["RGB"][index],errors["RGB_B2MH"][index]
        for category,values in [("shared",rgb&b2),("RGB_only",rgb-b2),("B2_only",b2-rgb)]: out.append({"comparison":"RGB_B2MH_vs_RGB","error_type":kind,"category":category,"count":len(values)})
    return out


def run(root:Path=ROOT)->dict[str,Any]:
    root=root.resolve(); before=_snapshot(root); frame=_load(root); report=root/"reports/so_r2x3_oof_readonly_diagnostic"
    transitions=correctness_transitions(frame,"RGB_B2MH")+correctness_transitions(frame,"RGB_B1MH")
    probability=probability_summary(frame,"RGB_B2MH")+probability_summary(frame,"RGB_B1MH")
    fold_rows, auc_delta=fold_metrics(frame); calib, bin_rows=calibration(frame); overlap=misclassification(frame)
    report.mkdir(parents=True,exist_ok=True); pd.DataFrame(transitions).to_csv(report/"correctness_transition_tables.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(probability).to_csv(report/"probability_shift_summary.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(fold_rows+auc_delta).to_csv(report/"fold_comparison_summary.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(calib).to_csv(report/"calibration_summary.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(bin_rows).to_csv(report/"calibration_bin_tables.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(overlap).to_csv(report/"misclassification_overlap.csv",index=False,encoding="utf-8-sig")
    b2_all=[x for x in probability if x["record_type"]=="delta_and_threshold_crossing" and x["comparison"]=="RGB_B2MH_vs_RGB" and x["true_class"]=="all cases"][0]
    after=_snapshot(root); unchanged=before==after
    _dump(report/"readonly_asset_hash_audit.json",{"unchanged":unchanged,"before":before,"after":after}); _dump(report/"field_access_audit.json",{"allowed_oof_columns":OOF_COLUMNS,"forbidden_columns_loaded":[],"model_forward_calls":0,"training_calls":0,"optimizer_step_count":0,"checkpoint_write_count":0,"OOF_modification_count":0})
    (report/"test_results.txt").write_text("readonly_input_hash_unchanged="+str(unchanged)+"\nmodel_forward_calls=0\ntraining_calls=0\n",encoding="utf-8")
    text=("# SO-R2-X3-D Read-Only OOF Diagnostic\n\nAll findings are descriptive diagnostics on fixed internal OOF predictions. No model forward, training, calibration, threshold modification, winner selection, or OOF rewrite occurred.\n\n"
          f"B2 patient-probability mean delta vs RGB (all cases): {b2_all['mean_delta']:.6f}; upward 0.5 crossings: {b2_all['threshold_cross_up_count']}; downward crossings: {b2_all['threshold_cross_down_count']}.\n\n"
          "AUC measures ranking across thresholds, whereas Macro-F1, balanced accuracy, sensitivity and specificity here use the frozen argmax/0.5 rule; therefore a small ranking change need not improve threshold-dependent metrics. See CSV artifacts for full class-specific transitions, probability shifts, calibration bins, fold heterogeneity, and error overlap.\n")
    (report/"SO_R2X3_D_ReadOnly_OOF_Diagnostic_Report.md").write_text(text,encoding="utf-8")
    result={"status":"COMPLETE_READONLY_OOF_DIAGNOSTIC" if unchanged else "FAIL_READONLY_ASSET_MUTATION","model_forward_calls":0,"training_calls":0,"optimizer_step_count":0,"checkpoint_write_count":0,"OOF_modification_count":0,"formal_SO_R1_C_status":"FAIL","official_SO_R2_authorization":False,"SO_R3_authorization":False,"next_stage_authorized":False,"b2_probability_mean_delta_vs_rgb":b2_all["mean_delta"],"b2_threshold_cross_up":b2_all["threshold_cross_up_count"],"b2_threshold_cross_down":b2_all["threshold_cross_down_count"]}
    _dump(root/"data/processed/SO_R2X3_OOFDiagnostic_v1/X3D_ACCEPTANCE.json",result); return result
