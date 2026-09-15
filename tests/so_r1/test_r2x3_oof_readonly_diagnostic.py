import numpy as np
import pandas as pd
from src.skin_optics_so_r1.realface_oof_readonly_diagnostic import calibration, correctness_transitions, probability_summary, fold_metrics

def _frame():
    rows=[]
    for c,probs in {"RGB":[.2,.7,.4,.8],"RGB_B1MH":[.3,.6,.3,.9],"RGB_B2MH":[.6,.8,.2,.7]}.items():
        for i,p in enumerate(probs): rows.append({"ID":str(i),"condition":c,"fold":i%2,"binary_label":int(i>=2),"prob_control":1-p,"prob_patient":p,"pred_class":int(p>=.5),"selected_epoch":1})
    return pd.DataFrame(rows)

def test_transition_and_crossing_counts_are_paired():
    f=_frame(); rows=correctness_transitions(f,"RGB_B2MH"); assert sum(x["count"] for x in rows if x["true_class"]=="all cases")==4
    shifts=probability_summary(f,"RGB_B2MH"); allrow=[x for x in shifts if x["record_type"]=="delta_and_threshold_crossing" and x["true_class"]=="all cases"][0]; assert allrow["threshold_cross_up_count"]==1

def test_fixed_calibration_has_ten_bins_without_adjusting_probabilities():
    f=_frame(); summary,bins=calibration(f); assert len(summary)==3 and len(bins)==30 and np.isfinite(summary[0]["brier_score"])

def test_fold_auc_deltas_have_one_row_per_comparison_and_scope():
    f=_frame(); _,deltas=fold_metrics(f)
    assert len(deltas)==6 and len({(x["condition"],x["scope"]) for x in deltas})==6
