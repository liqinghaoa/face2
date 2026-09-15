from pathlib import Path
from src.skin_optics_so_r1.realface_b1_mh_three_class_exploratory import inner_split, load_table, three_class_metrics

ROOT=Path(__file__).resolve().parents[2]
def test_x5_mapping_and_group_aware_inner_splits():
    table=load_table(ROOT)
    assert table.class3.value_counts().sort_index().tolist()==[115,238,147]
    for fold in range(5):
        tr,va=inner_split(table[table.fold!=fold],fold)
        assert not(set(tr.patient_group_id)&set(va.patient_group_id))
        assert set(va.class3)=={0,1,2}
def test_x5_multiclass_metrics_contract():
    p=[[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]*2
    m=three_class_metrics([0,1,2]*2,p)
    assert m['macro_ovr_auc']==1.0 and m['confusion_matrix'].shape==(3,3)
