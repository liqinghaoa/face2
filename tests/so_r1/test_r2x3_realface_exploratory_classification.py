from pathlib import Path
import pandas as pd
from src.skin_optics_so_r1.realface_exploratory_classification import _inner_split, _model

def test_nested_direct_split_is_320_80_and_leak_free():
    root=Path(__file__).resolve().parents[2]
    table=pd.read_csv(root/'data/processed/P0_Physics_Audit_v1/splits_500/nyha_2class_sex_stratified_group_5fold.csv',usecols=['ID','patient_group_id','fold','binary_label','SEX'])
    dev=table[table.fold!=0]; train,val=_inner_split(dev,0)
    assert len(train)==320 and len(val)==80 and not set(train.ID)&set(val.ID)

def test_five_channel_model_keeps_rgb_and_mean_initializes_mh():
    import torch
    rgb=_model(3); five=_model(5)
    assert five.conv1.weight.shape[1]==5 and five.fc.out_features==2
    assert torch.allclose(five.conv1.weight[:,3],five.conv1.weight[:,4])
