import pandas as pd
import pytest
from preprocessing.p0_physics_assets.metadata import validate_split

def _split():
    rows=[]
    for label,count in ((0,115),(1,237),(2,148)):
        for i in range(count):
            index=len(rows); group=str(index) if index < 483 else str(355 + (index % 5))
            rows.append({'ID':str(index),'patient_group_id':group,'SEX':i%2,'sex_name':'female','NYHA':label,'label_3class':label,'label_3class_name':'x','fold':index%5})
    return pd.DataFrame(rows)

def test_binary_labels_and_control_name():
    value=validate_split(_split())
    assert value.binary_label.value_counts().to_dict()=={1:385,0:115}
    assert set(value[value.binary_label==0].binary_name)=={'Control'}

def test_group_cross_fold_is_rejected():
    value=_split(); value.loc[0,'patient_group_id']='same'; value.loc[1,'patient_group_id']='same'; value.loc[1,'fold']=1
    with pytest.raises(ValueError,match='crosses'): validate_split(value)
