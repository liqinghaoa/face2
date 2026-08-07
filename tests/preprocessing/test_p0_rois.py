import numpy as np
from preprocessing.p0_physics_assets.roi_builder import effective_mask

def test_lip_effective_uses_only_lip_classes_and_source_valid():
    geo=np.full((2,3),255,np.uint8); labels=np.array([[12,13,11],[1,12,13]],np.uint8); source=np.array([[255,0,255],[255,255,255]],np.uint8); got=effective_mask('lip',geo,labels,source,geo,geo)
    assert np.array_equal(got,np.array([[255,0,0],[0,255,255]],np.uint8))

def test_combined_cheek_is_logical_union():
    left=np.array([[255,0],[0,0]],np.uint8); right=np.array([[0,0],[0,255]],np.uint8)
    assert np.array_equal(((left>0)|(right>0)).astype(np.uint8)*255,np.array([[255,0],[0,255]],np.uint8))
