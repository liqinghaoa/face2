import numpy as np
from preprocessing.p0_physics_assets.mask_builder import build_base_masks

def test_strict_skin_is_intersection_of_class_one_and_face_valid():
    labels=np.array([[1,1,2],[1,0,1],[0,1,1]],np.uint8); source=np.array([[255,0,255],[255,255,255],[0,255,255]],np.uint8); final=np.array([[255,255,0],[255,255,255],[255,0,255]],np.uint8)
    masks=build_base_masks(labels,source,final,np.zeros((3,3),np.float32))
    assert np.array_equal(masks.face_valid_mask,np.array([[255,0,0],[255,255,255],[0,0,255]],np.uint8))
    assert np.array_equal(masks.skin_strict_mask,np.array([[255,0,0],[255,0,255],[0,0,255]],np.uint8))
