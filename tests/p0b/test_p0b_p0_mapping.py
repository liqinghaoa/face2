import numpy as np
from p0b_deca.metrics import residual_p0
from p0b_deca.p0_mapper import map_to_p0
def test_residual_recomputed_in_p0_space():
 a=np.ones((2,2,3),np.float32); b=np.zeros_like(a); signed,absolute=residual_p0(a,b); assert np.all(signed==1) and np.all(absolute==1)

def test_identity_mapping_preserves_raster_and_mask_semantics():
 value=np.zeros((224,224),np.uint8); value[20:40,30:60]=255
 assert np.array_equal(map_to_p0(value,np.eye(3),"nearest"),value)
