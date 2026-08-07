import numpy as np
from p0b_deca.input_adapter import bbox_mapping,identity_mapping,mask_bbox,mask_bbox_crop
def test_bbox_mapping_inverse_and_identity():
 f,i=bbox_mapping((10,20,110,120)); p=np.array([32.,48.,1.]); assert np.allclose(i@(f@p),p); a,b=identity_mapping(); assert np.array_equal(a,b)

def test_mask_bbox_crop_is_fixed_and_preserves_inverse_mapping():
 image=np.zeros((224,224,3),np.uint8); mask=np.zeros((224,224),np.uint8); mask[50:150,80:140]=255
 assert mask_bbox(mask,.15)==(71,35,149,165)
 crop,bbox,p0_to_deca,deca_to_p0=mask_bbox_crop(image,mask,.15)
 assert crop.shape==(224,224,3) and bbox==(71,35,149,165)
 assert np.allclose(deca_to_p0@p0_to_deca,np.eye(3),atol=1e-5)
