import numpy as np
from src.skin_optics_so_r1.highclip_attribution import metrics

def test_element_and_pixel_clip_metrics_are_distinct():
    x=np.array([[[1.,.5,.5],[1.,1.,1.]]],dtype='float32'); m=np.ones((1,2),dtype='uint8')
    q=metrics(x,m)
    assert q['high_clip_element_fraction']==4/6
    assert q['high_clip_pixel_any_fraction']==1.
    assert q['high_clip_pixel_all_fraction']==.5

def test_clip_boundary_is_inclusive():
    x=np.array([[[1-1e-6,.1,.1]]],dtype='float32'); q=metrics(x,np.ones((1,1),dtype='uint8'))
    assert q['R_high_clip_fraction']==1.
