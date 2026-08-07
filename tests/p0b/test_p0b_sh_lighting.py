import numpy as np
from p0b_deca.sh_lighting import DirectionalToSHProjector,render_sh
def test_projector_outputs_shared_rgb_channels():
 c,e=DirectionalToSHProjector(512).fit(np.array([0,0,1.]),.3,.7); assert c.shape==(9,3) and np.allclose(c[:,0],c[:,1]) and np.allclose(c[:,1],c[:,2]) and e>=0

def test_fixed_preset_semantics_are_monotonic_and_directional():
 projector=DirectionalToSHProjector(4096); normals=np.array([[-.7,0,.7],[.7,0,.7],[0,-.7,.7],[0,0,1.]],float); normals/=np.linalg.norm(normals,axis=1,keepdims=True)
 dim,_=projector.fit(np.array([0,0,1.]),.12,.30); neutral,_=projector.fit(np.array([0,0,1.]),.35,.65); bright,_=projector.fit(np.array([0,0,1.]),.60,1.0)
 left,_=projector.fit(np.array([-.7,0,.7]),.25,.75); right,_=projector.fit(np.array([.7,0,.7]),.25,.75); top,_=projector.fit(np.array([0,-.7,.7]),.25,.75)
 assert render_sh(normals,dim)[3,0] < render_sh(normals,neutral)[3,0] < render_sh(normals,bright)[3,0]
 assert render_sh(normals,left)[0,0] > render_sh(normals,left)[1,0]
 assert render_sh(normals,right)[1,0] > render_sh(normals,right)[0,0]
 assert render_sh(normals,top)[2,0] > render_sh(normals,top)[3,0]
