import numpy as np
from p0b_deca.metrics import coverage,finite_fraction
def test_coverage_and_finiteness():
 assert coverage(np.array([[255,0]],np.uint8),np.array([[1,0]],np.uint8))==1; assert finite_fraction(np.array([1,np.nan]))==.5
