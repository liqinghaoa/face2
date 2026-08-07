import numpy as np
from preprocessing.p0_physics_assets.regression import evaluate_regression
from preprocessing.p0_physics_assets.types import RegressionConfig

def test_identical_rgb_passes_regression():
    image=np.full((16,16,3),100,np.uint8); result=evaluate_regression(image,image,RegressionConfig())
    assert result.status=='passed' and result.mae==0 and result.rgb_ssim==1

def test_large_rgb_difference_fails_regression():
    result=evaluate_regression(np.zeros((16,16,3),np.uint8),np.full((16,16,3),255,np.uint8),RegressionConfig())
    assert result.status=='failed' and result.large_diff_fraction==1
