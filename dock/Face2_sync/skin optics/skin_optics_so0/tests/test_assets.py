from skin_optics.assets import load_assets, read_standardization_decision


def test_assets_decision_and_shapes(assets5, assets1):
    decision = read_standardization_decision()
    assert decision["status"] == "PASS_WITH_5NM_FALLBACK"
    assert decision["production_step_nm"] == 5
    assert assets1.n_lambda == 321
    assert assets5.n_lambda == 65
    assert assets5.source["camera_ssf"].shape == (28, 65, 3)


def test_reject_10nm():
    import pytest

    with pytest.raises(ValueError):
        load_assets(10)
