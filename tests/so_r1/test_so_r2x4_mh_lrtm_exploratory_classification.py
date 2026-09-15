from src.skin_optics_so_r1.mh_lrtm_fusion import FusionNet


def test_x4_fusion_parameter_lock_and_late_head_contract():
    expected = {"LATE_CONCAT": 13126762, "MH_LRF": 13212140, "MH_LRTM": 13376940}
    for variant, count in expected.items():
        model = FusionNet(variant)
        assert sum(p.numel() for p in model.parameters()) == count
        assert (not hasattr(model, "rgb_head")) is (variant == "LATE_CONCAT")
