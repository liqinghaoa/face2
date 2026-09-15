from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.skin_optics_so_r1 import baseline_training as bt
from src.skin_optics_so_r1 import proposed_training as pt
from src.skin_optics_so_r1 import synthetic_id_ood_evaluation as ce


def test_c_accepts_only_frozen_authoritative_inputs_and_accepted_g1_data():
    _, _, rows = bt.validate_authoritative_input(Path("."))
    b1, b2, selected, paths = ce._authority(Path("."), rows)
    assert b1["status"] == "PASS" and b2["selected_lambda"] == pytest.approx(.5)
    assert selected["selected_checkpoint_sha256"] == ce._sha(paths["proposed"])


def test_c_test_counts_roles_and_pair_manifest_are_complete_and_isolated():
    _, _, rows = bt.validate_authoritative_input(Path("."))
    audit = ce._pair_integrity(Path("."), rows)
    assert audit["status"] == "PASS"
    assert audit["pair_counts"] == {"ID Test": 4000, "Camera-OOD": 2000, "Light-OOD": 2000, "Joint-OOD": 2000}
    assert ce.ROLES == ("A0", "A1", "A2", "A3", "A4")


def test_masked_metrics_preserve_m_h_order_and_ignore_invalid_pixels():
    prediction = np.array([[[[.2, .9]], [[.4, .8]]]], dtype=np.float32)
    target = np.array([[[[.1, .3]], [[.5, .7]]]], dtype=np.float32)
    mask = np.array([[[[1., 0.]]]], dtype=np.float32)
    result = ce.masked_map_metrics(prediction, target, mask, np.zeros((2, 1, 2), dtype=np.float32))
    assert result["mae"][0].tolist() == pytest.approx([.1, .1])
    assert result["prediction_mean"][0].tolist() == pytest.approx([.2, .4])


def test_paired_and_stratified_bootstraps_have_latent_level_equal_strata_behavior():
    paired = ce.paired_bootstrap(np.array([.1, .2, .3]), np.array([.2, .3, .4]), draws=200, seed=4)
    stratified = ce.stratified_bootstrap({"Camera-OOD": np.array([.1, .1]), "Light-OOD": np.array([.3])}, draws=200, seed=5)
    assert paired["n"] == 3 and paired["mean_delta"] == pytest.approx(-.1)
    assert stratified["mean_delta"] == pytest.approx(.2) and stratified["strata"] == {"Camera-OOD": 2, "Light-OOD": 1}


def test_primary_endpoint_is_equal_weighted_across_ood_splits():
    value = ce.stratified_bootstrap({"Camera-OOD": np.array([.1] * 500), "Light-OOD": np.array([.2]), "Joint-OOD": np.array([.3])}, draws=100, seed=1)
    assert value["mean_delta"] == pytest.approx(.2)


def test_noninferiority_uses_one_sided_five_percent_upper_bound():
    result = ce.stratified_bootstrap({"ID Test": np.array([.01] * 10)}, draws=200, seed=2, one_sided=True)
    assert result["one_sided"] and result["ci"][1] == pytest.approx(.01)


def test_protocol_is_fp32_read_only_and_excludes_train_validation_inference():
    _, _, rows = bt.validate_authoritative_input(Path("."))
    b1, b2, selected, paths = ce._authority(Path("."), rows)
    protocol = ce._protocol(b1, b2, selected, paths)
    assert protocol["read_only"] and protocol["precision"] == "FP32"
    assert tuple(protocol["splits"]) == ce.TEST_SPLITS


def test_c_input_chain_hashes_and_protected_assets_are_frozen_before_inference():
    _, _, rows = bt.validate_authoritative_input(Path("."))
    b1, b2, selected, paths = ce._authority(Path("."), rows)
    protocol = ce._protocol(b1, b2, selected, paths)
    audit = pt.audit_protected_assets(Path("."), verify_payloads=False)
    assert protocol["b1_contract_hash"] == b2["B1_training_contract_hash"]
    assert audit["status"] == "PASS"


def test_run_contract_forbids_overwriting_an_existing_c_evaluation(tmp_path):
    (tmp_path / ce.C_DATA).mkdir(parents=True)
    with pytest.raises(RuntimeError, match="REFUSE_OVERWRITE"):
        ce.run(tmp_path)


def test_mean_predictor_requires_only_train_targets_and_masks(tmp_path, monkeypatch):
    source = tmp_path / "sample.npz"
    np.savez(source, m=np.ones((256, 256), np.float32), h=np.full((256, 256), .5, np.float32), mask=np.ones((256, 256), np.float32), rgb=np.full((5, 3, 256, 256), np.nan, np.float32))
    original = np.load
    def guarded(*args, **kwargs):
        loaded = original(*args, **kwargs)
        class Guard:
            def __enter__(self): return self
            def __exit__(self, *unused): loaded.close()
            def __getitem__(self, key):
                assert key != "rgb"
                return loaded[key]
        return Guard()
    monkeypatch.setattr(ce.np, "load", guarded)
    mean, access = ce._mean_predictor_targets_only([{"split": "Train", "file_path": str(source)}])
    assert mean[:, 0, 0].tolist() == pytest.approx([1., .5])
    assert access == 1


def test_collapse_and_within_between_thresholds_are_declared():
    _, _, rows = bt.validate_authoritative_input(Path("."))
    b1, b2, selected, paths = ce._authority(Path("."), rows)
    assert ce._protocol(b1, b2, selected, paths)["within_between_min_fraction"] == .5


def test_within_between_batched_mask_retains_channel_dimension(tmp_path):
    rows=[]
    for number in range(2):
        path=tmp_path / f"L{number}.npz"
        rgb=np.full((5,3,256,256), .1+number*.1, dtype=np.float32)
        np.savez(path, rgb=rgb, m=np.zeros((256,256),np.float32), h=np.zeros((256,256),np.float32), mask=np.ones((256,256),np.float32))
        rows.append({"split":"ID Test","latent_id":f"L{number}","file_path":str(path)})
    drift=pd.DataFrame([{"model":"Test","split":"ID Test","latent_id":row["latent_id"],"pair_type":pair,"channel":channel,"masked_mae_drift":.01} for row in rows for pair,_ in ce.PAIR_TYPES.values() for channel in ("M","H")])
    class TwoChannel(torch.nn.Module):
        def forward(self, images): return images[:, :2]
    result=ce._within_between(rows,{"Test":TwoChannel()},torch.device("cpu"),drift)
    assert len(result)==len(rows)*len(ce.PAIR_TYPES)*2
    assert np.isfinite(result["between_drift"]).all()
