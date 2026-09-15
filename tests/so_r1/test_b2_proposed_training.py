from pathlib import Path

import numpy as np
import pytest
import torch

from src.skin_optics_so_r1 import baseline_training as bt
from src.skin_optics_so_r1 import proposed_training as pt


def test_b2_accepts_frozen_b1_and_accepted_g1_inputs():
    contract, _, rows, provenance = pt.load_and_validate_b1_contract(Path("."))
    assert contract["amp_contract"]["precision"] == "FP32"
    assert len(rows) == 13500 and provenance["b1_acceptance"]["status"] == "PASS"


def test_proposed_architecture_and_seed_match_b1_without_checkpoint_loading():
    contract, _, _, _ = pt.load_and_validate_b1_contract(Path("."))
    bt._configure_determinism(contract["model_init_seed"]); first=bt.BaselineMHUNet()
    bt._configure_determinism(contract["model_init_seed"]); second=bt.BaselineMHUNet()
    assert first.architecture_id == contract["model_architecture"]["id"]
    assert bt.count_parameters(first) == contract["model_architecture"]["parameter_count"]
    assert torch.equal(first.head.weight,second.head.weight)


def test_supervised_and_pair_losses_respect_mask_and_pair_both_branches_receive_gradient():
    prediction=torch.tensor([[[[[.2]],[[.3]]],[[[.6]],[[.7]]]]],requires_grad=True)
    target=torch.tensor([[[[.2]],[[.3]]]])
    mask=torch.ones(1,1,1,1)
    supervised=pt.masked_supervised_loss(prediction,target,mask); paired=pt.masked_pair_consistency_loss(prediction,mask)
    (supervised+.2*paired).backward()
    assert supervised.item() > 0 and paired.item() > 0
    assert prediction.grad[:,0].abs().sum() > 0 and prediction.grad[:,1].abs().sum() > 0


def test_pair_loss_only_compares_within_group():
    prediction=torch.zeros(2,2,2,1,1); prediction[0,1]=1
    mask=torch.ones(2,1,1,1)
    assert pt.masked_pair_consistency_loss(prediction,mask).item() > 0
    with torch.no_grad(): prediction[1]=999
    assert pt.masked_pair_consistency_loss(prediction,mask).item() > 0


def test_lambda_candidates_are_exactly_prefrozen_set():
    assert pt.LAMBDA_CANDIDATES == (.05,.10,.20,.50)


def test_noninferiority_uses_five_percent_margin_and_paired_bootstrap():
    baseline=np.full(100,.2); proposed=np.full(100,.205)
    result=pt.paired_bootstrap_noninferiority(proposed,baseline,draws=200,seed=1)
    assert np.isclose(result["margin"],.01) and result["pass"]


def test_drift_score_and_tie_break_choose_smaller_lambda_then_run_id():
    base={label:{pair:1. for pair in pt.PAIR_KEYS} for label in ("M","H")}
    good={label:{pair:.5 for pair in pt.PAIR_KEYS} for label in ("M","H")}
    assert pt.drift_score(base,good) == .5
    selected=pt.select_lambda([{"run_id":"lambda_0p10","lambda_pair":.10,"noninferiority_status":"PASS","viability_status":"PASS","drift_score":.5},{"run_id":"lambda_0p05","lambda_pair":.05,"noninferiority_status":"PASS","viability_status":"PASS","drift_score":.500000001}])
    assert selected["run_id"] == "lambda_0p05"


def test_sampler_schedule_and_forbidden_splits_remain_b1_compatible():
    rows=[{"latent_id":f"F{i:05d}","split":"Train","file_path":"x"} for i in range(10000)] + [{"latent_id":"X","split":"ID Test","file_path":"x"}]
    groups,audit=bt.build_epoch_groups(rows,983833781,1)
    assert len(groups) == 10000 and audit["duplicate_latent_count"] == 0
    assert all(group.split == "Train" for group in groups)


def test_protected_asset_audit_and_b1_schedule_gate_pass_without_payload_rescan():
    audit=pt.audit_protected_assets(Path("."),verify_payloads=False)
    assert audit["status"] == "PASS" and audit["allowlist_pair_count"] == 24


def test_streaming_validation_accepts_b1_spatial_mean_predictor(monkeypatch):
    class IdentityChannels(torch.nn.Module):
        def forward(self, images): return images[:, :2]
    target=torch.empty((2,2,256,256)); target[0,0].fill_(.5); target[1,0].fill_(.6); target[0,1].fill_(.4); target[1,1].fill_(.3)
    paired_predictions=(target-.1).unsqueeze(1).expand(-1,5,-1,-1,-1)
    batch={"images":torch.cat((paired_predictions,torch.zeros((2,5,1,256,256))),dim=2),"target":target,"mask":torch.ones((2,1,256,256))}
    monkeypatch.setattr(pt.bt,"_loader",lambda *args,**kwargs:[batch])
    rows=[{"split":"Validation","latent_id":f"L{i:05d}","file_path":"unused"} for i in range(2)]
    result=pt.evaluate_validation_supervision(IdentityChannels(),rows,np.full((2,256,256),.5,dtype=np.float32),torch.device("cpu"),workers=0,timeout_seconds=1)
    assert result["latent_count"] == 2
    assert result["metrics"]["mae"] == pytest.approx([.1,.1])
