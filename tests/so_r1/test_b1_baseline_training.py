from pathlib import Path
import torch
from src.skin_optics_so_r1 import baseline_training as bt
from src.skin_optics_so_r1.baseline_training import _seed_from_protocol, build_epoch_groups, masked_smooth_l1

def test_schedule_is_reproducible_complete_and_pair_balanced():
    rows=[{'latent_id':f'F{i:05d}','split':'Train','file_path':'x'} for i in range(10000)]
    seed=_seed_from_protocol('a'*64); first,audit=build_epoch_groups(rows,seed,1); second,_=build_epoch_groups(rows,seed,1)
    assert [x.pair_type for x in first] == [x.pair_type for x in second]
    assert audit['unique_latent_count']==10000 and audit['duplicate_latent_count']==0
    assert set(audit['pair_type_count'].values())=={2500}

def test_masked_loss_ignores_background_and_uses_two_channels():
    target=torch.tensor([[[[0.,1.]],[[1.,0.]]]]); pred=target.clone(); pred[...,0]=0.; mask=torch.tensor([[[[0.,1.]]]])
    assert masked_smooth_l1(pred,target,mask).item()==0.


def _rows():
    return [{"latent_id": f"F{i:05d}", "split": "Train", "file_path": "train"} for i in range(10000)]


def test_authoritative_dataset_gate_accepts_only_current_g1_dataset():
    acceptance, _, rows = bt.validate_authoritative_input(Path("."))
    assert acceptance["dataset_status"] == "ACCEPTED" and len(rows) == 13500


def test_sampler_never_uses_nontrain_rows():
    rows = _rows() + [{"latent_id": "VAL", "split": "Validation", "file_path": "val"}, {"latent_id": "OOD", "split": "Camera-OOD", "file_path": "ood"}]
    groups, _ = build_epoch_groups(rows, 7, 1)
    assert len(groups) == 10000 and all(group.split == "Train" for group in groups)


def test_test_and_ood_training_sample_count_is_zero():
    rows = _rows() + [{"latent_id": name, "split": name, "file_path": name} for name in ("ID Test", "Camera-OOD", "Light-OOD", "Joint-OOD")]
    groups, _ = build_epoch_groups(rows, 17, 2)
    forbidden = {"ID Test", "Camera-OOD", "Light-OOD", "Joint-OOD"}
    assert sum(group.split in forbidden for group in groups) == 0


def test_paired_batch_is_flattened_to_independent_supervision():
    batch = {"images": torch.zeros(2,2,3,4,4), "target": torch.tensor([[[[1.]],[[2.]]],[[[3.]],[[4.]]]]).expand(2,2,4,4).clone(), "mask": torch.ones(2,1,4,4)}
    images, target, mask = bt._flatten(batch, torch.device("cpu"))
    assert images.shape == (4,3,4,4) and target.shape == (4,2,4,4) and mask.shape == (4,1,4,4)
    assert torch.equal(target[0],target[1]) and torch.equal(target[2],target[3])


def test_baseline_loss_contract_has_no_pair_or_auxiliary_term():
    source = Path(bt.__file__).read_text(encoding="utf-8")
    assert "lambda_pair\":0" in source and "pair_loss\":\"absent\"" in source
    assert "feature_loss\":\"absent\"" in source and "adversarial_loss\":\"absent\"" in source


def test_model_has_fixed_mh_output_order_and_sigmoid_head():
    model = bt.BaselineMHUNet()
    assert model.output_order == ["M-sensitive", "H-sensitive"] and model.head.out_channels == 2
    assert isinstance(model.activation, torch.nn.Sigmoid)


def test_schedule_has_no_duplicate_or_missing_train_latent():
    groups, audit = build_epoch_groups(_rows(), 9, 8)
    assert audit["duplicate_latent_count"] == 0 and audit["unique_latent_count"] == 10000
    assert {group.latent_id for group in groups} == {row["latent_id"] for row in _rows()}


def test_mean_predictor_reads_train_split_only(monkeypatch):
    def fake_load(path, indices):
        value = 0.0 if path == "train" else 1.0
        return torch.zeros(1,3,256,256).numpy(), torch.full((2,256,256),value).numpy(), torch.ones(1,256,256).numpy()
    monkeypatch.setattr(bt,"_load_npz",fake_load)
    mean, valid = bt.train_mean_predictor([{"split":"Train","file_path":"train"},{"split":"Validation","file_path":"validation"}])
    assert mean.max() == 0.0 and valid.min() == 1.0


def test_checkpoint_contains_required_provenance(tmp_path):
    model=bt.BaselineMHUNet(); optimizer=torch.optim.AdamW(model.parameters()); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=25)
    path=bt._checkpoint(tmp_path,1,model,optimizer,scheduler,"contract",{"protocol_hash":"d0"},"g1",{"latent":"hash"},"schedule","AMP")
    payload=torch.load(path,map_location="cpu",weights_only=False)
    for key in ("model_state","optimizer_state","scheduler_state","epoch","model_architecture","training_contract_hash","d0_am1_protocol_hash","g1_am1_dataset_acceptance_hash","dataset_manifest_hashes","paired_schedule_hash","selected_precision"):
        assert key in payload


def test_source_hash_audit_is_repeatable_without_payload_scan():
    first=bt.source_audit(Path("."),verify_payloads=False); second=bt.source_audit(Path("."),verify_payloads=False)
    assert first == second and first["ledger_records"] == 13500
