from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader

from datasets.p1_component_dataset import P1ComponentDataset
from evaluators.p1_component_evaluator import P1ComponentEvaluator
from models.p1_light_linear_probe import P1LightLinearProbe
from trainers.p1_component_trainer import P1ComponentTrainer, Phase2ExecutionNotApprovedError
from utils.p1_representation_normalization import build_training_collate, fit_component_normalization
from utils.p1_sweep_state import load_state, save_state, state_from_status, validate_resume_contract


def _frame(tmp: Path) -> pd.DataFrame:
    Image.fromarray(np.full((2, 2, 3), 128, dtype=np.uint8)).save(tmp / "rgb.png")
    Image.fromarray(np.array([[255, 0], [255, 255]], dtype=np.uint8)).save(tmp / "mask.png")
    np.savez(tmp / "maps.npz", albedo_like=np.full((2, 2, 3), 0.5, np.float32), normal_coarse=np.full((2, 2, 3), -0.5, np.float32), shading_like=np.full((2, 2, 3), 0.5, np.float32), signed_residual=np.full((2, 2, 3), 0.1, np.float32))
    np.savez(tmp / "latents.npz", light_code=np.arange(27, dtype=np.float32).reshape(1, 9, 3))
    rows = []
    for idx, label in enumerate([0, 1, 0, 1]):
        rows.append(
            {
                "case_id": f"c{idx}",
                "patient_group_id": "g1" if idx < 2 else "g2",
                "group_id": "g1" if idx < 2 else "g2",
                "fold": 0 if idx < 2 else 1,
                "label_original": 3 if label else 0,
                "label_3class": 2 if label else 0,
                "label_binary": label,
                "rgb_path": str(tmp / "rgb.png"),
                "face_valid_mask_path": str(tmp / "mask.png"),
                "physics_core_skin_mask_path": str(tmp / "mask.png"),
                "maps_path": str(tmp / "maps.npz"),
                "latents_path": str(tmp / "latents.npz"),
            }
        )
    return pd.DataFrame(rows)


def test_sweep_state_round_trip_and_resume_hash_validation():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.json"
        state = state_from_status("p1_a", "FRAMEWORK_VALIDATED", "p1_component_stage1", config_sha256="cfg", split_sha256="split", manifest_sha256="man")
        save_state(state, path)
        loaded = load_state(path)
        assert loaded.status == "FRAMEWORK_VALIDATED"
        validate_resume_contract(loaded, current_config_sha256="cfg", current_split_sha256="split", current_manifest_sha256="man", current_contract_version="p1_component_stage1")


def test_trainer_validate_only_and_phase2_guard():
    with tempfile.TemporaryDirectory() as td:
        frame = _frame(Path(td))
        trainer = P1ComponentTrainer()
        config = {"frame": frame, "experiment_key": "p1_l", "seed": 2026, "batch_size": 2}
        result = trainer.fit_fold(config, 0, Path(td) / "out", "validate_only")
        assert result["status"] == "FRAMEWORK_VALIDATED"
        try:
            trainer.fit_fold(config, 0, Path(td) / "out2", "formal")
        except Phase2ExecutionNotApprovedError as exc:
            assert "PHASE2_EXECUTION_NOT_APPROVED" in str(exc)
        else:
            raise AssertionError("formal mode should require phase2 approval")


def test_evaluator_emits_case_level_oof_and_bootstrap():
    with tempfile.TemporaryDirectory() as td:
        frame = _frame(Path(td))
        dataset = P1ComponentDataset(frame, "p1_l")
        state = fit_component_normalization(frame, "p1_l")
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=build_training_collate("p1_l", state))
        model = P1LightLinearProbe()
        evaluator = P1ComponentEvaluator(model, output_dir=Path(td) / "eval")
        case_frame, payload = evaluator.evaluate_loader(loader, iterations=20, seed=2026)
        assert len(case_frame) == 4
        assert payload["cluster_unit"] == "patient_group_id"
        assert payload["metric_unit"] == "visit_case"
        assert "bootstrap" in payload

