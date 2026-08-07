from __future__ import annotations
import csv, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from p0b_deca.p1_audit import validate_case
from p0b_deca.p1_generation import BOUNDARY_IDS, CASE_FILES, LATENT_KEYS, MAP_KEYS, _atomic_commit, _failure, _pilot_latent_error, _pilot_map_error, _pilot_relighting_error, _require_pilot12_gate, _write_manifests, _write_qc, _success_valid, freeze_deca, load_config, preflight, sha, source_records


class P1FrozenGenerationTests(unittest.TestCase):
    def test_real_master_is_the_unique_500_case_source_and_preserves_boundaries(self):
        root = Path(__file__).resolve().parents[2]
        cfg = load_config(root / "config/p1/p1_deca_full500_v1.yaml", root)
        report = preflight(root, cfg)
        self.assertEqual((report["expected_cases"], report["unique_ids"]), (500, 500))
        self.assertEqual(report["fixed_input_mode"], "direct_p0_aligned")
        self.assertTrue(all(report["boundary_cases"][case]["present"] and report["boundary_cases"][case]["p0_usable"] for case in BOUNDARY_IDS))

    def test_config_is_direct_input_and_does_not_define_label_fields(self):
        root = Path(__file__).resolve().parents[2]
        text = (root / "config/p1/p1_deca_full500_v1.yaml").read_text(encoding="utf-8")
        self.assertIn("fixed_input_mode: direct_p0_aligned", text)
        self.assertNotIn("label_field", text.lower())
        self.assertNotIn("exif", text.lower())

    def test_generation_records_are_label_blind(self):
        root = Path(__file__).resolve().parents[2]; cfg = load_config(root / "config/p1/p1_deca_full500_v1.yaml", root)
        sample = source_records(root, cfg)[0]
        forbidden = ("label", "nyha", "class", "camera", "exif", "patient_group")
        self.assertFalse(any(any(token in key.lower() for token in forbidden) for key in sample))

    def test_residual_contract_and_no_specular_substitution(self):
        input_rgb = np.array([[[.2, .4, .6]]], dtype=np.float32); reconstruction = np.array([[[.1, .5, .4]]], dtype=np.float32)
        signed = input_rgb - reconstruction
        self.assertTrue(np.allclose(signed, [[[.1, -.1, .2]]]))
        self.assertTrue(np.allclose(np.abs(signed), [[[.1, .1, .2]]]))
        self.assertNotIn("specular_like", MAP_KEYS)

    def test_resume_requires_all_hashes_not_only_directory(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td); record = {"input_sha256": "i"}
            for name in CASE_FILES: (d / name).write_bytes(b"x")
            marker = {"input_sha256": "i", "effective_config_sha256": "c", "deca_checkpoint_sha256": "k", "validation_passed": True, "output_file_sha256": {name: sha(d / name) for name in CASE_FILES}}
            (d / "_SUCCESS.json").write_text(json.dumps(marker))
            self.assertTrue(_success_valid(d, record, "c", "k"))
            (d / "maps.npz").write_bytes(b"changed")
            self.assertFalse(_success_valid(d, record, "c", "k"))

    def test_atomic_commit_refuses_to_overwrite_and_is_atomic_rename(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); tmp, dest = root / ".tmp_id_uuid", root / "id"; tmp.mkdir(); (tmp / "asset").write_text("ok")
            _atomic_commit(tmp, dest)
            self.assertTrue((dest / "asset").is_file())
            with self.assertRaises(FileExistsError): _atomic_commit(root / "missing", dest)

    def test_failure_record_does_not_claim_success(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td); (out / "failures").mkdir(); record = {"case_id": "x", "input_path": Path("in.png")}
            try: raise ValueError("bad input")
            except ValueError as exc: result = _failure(out, record, exc, "cfg", "ckpt")
            self.assertEqual(result["status"], "failed")
            self.assertTrue((out / "failures/x.json").is_file())

    def test_freeze_sets_every_parameter_requires_grad_false(self):
        class P:
            def __init__(self): self.requires_grad = True
            def requires_grad_(self, value): self.requires_grad = value; return self
        class M:
            def __init__(self): self.ps = [P(), P()]
            def parameters(self): return self.ps
        model = M(); freeze_deca(model); self.assertTrue(all(not x.requires_grad for x in model.parameters()))

    def test_full_gate_requires_a_passing_complete_pilot12_record(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td); (out / "metadata").mkdir()
            with self.assertRaises(RuntimeError): _require_pilot12_gate(out, "c", "k")
            (out / "metadata/pilot12_regression.json").write_text(json.dumps({"status": "blocked", "case_count": 12, "success_cases": 0, "effective_config_sha256": "c", "deca_checkpoint_sha256": "k"}))
            with self.assertRaises(RuntimeError): _require_pilot12_gate(out, "c", "k")

    def test_pilot_code_tolerance_is_the_p0b_repeat_inference_contract(self):
        baseline = np.array([[0.2, -0.4]], dtype=np.float32)
        self.assertTrue(_pilot_latent_error(baseline + 1e-6, baseline)["passed"])
        self.assertFalse(_pilot_latent_error(baseline + 1e-3, baseline)["passed"])

    def test_pilot_map_tolerance_ignores_isolated_raster_edge_pixels(self):
        baseline = np.linspace(-1, 1, 224 * 224 * 3, dtype=np.float32).reshape(224, 224, 3)
        current = baseline.copy(); current[0, 0, 0] += 1.0
        self.assertTrue(_pilot_map_error(current, baseline)["passed"])
        self.assertFalse(_pilot_map_error(np.full((224, 224, 3), 1e-3, np.float32), baseline)["passed"])

    def test_pilot_relighting_compares_to_retained_png_with_quantization_tolerance(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "legacy.png"
            from PIL import Image
            Image.fromarray(np.full((2, 2, 3), 128, dtype=np.uint8)).save(path)
            self.assertTrue(_pilot_relighting_error(np.full((2, 2, 3), 128 / 255.0, np.float32), path)["passed"])
            self.assertFalse(_pilot_relighting_error(np.zeros((2, 2, 3), np.float32), path)["passed"])

    def test_manifests_keep_input_rows_and_failure_rows_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td); records = [{"case_id": "a", "input_sha256": "1", "p0_usable": True, "boundary_ambiguity_status": "none"}, {"case_id": "b", "input_sha256": "2", "p0_usable": True, "boundary_ambiguity_status": "jaw_neck_ambiguous"}]
            _write_manifests(out, records, {"a": {"case_id": "a", "status": "success"}, "b": {"case_id": "b", "status": "failed"}})
            with (out / "manifests/p1_deca_input_manifest.csv").open() as h: self.assertEqual(len(list(csv.DictReader(h))), 2)
            with (out / "manifests/p1_deca_failure_manifest.csv").open() as h: self.assertEqual(len(list(csv.DictReader(h))), 1)

    def test_qc_selection_is_label_free_and_uses_cached_maps(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td); d = out / "cases/a"; d.mkdir(parents=True)
            np.savez(d / "maps.npz", input_aligned_rgb=np.full((2,2,3), .2, np.float32), reconstruction=np.full((2,2,3), .3, np.float32))
            outputs = {"a": {"case_id": "a", "status": "success", "case_dir": str(d), "reconstruction_mae_full": .1, "alpha_coverage": .9, "relighting": {"neutral": {"saturated_fraction": .0}}}}
            selected = _write_qc(out, outputs, 17)
            self.assertEqual(set(selected), {"random_normal", "highest_reconstruction_error", "lowest_alpha_coverage", "highest_relighting_saturation"})
            self.assertTrue((out / "qc/qc_selection.json").is_file())

    def test_output_audit_requires_finite_complete_assets(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "case"; d.mkdir()
            np.savez(d / "latents.npz", **{key: np.ones((1, 2), np.float32) for key in LATENT_KEYS.values()})
            np.savez(d / "maps.npz", **{key: np.ones((2, 2, 3), np.float32) for key in MAP_KEYS})
            np.savez(d / "relighting.npz", preset_names=np.array([str(i) for i in range(6)]), sh_coefficients=np.ones((6,9,3),np.float32), relighted_images=np.ones((6,2,2,3),np.float32))
            for name in ("quality.json", "provenance.json", "preview.png"): (d / name).write_bytes(b"x")
            hashes = {name: sha(d / name) for name in CASE_FILES}
            (d / "_SUCCESS.json").write_text(json.dumps({"output_file_sha256": hashes, "validation_passed": True}))
            self.assertTrue(validate_case(d)["success"])
            np.savez(d / "maps.npz", reconstruction=np.array([np.nan]))
            self.assertFalse(validate_case(d)["success"])
