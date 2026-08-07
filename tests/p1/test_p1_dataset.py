from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from p0b_deca.p1_dataset import P1FrozenAssetDataset, P1RepresentationAdapter, UnavailableRepresentationError


class P1DatasetTests(unittest.TestCase):
    def _fixture(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        case = root / "case"
        case.mkdir()
        rgb = np.array([[[0, 128, 255], [10, 20, 30]], [[40, 50, 60], [70, 80, 90]]], dtype=np.uint8)
        Image.fromarray(rgb).save(root / "rgb.png")
        for name in ("face_valid.png", "skin_strict.png", "physics_core.png", "final_face.png"):
            Image.fromarray(np.array([[255, 0], [255, 255]], dtype=np.uint8)).save(root / name)
        maps = {
            "input_aligned_rgb": rgb.astype(np.float32) / 255.0,
            "albedo_like": np.full((2, 2, 3), 0.5, np.float32),
            "normal_coarse": np.array([[[-1.2, 0.0, 1.2]] * 2] * 2, dtype=np.float32),
            "shading_like": np.full((2, 2, 3), 1.4, np.float32),
            "reconstruction": np.full((2, 2, 3), 0.2, np.float32),
            "signed_residual": np.array([[[-0.2, 0.3, -0.4]] * 2] * 2, dtype=np.float32),
            "absolute_residual": np.array([[[0.2, 0.3, 0.4]] * 2] * 2, dtype=np.float32),
        }
        np.savez(case / "maps.npz", **maps)
        np.savez(
            case / "latents.npz",
            shape_code=np.ones((1, 100), np.float32),
            tex_code=np.ones((1, 50), np.float32),
            detail_code=np.ones((1, 128), np.float32),
            expression_code=np.ones((1, 50), np.float32),
            pose_code=np.ones((1, 6), np.float32),
            camera_code=np.ones((1, 3), np.float32),
            light_code=np.ones((1, 9, 3), np.float32),
        )
        (case / "quality.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        manifest = pd.DataFrame([{
            "case_id": "case",
            "patient_id": "p",
            "group_id": "p",
            "fold": 0,
            "label_original": 0,
            "label_3class": 0,
            "label_binary": 0,
            "rgb_path": str(root / "rgb.png"),
            "face_valid_mask_path": str(root / "face_valid.png"),
            "skin_strict_mask_path": str(root / "skin_strict.png"),
            "physics_core_skin_mask_path": str(root / "physics_core.png"),
            "maps_path": str(case / "maps.npz"),
            "latents_path": str(case / "latents.npz"),
            "quality_path": str(case / "quality.json"),
            "camera_model": "camera",
            "exposure_time_raw": "0.01",
            "exposure_time_seconds": 0.01,
            "log_exposure_time": -4.605,
            "fnumber": 1.8,
            "iso_raw": "100",
            "iso_numeric": 100,
            "log_iso": 4.605,
            "brightness_value": 1.0,
            "datetime_original": "2024:01:01 09:00:00",
            "shooting_time_period": "morning",
        }])
        path = root / "manifest.csv"
        manifest.to_csv(path, index=False)
        return td, path

    def test_lazy_load_uses_np_load_on_getitem(self):
        td, manifest = self._fixture()
        self.addCleanup(td.cleanup)
        dataset = P1FrozenAssetDataset(manifest)
        with patch("numpy.load", wraps=np.load) as mocked:
            self.assertEqual(mocked.call_count, 0)
            _ = dataset[0]
            self.assertGreaterEqual(mocked.call_count, 2)

    def test_outputs_are_chw_float32_and_preserve_raw_values(self):
        td, manifest = self._fixture()
        self.addCleanup(td.cleanup)
        item = P1FrozenAssetDataset(manifest)[0]
        self.assertEqual(tuple(item["rgb"].shape), (3, 2, 2))
        self.assertEqual(str(item["rgb"].dtype), "torch.float32")
        self.assertEqual(tuple(item["face_valid"].shape), (1, 2, 2))
        self.assertLess(float(item["signed_residual"].min()), 0)
        self.assertLess(float(item["normal_coarse"].min()), -1.0)
        self.assertGreater(float(item["shading_like"].max()), 1.0)
        self.assertEqual(tuple(item["light_code"].shape), (9, 3))

    def test_representation_adapter_masks_and_specular_status(self):
        td, manifest = self._fixture()
        self.addCleanup(td.cleanup)
        item = P1FrozenAssetDataset(manifest)[0]
        adapter = P1RepresentationAdapter()
        self.assertEqual(tuple(adapter(item, "albedo").shape), (3, 2, 2))
        self.assertEqual(tuple(adapter(item, "light").shape), (9, 3))
        paired = adapter(item, "rgb_albedo")
        self.assertEqual(set(paired), {"rgb", "albedo"})
        with self.assertRaisesRegex(UnavailableRepresentationError, "unavailable_by_current_frontend"):
            adapter(item, "specular")


if __name__ == "__main__":
    unittest.main()
