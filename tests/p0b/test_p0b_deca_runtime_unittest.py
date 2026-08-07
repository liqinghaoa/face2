from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from p0b_deca.deca_runtime import build_deca_config
from p0b_deca.input_mode_comparison import _map_chw
from p0b_deca.sh_lighting import DirectionalToSHProjector, sh_basis


class DecaRuntimeUnitTests(unittest.TestCase):
    def test_single_channel_mapping_preserves_channel_axis(self) -> None:
        import torch

        value = torch.ones((1, 1, 224, 224), dtype=torch.float32)
        mapped = _map_chw(value, np.eye(3, dtype=np.float32), "nearest")
        self.assertEqual(mapped.shape, (224, 224, 1))
        self.assertTrue(np.all(mapped == 1))

    def test_official_config_uses_explicit_bfm_texture_and_core_paths(self) -> None:
        root = Path("third_party/DECA").resolve()
        cfg = build_deca_config(root)
        self.assertEqual(cfg.pretrained_modelpath, str(root / "data/deca_model.tar"))
        self.assertEqual(cfg.model.flame_model_path, str(root / "data/generic_model.pkl"))
        self.assertEqual(cfg.model.flame_lmk_embedding_path, str(root / "data/landmark_embedding.npy"))
        self.assertEqual(cfg.model.tex_path, str(root / "data/FLAME_albedo_from_BFM.npz"))
        self.assertTrue(cfg.model.use_tex)
        self.assertEqual(cfg.model.tex_type, "BFM")
        self.assertFalse(cfg.model.extract_tex)
        self.assertEqual(cfg.rasterizer_type, "pytorch3d")

    def test_renderer_coefficient_conversion_preserves_canonical_shading(self) -> None:
        projector = DirectionalToSHProjector(4096)
        canonical, _ = projector.fit(np.array([0.0, 0.0, 1.0]), 0.35, 0.65)
        factor = np.array([0.28, 1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 0.4, 0.3], dtype=np.float32)
        renderer, _ = projector.fit_renderer(np.array([0.0, 0.0, 1.0]), 0.35, 0.65, factor)
        normals = np.array([[0.2, -0.4, 0.894], [-0.3, 0.1, 0.949]], dtype=np.float32)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        renderer_basis = sh_basis(normals)[:, [0, 1, 2, 3, 4, 5, 6, 8, 7]]
        actual = renderer_basis @ (renderer * factor[:, None])
        expected = sh_basis(normals) @ canonical
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
