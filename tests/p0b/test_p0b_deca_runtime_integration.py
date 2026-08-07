from __future__ import annotations

import unittest
from pathlib import Path

import torch

from p0b_deca.deca_runtime import forward_deca, load_deca


@unittest.skipUnless(torch.cuda.is_available(), "requires the P0-B CUDA environment")
class DecaRuntimeIntegrationTests(unittest.TestCase):
    def test_core_forward_is_finite_without_fan_or_testdata(self) -> None:
        model, cfg = load_deca(Path("third_party/DECA"), "cuda", 42)
        self.assertTrue(cfg.model.use_tex)
        image = torch.full((1, 3, 224, 224), 0.5, device="cuda")
        _, output, _ = forward_deca(model, image)
        for key in ("verts", "rendered_images", "alpha_images", "albedo", "uv_detail_normals"):
            self.assertTrue(torch.isfinite(output[key]).all().item(), key)


if __name__ == "__main__":
    unittest.main()
