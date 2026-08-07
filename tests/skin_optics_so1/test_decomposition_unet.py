from __future__ import annotations

import torch

from skin_optics_so1.decomposition.unet_decomposer import SO1UNetDecomposer


def test_so1_unet_output_shape_range_and_finite() -> None:
    model = SO1UNetDecomposer()
    model.eval()
    with torch.inference_mode():
        out1 = model(torch.rand(1, 3, 256, 256))
        out2 = model(torch.rand(2, 3, 256, 256))
    assert tuple(out1.shape) == (1, 4, 256, 256)
    assert tuple(out2.shape) == (2, 4, 256, 256)
    assert torch.isfinite(out1).all()
    assert float(out1.min()) >= 0.0
    assert float(out1.max()) <= 1.0


def test_so1_unet_backward_reaches_main_modules() -> None:
    model = SO1UNetDecomposer()
    out = model(torch.rand(1, 3, 64, 64))
    loss = out[:, 0].mean() + out[:, 1].mean() + out[:, 2].mean() + out[:, 3].mean()
    loss.backward()
    for name in ("inc.net.0.weight", "down4.net.1.net.0.weight", "up4.conv.net.3.weight", "outc.weight"):
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
