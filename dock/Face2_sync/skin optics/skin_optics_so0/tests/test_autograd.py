import torch

from skin_optics.torch_backend.forward_model import SO0TorchForwardModel


def test_torch_autograd_gradcheck(assets5):
    model = SO0TorchForwardModel(assets5, dtype=torch.float64)
    m = torch.tensor([0.45], dtype=torch.float64, requires_grad=True)
    h = torch.tensor([0.55], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, y: model.compute_skin_reflectance(x, y), (m, h), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_torch_gradients_finite_nonzero(assets5):
    model = SO0TorchForwardModel(assets5, dtype=torch.float64)
    m = torch.tensor([0.45], dtype=torch.float64, requires_grad=True)
    h = torch.tensor([0.55], dtype=torch.float64, requires_grad=True)
    shading = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    specular = torch.tensor([0.03], dtype=torch.float64, requires_grad=True)
    exposure = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    y = model.render_cie_reference(m, h, shading, specular, exposure)["linear_srgb_unclipped"].sum()
    grads = torch.autograd.grad(y, (m, h, shading, specular, exposure))
    assert all(torch.all(torch.isfinite(g)) for g in grads)
    assert all(torch.any(torch.abs(g) > 0) for g in grads)
