"""Torch reflectance formulae."""

from __future__ import annotations

import torch


def finite_dermis_reflectance(mu_a: torch.Tensor, mu_s_prime: torch.Tensor, dermis_thickness_cm: float) -> torch.Tensor:
    """Finite dermis reflectance with explicit zero-absorption limits."""

    denom = mu_a + 2.0 * mu_s_prime
    beta = torch.sqrt(torch.clamp(mu_a / denom, min=0.0))
    kappa = torch.sqrt(torch.clamp(mu_a * denom, min=0.0))
    tau = torch.tanh(kappa * dermis_thickness_cm)
    normal = ((1.0 - beta**2) * tau) / ((1.0 + beta**2) * tau + 2.0 * beta)
    limit = (mu_s_prime * dermis_thickness_cm) / (1.0 + mu_s_prime * dermis_thickness_cm)
    zero_abs_scatter = (mu_a == 0.0) & (mu_s_prime > 0.0)
    zero_all = (mu_a == 0.0) & (mu_s_prime == 0.0)
    return torch.where(zero_all, torch.zeros_like(normal), torch.where(zero_abs_scatter, limit, normal))


def compute_skin_reflectance(
    m: torch.Tensor,
    h: torch.Tensor,
    mua_mel_primary: torch.Tensor,
    mua_base: torch.Tensor,
    mua_blood_y075: torch.Tensor,
    musp_total: torch.Tensor,
    epidermis_thickness_cm: float = 0.006,
    dermis_thickness_cm: float = 0.20,
    validate_range: bool = True,
) -> torch.Tensor:
    """Compute differentiable synthetic skin reflectance."""

    if validate_range:
        if not bool(torch.all(torch.isfinite(m)) and torch.all(torch.isfinite(h))):
            raise ValueError("Controls contain non-finite values")
        if not bool(torch.all((m >= 0.0) & (m <= 1.0)) and torch.all((h >= 0.0) & (h <= 1.0))):
            raise ValueError("Controls outside [0,1]")
    f_mel = (0.013 + 0.417 * m).unsqueeze(-1)
    f_blood = (0.02 + 0.05 * h).unsqueeze(-1)
    epi_abs = f_mel * mua_mel_primary + (1.0 - f_mel) * mua_base
    derm_abs = f_blood * mua_blood_y075 + (1.0 - f_blood) * mua_base
    transmission = torch.exp(-epi_abs * epidermis_thickness_cm)
    dermis = finite_dermis_reflectance(derm_abs, musp_total, dermis_thickness_cm)
    return transmission * transmission * dermis
