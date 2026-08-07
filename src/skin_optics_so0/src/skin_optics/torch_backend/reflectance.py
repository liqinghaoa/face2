"""Torch reflectance formulae."""

from __future__ import annotations

import torch

from skin_optics.config import SO0Config, load_so0_config


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
    epidermis_thickness_cm: float | None = None,
    dermis_thickness_cm: float | None = None,
    validate_range: bool = True,
    config: SO0Config | None = None,
    melanin_fraction_min: float | None = None,
    melanin_fraction_span: float | None = None,
    blood_fraction_min: float | None = None,
    blood_fraction_span: float | None = None,
) -> torch.Tensor:
    """Compute differentiable synthetic skin reflectance."""

    cfg = config or load_so0_config()
    if validate_range:
        if not bool(torch.all(torch.isfinite(m)) and torch.all(torch.isfinite(h))):
            raise ValueError("Controls contain non-finite values")
        m_def = cfg.parameter("melanin_control")
        h_def = cfg.parameter("hemoglobin_control")
        if not bool(
            torch.all((m >= float(m_def["min"])) & (m <= float(m_def["max"])))
            and torch.all((h >= float(h_def["min"])) & (h <= float(h_def["max"])))
        ):
            raise ValueError("Controls outside [0,1]")
    mel_def = cfg.parameter("melanin_fraction")
    blood_def = cfg.parameter("blood_fraction")
    mel_min = float(mel_def["min"]) if melanin_fraction_min is None else float(melanin_fraction_min)
    mel_span = float(mel_def["max"]) - float(mel_def["min"]) if melanin_fraction_span is None else float(melanin_fraction_span)
    blood_min = float(blood_def["min"]) if blood_fraction_min is None else float(blood_fraction_min)
    blood_span = float(blood_def["max"]) - float(blood_def["min"]) if blood_fraction_span is None else float(blood_fraction_span)
    epi_d = cfg.primary_value("epidermis_thickness_cm") if epidermis_thickness_cm is None else float(epidermis_thickness_cm)
    der_d = cfg.primary_value("dermis_thickness_cm") if dermis_thickness_cm is None else float(dermis_thickness_cm)
    f_mel = (mel_min + mel_span * m).unsqueeze(-1)
    f_blood = (blood_min + blood_span * h).unsqueeze(-1)
    epi_abs = f_mel * mua_mel_primary + (1.0 - f_mel) * mua_base
    derm_abs = f_blood * mua_blood_y075 + (1.0 - f_blood) * mua_base
    transmission = torch.exp(-epi_abs * epi_d)
    dermis = finite_dermis_reflectance(derm_abs, musp_total, der_d)
    return transmission * transmission * dermis
