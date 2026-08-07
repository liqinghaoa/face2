"""Finite-thickness Kubelka-Munk dermis reflectance."""

from __future__ import annotations

import numpy as np


def finite_dermis_reflectance(mu_a: np.ndarray, mu_s_prime: np.ndarray, dermis_thickness_cm: float) -> np.ndarray:
    """Compute finite dermis reflectance with explicit zero-absorption limits."""

    mu_a = np.asarray(mu_a, dtype=np.float64)
    mu_s_prime = np.asarray(mu_s_prime, dtype=np.float64)
    if dermis_thickness_cm < 0:
        raise ValueError("dermis_thickness_cm must be non-negative")
    denom = mu_a + 2.0 * mu_s_prime
    normal = (mu_a > 0.0) & (denom > 0.0)
    out = np.zeros(np.broadcast_shapes(mu_a.shape, mu_s_prime.shape), dtype=np.float64)
    mu_a_b, mu_s_b = np.broadcast_arrays(mu_a, mu_s_prime)
    if np.any(normal):
        beta = np.sqrt(mu_a_b[normal] / (mu_a_b[normal] + 2.0 * mu_s_b[normal]))
        kappa = np.sqrt(mu_a_b[normal] * (mu_a_b[normal] + 2.0 * mu_s_b[normal]))
        tau = np.tanh(kappa * dermis_thickness_cm)
        out[normal] = ((1.0 - beta**2) * tau) / ((1.0 + beta**2) * tau + 2.0 * beta)
    zero_abs_scatter = (mu_a_b == 0.0) & (mu_s_b > 0.0)
    if np.any(zero_abs_scatter):
        path = mu_s_b[zero_abs_scatter] * dermis_thickness_cm
        out[zero_abs_scatter] = path / (1.0 + path)
    return out
