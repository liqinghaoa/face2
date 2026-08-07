"""Torch image formation utilities."""

from __future__ import annotations

import torch

from skin_optics.torch_backend.color_spaces import bradford_adapt, srgb_encode, xyz_to_linear_srgb


def spectral_radiance(
    reflectance: torch.Tensor,
    illuminant_y1: torch.Tensor,
    shading: torch.Tensor,
    specular: torch.Tensor,
    exposure: torch.Tensor,
) -> torch.Tensor:
    """Build relative spectral radiance."""

    return exposure.unsqueeze(-1) * illuminant_y1 * (shading.unsqueeze(-1) * reflectance + specular.unsqueeze(-1))


def integrate_xyz(radiance: torch.Tensor, weights_nm: torch.Tensor, cie_xyz: torch.Tensor) -> torch.Tensor:
    """Integrate radiance to XYZ."""

    return radiance @ (weights_nm.unsqueeze(-1) * cie_xyz)


def render_cie_reference(
    reflectance: torch.Tensor,
    weights_nm: torch.Tensor,
    cie_xyz: torch.Tensor,
    illuminant_y1: torch.Tensor,
    source_white: torch.Tensor,
    d65_white: torch.Tensor,
    shading: torch.Tensor,
    specular: torch.Tensor,
    exposure: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Torch CIE path."""

    rad = spectral_radiance(reflectance, illuminant_y1, shading, specular, exposure)
    xyz_source = integrate_xyz(rad, weights_nm, cie_xyz)
    xyz_d65 = bradford_adapt(xyz_source, source_white, d65_white)
    linear = xyz_to_linear_srgb(xyz_d65)
    srgb = srgb_encode(linear)
    return {
        "xyz_source": xyz_source,
        "xyz_d65": xyz_d65,
        "linear_srgb_unclipped": linear,
        "srgb_unclipped": srgb,
        "srgb_display_clipped": torch.clamp(srgb, 0.0, 1.0),
    }


def integrate_camera_rgb(radiance: torch.Tensor, weights_nm: torch.Tensor, camera_ssf: torch.Tensor) -> torch.Tensor:
    """Integrate radiance to raw camera RGB."""

    return radiance @ (weights_nm.unsqueeze(-1) * camera_ssf)


def render_camera(
    reflectance: torch.Tensor,
    weights_nm: torch.Tensor,
    cie_xyz: torch.Tensor,
    illuminant_y1: torch.Tensor,
    camera_ssf: torch.Tensor,
    white_response: torch.Tensor,
    matrix_3x3: torch.Tensor,
    source_white: torch.Tensor,
    d65_white: torch.Tensor,
    shading: torch.Tensor,
    specular: torch.Tensor,
    exposure: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Torch camera path."""

    rad = spectral_radiance(reflectance, illuminant_y1, shading, specular, exposure)
    raw = integrate_camera_rgb(rad, weights_nm, camera_ssf)
    wb = raw / white_response
    xyz_source = wb @ matrix_3x3
    xyz_d65 = bradford_adapt(xyz_source, source_white, d65_white)
    linear = xyz_to_linear_srgb(xyz_d65)
    srgb = srgb_encode(linear)
    return {
        "camera_rgb_raw": raw,
        "camera_rgb_wb": wb,
        "xyz_source": xyz_source,
        "xyz_d65": xyz_d65,
        "linear_srgb_unclipped": linear,
        "srgb_unclipped": srgb,
        "srgb_display_clipped": torch.clamp(srgb, 0.0, 1.0),
    }
