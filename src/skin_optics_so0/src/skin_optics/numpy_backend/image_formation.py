"""CIE and camera image formation for NumPy."""

from __future__ import annotations

import numpy as np

from skin_optics.assets import SpectralAssets
from skin_optics.config import SO0Config, load_so0_config
from skin_optics.numpy_backend.color_spaces import bradford_adapt, srgb_encode, xyz_to_linear_srgb
from skin_optics.types import CameraRenderResult, RenderResult


def _check_factor(name: str, value: np.ndarray | float, min_value: float, max_value: float) -> None:
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any((arr < min_value) | (arr > max_value)):
        raise ValueError(f"{name} outside [{min_value}, {max_value}]")


def illuminant(assets: SpectralAssets, illuminant_name: str) -> np.ndarray:
    """Return a Y-normalised core illuminant."""

    return assets.source["core_illuminants_y1"][assets.illuminant_index(illuminant_name)].astype(np.float64, copy=False)


def spectral_radiance(
    assets: SpectralAssets,
    reflectance: np.ndarray,
    illuminant_name: str = "D65",
    shading: np.ndarray | float = 1.0,
    specular: np.ndarray | float = 0.0,
    exposure: np.ndarray | float = 1.0,
    config: SO0Config | None = None,
) -> np.ndarray:
    """Build relative spectral radiance without physical-path clipping."""

    cfg = config or load_so0_config()
    sh_min, sh_max = cfg.range_minmax("shading", "sensitivity_range")
    sp_min, _ = cfg.range_minmax("specular", "primary_range")
    ex_min, ex_max = cfg.range_minmax("exposure", "sensitivity_range")
    _check_factor("shading", shading, sh_min, sh_max)
    _check_factor("specular", specular, sp_min, float(cfg.parameter("specular")["sensitivity_max"]))
    _check_factor("exposure", exposure, ex_min, ex_max)
    r = np.asarray(reflectance, dtype=np.float64)
    s = np.asarray(shading, dtype=np.float64)[..., None]
    sp = np.asarray(specular, dtype=np.float64)[..., None]
    ex = np.asarray(exposure, dtype=np.float64)[..., None]
    return ex * illuminant(assets, illuminant_name) * (s * r + sp)


def integrate_xyz(assets: SpectralAssets, radiance: np.ndarray) -> np.ndarray:
    """Integrate spectral radiance to source-referred XYZ."""

    kernel = assets.weights_nm[:, None] * assets.source["cie_xyz_2deg"]
    return np.asarray(radiance, dtype=np.float64) @ kernel


def source_white_xyz(assets: SpectralAssets, illuminant_name: str = "D65") -> np.ndarray:
    """Compute source white from the same illuminant and observer, normalised to Y=1."""

    xyz = integrate_xyz(assets, illuminant(assets, illuminant_name))
    if abs(float(xyz[1])) < 1e-14:
        raise ValueError("Source white Y is zero")
    return xyz / xyz[1]


def d65_white_xyz(assets: SpectralAssets) -> np.ndarray:
    """Compute D65 white from frozen D65 SPD and CIE observer."""

    return source_white_xyz(assets, "D65")


def render_cie_reference(
    assets: SpectralAssets,
    reflectance: np.ndarray,
    illuminant_name: str = "D65",
    shading: np.ndarray | float = 1.0,
    specular: np.ndarray | float = 0.0,
    exposure: np.ndarray | float = 1.0,
    config: SO0Config | None = None,
) -> RenderResult:
    """Render via the CIE observer path without camera response."""

    rad = spectral_radiance(assets, reflectance, illuminant_name, shading, specular, exposure, config)
    xyz_source = integrate_xyz(assets, rad)
    xyz_d65 = bradford_adapt(xyz_source, source_white_xyz(assets, illuminant_name), d65_white_xyz(assets))
    linear = xyz_to_linear_srgb(xyz_d65)
    srgb = srgb_encode(linear)
    return RenderResult(
        xyz_source=xyz_source,
        xyz_d65=xyz_d65,
        linear_srgb_unclipped=linear,
        srgb_unclipped=srgb,
        srgb_display_clipped=np.clip(srgb, 0.0, 1.0),
    )


def camera_white_response(assets: SpectralAssets, camera_name: str, illuminant_name: str) -> np.ndarray:
    """Compute camera response to a perfect diffuser under one illuminant."""

    cam = assets.source["camera_ssf"][assets.camera_index(camera_name)]
    response = (assets.weights_nm[:, None] * illuminant(assets, illuminant_name)[:, None] * cam).sum(axis=0)
    if np.any(np.abs(response) < 1e-14):
        raise ValueError(f"Near-zero white response for {camera_name} / {illuminant_name}")
    return response


def integrate_camera_rgb(assets: SpectralAssets, radiance: np.ndarray, camera_name: str) -> np.ndarray:
    """Integrate spectral radiance to raw camera RGB."""

    cam = assets.source["camera_ssf"][assets.camera_index(camera_name)]
    kernel = assets.weights_nm[:, None] * cam
    return np.asarray(radiance, dtype=np.float64) @ kernel


def render_camera(
    assets: SpectralAssets,
    reflectance: np.ndarray,
    matrix_3x3: np.ndarray,
    camera_name: str = "Canon 5DMarkII",
    illuminant_name: str = "D65",
    shading: np.ndarray | float = 1.0,
    specular: np.ndarray | float = 0.0,
    exposure: np.ndarray | float = 1.0,
    config: SO0Config | None = None,
) -> CameraRenderResult:
    """Render through camera response, diagonal white balance, fixed matrix, and D65 display path."""

    rad = spectral_radiance(assets, reflectance, illuminant_name, shading, specular, exposure, config)
    raw = integrate_camera_rgb(assets, rad, camera_name)
    wb = raw / camera_white_response(assets, camera_name, illuminant_name)
    xyz_source = wb @ np.asarray(matrix_3x3, dtype=np.float64)
    xyz_d65 = bradford_adapt(xyz_source, source_white_xyz(assets, illuminant_name), d65_white_xyz(assets))
    linear = xyz_to_linear_srgb(xyz_d65)
    srgb = srgb_encode(linear)
    return CameraRenderResult(
        camera_rgb_raw=raw,
        camera_rgb_wb=wb,
        xyz_source=xyz_source,
        xyz_d65=xyz_d65,
        linear_srgb_unclipped=linear,
        srgb_unclipped=srgb,
        srgb_display_clipped=np.clip(srgb, 0.0, 1.0),
    )
