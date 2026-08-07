"""High-level NumPy forward model."""

from __future__ import annotations

import numpy as np

from skin_optics.assets import SpectralAssets, load_assets
from skin_optics.numpy_backend.colorchecker import solve_camera_to_xyz_matrix
from skin_optics.numpy_backend.image_formation import render_camera, render_cie_reference
from skin_optics.numpy_backend.reflectance import compute_skin_reflectance
from skin_optics.types import CameraRenderResult, RenderResult


class SO0NumpyForwardModel:
    """NumPy float64 reference implementation."""

    def __init__(self, assets: SpectralAssets | None = None) -> None:
        self.assets = assets or load_assets(5)

    def compute_skin_reflectance(self, m: np.ndarray | float, h: np.ndarray | float) -> np.ndarray:
        """Compute synthetic skin reflectance."""

        return compute_skin_reflectance(self.assets, m, h)

    def render_cie_reference(
        self,
        m: np.ndarray | float,
        h: np.ndarray | float,
        illuminant_name: str = "D65",
        shading: np.ndarray | float = 1.0,
        specular: np.ndarray | float = 0.0,
        exposure: np.ndarray | float = 1.0,
    ) -> RenderResult:
        """Render through the CIE reference path."""

        return render_cie_reference(self.assets, self.compute_skin_reflectance(m, h), illuminant_name, shading, specular, exposure)

    def render_camera(
        self,
        m: np.ndarray | float,
        h: np.ndarray | float,
        camera_name: str = "Canon 5DMarkII",
        illuminant_name: str = "D65",
        matrix_3x3: np.ndarray | None = None,
        shading: np.ndarray | float = 1.0,
        specular: np.ndarray | float = 0.0,
        exposure: np.ndarray | float = 1.0,
    ) -> CameraRenderResult:
        """Render through a camera and fixed ColorChecker matrix."""

        matrix = matrix_3x3 if matrix_3x3 is not None else solve_camera_to_xyz_matrix(self.assets, camera_name, illuminant_name)[0]
        return render_camera(
            self.assets,
            self.compute_skin_reflectance(m, h),
            matrix,
            camera_name,
            illuminant_name,
            shading,
            specular,
            exposure,
        )

    def forward(self, m: np.ndarray | float, h: np.ndarray | float) -> np.ndarray:
        """Return D65 CIE reference linear sRGB without display clipping."""

        return self.render_cie_reference(m, h).linear_srgb_unclipped
