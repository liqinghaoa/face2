"""High-level PyTorch forward model."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from skin_optics.assets import SpectralAssets, load_assets
from skin_optics.config import SO0Config, load_so0_config
from skin_optics.numpy_backend.colorchecker import solve_camera_to_xyz_matrix
from skin_optics.numpy_backend.image_formation import camera_white_response, d65_white_xyz, source_white_xyz
from skin_optics.torch_backend.image_formation import render_camera, render_cie_reference
from skin_optics.torch_backend.reflectance import compute_skin_reflectance


class SO0TorchForwardModel(nn.Module):
    """Differentiable PyTorch SO-0 production backend."""

    def __init__(
        self,
        assets: SpectralAssets | None = None,
        camera_name: str = "Canon 5DMarkII",
        illuminant_name: str = "D65",
        matrix_3x3: np.ndarray | None = None,
        dtype: torch.dtype = torch.float32,
        config: SO0Config | None = None,
    ) -> None:
        super().__init__()
        self.config = config or load_so0_config()
        self.assets = assets or load_assets(5)
        self.camera_name = camera_name
        self.illuminant_name = illuminant_name
        self.camera_index = self.assets.camera_index(camera_name)
        self.illuminant_index = self.assets.illuminant_index(illuminant_name)
        matrix = matrix_3x3 if matrix_3x3 is not None else solve_camera_to_xyz_matrix(self.assets, camera_name, illuminant_name)[0]
        self.register_buffer("weights_nm", torch.as_tensor(self.assets.weights_nm, dtype=dtype))
        self.register_buffer("cie_xyz", torch.as_tensor(self.assets.source["cie_xyz_2deg"], dtype=dtype))
        self.register_buffer("illuminant_y1", torch.as_tensor(self.assets.source["core_illuminants_y1"][self.illuminant_index], dtype=dtype))
        self.register_buffer("camera_ssf", torch.as_tensor(self.assets.source["camera_ssf"][self.camera_index], dtype=dtype))
        self.register_buffer("white_response", torch.as_tensor(camera_white_response(self.assets, camera_name, illuminant_name), dtype=dtype))
        self.register_buffer("matrix_3x3", torch.as_tensor(matrix, dtype=dtype))
        self.register_buffer("source_white", torch.as_tensor(source_white_xyz(self.assets, illuminant_name), dtype=dtype))
        self.register_buffer("d65_white", torch.as_tensor(d65_white_xyz(self.assets), dtype=dtype))
        self.register_buffer("mua_mel_primary", torch.as_tensor(self.assets.derived["mua_mel_primary_cm1"], dtype=dtype))
        self.register_buffer("mua_base", torch.as_tensor(self.assets.derived["mua_base_cm1"], dtype=dtype))
        self.register_buffer("mua_blood_y075", torch.as_tensor(self.assets.derived["mua_blood_y075_cm1"], dtype=dtype))
        self.register_buffer("musp_total", torch.as_tensor(self.assets.derived["musp_total_cm1"], dtype=dtype))
        mel = self.config.parameter("melanin_fraction")
        blood = self.config.parameter("blood_fraction")
        self._melanin_fraction_min = float(mel["min"])
        self._melanin_fraction_span = float(mel["max"]) - float(mel["min"])
        self._blood_fraction_min = float(blood["min"])
        self._blood_fraction_span = float(blood["max"]) - float(blood["min"])
        self._epidermis_thickness_cm = self.config.primary_value("epidermis_thickness_cm")
        self._dermis_thickness_cm = self.config.primary_value("dermis_thickness_cm")

    def compute_skin_reflectance(self, m: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute synthetic skin reflectance."""

        return compute_skin_reflectance(
            m,
            h,
            self.mua_mel_primary,
            self.mua_base,
            self.mua_blood_y075,
            self.musp_total,
            epidermis_thickness_cm=self._epidermis_thickness_cm,
            dermis_thickness_cm=self._dermis_thickness_cm,
            config=self.config,
            melanin_fraction_min=self._melanin_fraction_min,
            melanin_fraction_span=self._melanin_fraction_span,
            blood_fraction_min=self._blood_fraction_min,
            blood_fraction_span=self._blood_fraction_span,
        )

    @staticmethod
    def _validate_factor(name: str, value: torch.Tensor, min_value: float, max_value: float) -> None:
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"{name} contains non-finite values")
        if not bool(torch.all((value >= min_value) & (value <= max_value))):
            raise ValueError(f"{name} outside [{min_value}, {max_value}]")

    def render_cie_reference(
        self,
        m: torch.Tensor,
        h: torch.Tensor,
        shading: torch.Tensor | None = None,
        specular: torch.Tensor | None = None,
        exposure: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Render through the CIE reference path."""

        shading = torch.ones_like(m) if shading is None else shading
        specular = torch.zeros_like(m) if specular is None else specular
        exposure = torch.ones_like(m) if exposure is None else exposure
        sh_min, sh_max = self.config.range_minmax("shading", "sensitivity_range")
        sp_min, _ = self.config.range_minmax("specular", "primary_range")
        ex_min, ex_max = self.config.range_minmax("exposure", "sensitivity_range")
        self._validate_factor("shading", shading, sh_min, sh_max)
        self._validate_factor("specular", specular, sp_min, float(self.config.parameter("specular")["sensitivity_max"]))
        self._validate_factor("exposure", exposure, ex_min, ex_max)
        return render_cie_reference(
            self.compute_skin_reflectance(m, h),
            self.weights_nm,
            self.cie_xyz,
            self.illuminant_y1,
            self.source_white,
            self.d65_white,
            shading,
            specular,
            exposure,
        )

    def render_camera(
        self,
        m: torch.Tensor,
        h: torch.Tensor,
        shading: torch.Tensor | None = None,
        specular: torch.Tensor | None = None,
        exposure: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Render through the fixed camera path."""

        shading = torch.ones_like(m) if shading is None else shading
        specular = torch.zeros_like(m) if specular is None else specular
        exposure = torch.ones_like(m) if exposure is None else exposure
        sh_min, sh_max = self.config.range_minmax("shading", "sensitivity_range")
        sp_min, _ = self.config.range_minmax("specular", "primary_range")
        ex_min, ex_max = self.config.range_minmax("exposure", "sensitivity_range")
        self._validate_factor("shading", shading, sh_min, sh_max)
        self._validate_factor("specular", specular, sp_min, float(self.config.parameter("specular")["sensitivity_max"]))
        self._validate_factor("exposure", exposure, ex_min, ex_max)
        return render_camera(
            self.compute_skin_reflectance(m, h),
            self.weights_nm,
            self.cie_xyz,
            self.illuminant_y1,
            self.camera_ssf,
            self.white_response,
            self.matrix_3x3,
            self.source_white,
            self.d65_white,
            shading,
            specular,
            exposure,
        )

    def forward(self, m: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Return CIE reference linear sRGB for arbitrary parameter shape."""

        return self.render_cie_reference(m, h)["linear_srgb_unclipped"]

    def forward_bchw(self, m: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """BCHW wrapper returning [B,3,H,W] from [B,1,H,W] controls."""

        if m.ndim != 4 or h.ndim != 4 or m.shape[1] != 1 or h.shape[1] != 1:
            raise ValueError("Expected m and h as [B,1,H,W]")
        rgb = self.forward(m[:, 0], h[:, 0])
        return rgb.permute(0, 3, 1, 2).contiguous()
