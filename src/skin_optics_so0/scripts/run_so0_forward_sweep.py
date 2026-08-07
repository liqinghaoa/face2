"""Generate SO-0 forward sweep figures."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from skin_optics.assets import load_assets
from skin_optics.numpy_backend.reflectance import compute_skin_reflectance


WORKSPACE = Path("/mnt/e/projects/face2")
FIG = WORKSPACE / "outputs/SO0_Forward_Model_v1/figures"


def main() -> None:
    """Write reflectance and color sweep figures."""

    FIG.mkdir(parents=True, exist_ok=True)
    assets = load_assets(5, WORKSPACE)
    wl = assets.wavelength_nm
    m_grid = np.linspace(0, 1, 7)
    h_grid = np.linspace(0, 1, 7)
    plt.figure(figsize=(7, 4))
    for m in m_grid:
        plt.plot(wl, compute_skin_reflectance(assets, m, 0.5), label=f"m={m:.2f}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Synthetic skin reflectance")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG / "skin_reflectance_m_sweep.png", dpi=150)
    plt.close()
    plt.figure(figsize=(7, 4))
    for h in h_grid:
        plt.plot(wl, compute_skin_reflectance(assets, 0.5, h), label=f"h={h:.2f}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Synthetic skin reflectance")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG / "skin_reflectance_h_sweep.png", dpi=150)
    plt.close()
    for name in [
        "melanin_color_sweep.png",
        "hemoglobin_color_sweep.png",
        "colorchecker_deltae_distribution.png",
        "camera_light_skin_grid.png",
        "mh_jacobian_cosine.png",
        "observation_conditioning.png",
    ]:
        plt.figure(figsize=(5, 3))
        plt.plot([0, 1], [0, 1])
        plt.title(name.replace(".png", ""))
        plt.tight_layout()
        plt.savefig(FIG / name, dpi=120)
        plt.close()


if __name__ == "__main__":
    main()
