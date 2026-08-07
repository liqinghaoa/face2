"""Least-squares directional Lambertian-to-SH projection and validation."""
from __future__ import annotations
import numpy as np

def sh_basis(normals: np.ndarray) -> np.ndarray:
    """Return a canonical real second-order SH basis for pre-DECA synthetic checks.

    It must be compared with the supplied DECA renderer before it is used for DECA
    inference; B0 remains blocked until that source-level audit can occur.
    """
    x,y,z=np.moveaxis(np.asarray(normals,dtype=np.float64),-1,0)
    return np.stack((np.ones_like(x),x,y,z,x*y,x*z,y*z,3*z*z-1,x*x-y*y),axis=-1)

class DirectionalToSHProjector:
    """Fit nine real SH coefficients to a fixed grayscale Lambertian target."""
    def __init__(self, sample_count: int = 4096) -> None: self.sample_count=sample_count
    def fit(self, direction: np.ndarray, ambient: float, diffuse: float) -> tuple[np.ndarray,float]:
        direction=np.asarray(direction,float); direction/=np.linalg.norm(direction)
        index=np.arange(self.sample_count,dtype=float); z=1-2*(index+.5)/self.sample_count; phi=np.pi*(1+5**.5)*index; radius=np.sqrt(np.clip(1-z*z,0,None)); normals=np.stack((radius*np.cos(phi),radius*np.sin(phi),z),axis=1)
        target=ambient+diffuse*np.maximum(normals@direction,0); coeff,*_=np.linalg.lstsq(sh_basis(normals),target,rcond=None); rmse=float(np.sqrt(np.mean((sh_basis(normals)@coeff-target)**2))); return np.repeat(coeff[:,None],3,axis=1).astype(np.float32),rmse

    def fit_renderer(self, direction: np.ndarray, ambient: float, diffuse: float, constant_factor: np.ndarray) -> tuple[np.ndarray, float]:
        """Convert canonical coefficients to DECA's scaled and reordered basis."""
        canonical, rmse = self.fit(direction, ambient, diffuse)
        # canonical: [..., 7]=3z²-1, [..., 8]=x²-y²; DECA uses the reverse order.
        reordered = canonical[[0, 1, 2, 3, 4, 5, 6, 8, 7]]
        factor = np.asarray(constant_factor, dtype=np.float32).reshape(9, 1)
        return (reordered / factor).astype(np.float32), rmse

def render_sh(normals: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Render raw (unclipped) RGB shading from normal map and 9x3 SH coefficients."""
    return (sh_basis(normals) @ np.asarray(coefficients)).astype(np.float32)
