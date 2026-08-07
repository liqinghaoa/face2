# SO-0 Skin Optics Forward Model

This is an independent skin-optics-constrained forward simulator. It is not a
strict reproduction of Jung et al. 2023, not a disease classifier, and not a
route for recovering real physiological concentrations from JPEG images.

The package uses frozen SO0 spectral assets, a finite-thickness Kubelka-Munk
dermis reflection model, a melanin-sensitive epidermal absorption control, a
hemoglobin-sensitive dermal absorption control, CIE reference rendering, fixed
linear Virtual ColorChecker camera matrices, and NumPy/PyTorch backends.
