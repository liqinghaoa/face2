"""Global semantic/mask construction and pure mask-set definitions."""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import GlobalMaskConfig, MaskArtifacts, ParsingArtifacts


def binary_mask(value: np.ndarray) -> np.ndarray:
    """Return the required uint8 0/255 single-channel representation."""
    return (np.asarray(value) > 0).astype(np.uint8) * 255


def build_base_masks(label_map: np.ndarray, source_valid: np.ndarray, final_face_mask: np.ndarray, feather_alpha: np.ndarray) -> MaskArtifacts:
    """Define source-valid, face-valid, and semantic-only strict-skin sets."""
    source = binary_mask(source_valid); final = binary_mask(final_face_mask)
    face_valid = binary_mask((source > 0) & (final > 0))
    # CelebrityMask-HQ class 1 is the only strict-skin class; no envelope fill.
    skin = binary_mask((label_map == 1) & (face_valid > 0))
    return MaskArtifacts(final, face_valid, skin, np.asarray(feather_alpha, dtype=np.float32))


def build_parsing_artifacts(aligned_rgb: np.ndarray, model: Any, device: Any, config: GlobalMaskConfig) -> tuple[ParsingArtifacts, MaskArtifacts]:
    """Run one BiSeNet inference then reproduce the specified hybrid final mask."""
    import torch
    from preprocessing import build_global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict as hybrid
    from preprocessing import build_global_face_parsing_regularmask_blackbg_224_png_strict as parsing
    with torch.inference_mode():
        labels = parsing.run_face_parsing(aligned_rgb, model, device)
    selected = parsing.build_selected_semantic_mask(labels)
    regularized = hybrid.build_semantic_regularized_mask(selected)
    envelope, _ = parsing.build_regularized_face_envelope_mask(regularized, labels, config.forehead_expand_ratio, config.side_expand_ratio, config.chin_expand_ratio)
    _, hair_ratio = hybrid._ratio_inside_mask(labels, envelope, (parsing.CLASS_HAIR,))
    jaggedness = hybrid.compute_forehead_top_jaggedness(regularized, config.forehead_band_ratio)
    final, _, _ = hybrid.build_final_mask(regularized, envelope, config.final_mask_mode, config.forehead_band_ratio, hair_ratio, config.hair_repair_threshold, jaggedness, config.jaggedness_threshold, config.enable_jaggedness_trigger)
    alpha = parsing.feather_mask(final, config.feather_kernel)
    return ParsingArtifacts(labels.astype(np.uint8), binary_mask(selected), binary_mask(regularized), binary_mask(envelope)), build_base_masks(labels, np.full(labels.shape, 255, np.uint8), final, alpha)
