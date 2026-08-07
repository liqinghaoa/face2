"""Deterministic camera-light pair selection for clipping retry repair."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable

from skin_optics_so1.camera_light_split import FAILED_PAIRS, CameraLightPair


class RetryPairSelectionError(RuntimeError):
    """Raised when deterministic retry has no valid camera-light candidate."""


def pair_key(pair: CameraLightPair | str | tuple[str, str]) -> str:
    if isinstance(pair, CameraLightPair):
        return pair.key
    if isinstance(pair, tuple):
        return f"{pair[0]} / {pair[1]}"
    return str(pair)


def stable_pair_order_token(
    *,
    split: str,
    base_latent_id: int,
    acquisition_variant_id: int,
    retry_index: int,
    acquisition_seed: int,
    pair: CameraLightPair,
) -> str:
    payload = (
        f"{split}|{int(base_latent_id)}|{int(acquisition_variant_id)}|"
        f"{int(retry_index)}|{int(acquisition_seed)}|{pair.key}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_retry_pair(
    *,
    split: str,
    base_latent_id: int,
    acquisition_variant_id: int,
    retry_index: int,
    allowed_pairs: Iterable[CameraLightPair],
    forbidden_pairs: Iterable[CameraLightPair | str | tuple[str, str]],
    acquisition_seed: int,
) -> CameraLightPair:
    """Select one deterministic retry pair from split-allowed qualified pairs.

    The function intentionally uses SHA256 ordering rather than Python's
    process-randomized ``hash()``.  It never falls back to a forbidden pair.
    """

    forbidden_keys = {pair_key(p) for p in forbidden_pairs}
    candidates = []
    for pair in allowed_pairs:
        if (pair.camera_name, pair.light_name) in FAILED_PAIRS:
            continue
        if pair.key in forbidden_keys:
            continue
        candidates.append(pair)
    if not candidates:
        raise RetryPairSelectionError(
            "No valid retry camera-light pair remains after applying forbidden pairs "
            f"for split={split} base_latent_id={base_latent_id} "
            f"variant={acquisition_variant_id} retry={retry_index}"
        )
    return sorted(
        candidates,
        key=lambda p: stable_pair_order_token(
            split=split,
            base_latent_id=base_latent_id,
            acquisition_variant_id=acquisition_variant_id,
            retry_index=retry_index,
            acquisition_seed=acquisition_seed,
            pair=p,
        ),
    )[0]


def train_final_pair_collisions(rows) -> list[dict]:
    by_base: dict[int, list] = defaultdict(list)
    for row in rows:
        if row.split == "train":
            by_base[int(row.base_latent_id)].append(row)
    collisions: list[dict] = []
    for base_latent_id, group in sorted(by_base.items()):
        if len(group) != 2:
            collisions.append({"base_latent_id": base_latent_id, "reason": "variant_count_not_2", "count": len(group)})
            continue
        a, b = sorted(group, key=lambda r: int(r.acquisition_variant_id))
        a_pair = getattr(a, "final_camera_light_pair", "") or a.camera_light_pair
        b_pair = getattr(b, "final_camera_light_pair", "") or b.camera_light_pair
        if a_pair == b_pair:
            collisions.append(
                {
                    "base_latent_id": base_latent_id,
                    "variant_0_sample_id": a.sample_id,
                    "variant_1_sample_id": b.sample_id,
                    "camera_light_pair": a_pair,
                }
            )
    return collisions


def assert_train_final_pair_invariants(rows) -> None:
    collisions = train_final_pair_collisions(rows)
    if collisions:
        raise ValueError(f"Train paired variants must use different final camera-light pairs: {collisions[:8]}")
