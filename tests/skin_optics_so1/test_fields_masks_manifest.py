from __future__ import annotations

import numpy as np

from skin_optics_so1.camera_light_split import FAILED_PAIRS, build_camera_light_split
from skin_optics_so1.random_fields import generate_mh_field, generate_p_field, generate_s_field, stable_seed
from skin_optics_so1.synthetic_config import load_config
from skin_optics_so1.synthetic_manifest import build_manifest
from skin_optics_so1.synthetic_masks import generate_valid_mask


def test_stable_seed_is_reproducible_and_component_separated() -> None:
    a = stable_seed(20260801, "train", 1, 0, "m_field")
    b = stable_seed(20260801, "train", 1, 0, "m_field")
    c = stable_seed(20260801, "train", 1, 0, "h_field")
    assert a == b
    assert a != c


def test_fields_and_masks_ranges() -> None:
    size = 256
    m, ms = generate_mh_field(0.5, 1, size)
    s, ss = generate_s_field(2, size)
    p, ps = generate_p_field(3, size)
    mask, frac, _ = generate_valid_mask(4, size, "mixed", 0.20)
    assert m.shape == (size, size)
    assert s.shape == (size, size)
    assert p.shape == (size, size)
    assert mask.shape == (size, size)
    assert 0.0 <= float(m.min()) <= float(m.max()) <= 1.0
    assert 0.25 <= float(s.min()) <= float(s.max()) <= 2.0
    assert 0.0 <= float(p.min()) <= float(p.max()) <= 0.10
    assert frac >= 0.20
    assert set(np.unique(mask)).issubset({0, 1})
    assert ms.std > 0
    assert ss.mean >= 0.25
    assert ps.nonzero_fraction >= 0.0


def test_camera_light_split_from_so0_audit() -> None:
    split = build_camera_light_split(20260801)
    assert len(split.seen_cameras) == 25
    assert len(split.unseen_cameras) == 3
    assert "Canon 5DMarkII" in split.seen_cameras
    assert "Point Grey Grasshopper 50S5C" not in split.unseen_cameras
    assert len(split.qualified_pairs) == 109
    for p in split.qualified_pairs:
        assert (p.camera_name, p.light_name) not in FAILED_PAIRS
    for cam in split.unseen_cameras:
        lights = {p.light_name for p in split.qualified_pairs if p.camera_name == cam}
        assert {"D65", "A", "FL2", "FL11"}.issubset(lights)


def test_manifest_smoke_contracts() -> None:
    cfg = load_config("configs/so1_synthetic_smoke_v1.yaml")
    split = build_camera_light_split(cfg.global_seed)
    rows = build_manifest(cfg, split)
    assert len(rows) == 64
    assert len({r.sample_id for r in rows}) == 64
    assert [r.row_index for r in rows] == list(range(64))
    train = [r for r in rows if r.split == "train"]
    assert len(train) == 24
    for i in range(0, len(train), 2):
        a, b = train[i], train[i + 1]
        assert a.base_latent_id == b.base_latent_id
        assert a.m_seed == b.m_seed
        assert a.h_seed == b.h_seed
        assert a.mask_seed == b.mask_seed
        assert a.camera_light_pair != b.camera_light_pair
    base_by_split = {}
    for r in rows:
        base_by_split.setdefault(r.split, set()).add(r.base_latent_id)
    splits = sorted(base_by_split)
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            assert not (base_by_split[s1] & base_by_split[s2])

