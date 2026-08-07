from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from skin_optics_so1.camera_light_split import FAILED_PAIRS, CameraLightPair, build_camera_light_split
from skin_optics_so1.retry_pairs import (
    RetryPairSelectionError,
    assert_train_final_pair_invariants,
    select_retry_pair,
    train_final_pair_collisions,
)
from skin_optics_so1.synthetic_config import load_config
from skin_optics_so1.synthetic_manifest import build_manifest
from skin_optics_so1.build_synthetic_dataset import _coerce_manifest_value


def _pick(split_name: str = "train"):
    cfg = load_config("configs/so1_synthetic_generation_v1.yaml")
    split = build_camera_light_split(cfg.global_seed)
    allowed = split.allowed_pairs(split_name)
    return cfg, split, allowed


def test_retry_candidate_excludes_sibling_final_pair() -> None:
    _cfg, _split, allowed = _pick()
    sibling = allowed[0]
    selected = select_retry_pair(
        split="train",
        base_latent_id=1,
        acquisition_variant_id=0,
        retry_index=1,
        allowed_pairs=allowed,
        forbidden_pairs={sibling.key},
        acquisition_seed=11,
    )
    assert selected.key != sibling.key


def test_variant0_retry_keeps_pair_different_from_variant1_final() -> None:
    _cfg, _split, allowed = _pick()
    variant1_final = allowed[3].key
    selected = select_retry_pair(
        split="train",
        base_latent_id=2,
        acquisition_variant_id=0,
        retry_index=1,
        allowed_pairs=allowed,
        forbidden_pairs={variant1_final},
        acquisition_seed=12,
    )
    assert selected.key != variant1_final


def test_variant1_retry_keeps_pair_different_from_variant0_final() -> None:
    _cfg, _split, allowed = _pick()
    variant0_final = allowed[4].key
    selected = select_retry_pair(
        split="train",
        base_latent_id=3,
        acquisition_variant_id=1,
        retry_index=1,
        allowed_pairs=allowed,
        forbidden_pairs={variant0_final},
        acquisition_seed=13,
    )
    assert selected.key != variant0_final


def test_both_variants_retry_with_sequential_commits_remain_different() -> None:
    _cfg, _split, allowed = _pick()
    initial0 = allowed[0].key
    initial1 = allowed[1].key
    final0 = select_retry_pair(
        split="train",
        base_latent_id=4,
        acquisition_variant_id=0,
        retry_index=1,
        allowed_pairs=allowed,
        forbidden_pairs={initial0, initial1},
        acquisition_seed=14,
    )
    final1 = select_retry_pair(
        split="train",
        base_latent_id=4,
        acquisition_variant_id=1,
        retry_index=1,
        allowed_pairs=allowed,
        forbidden_pairs={initial0, initial1, final0.key},
        acquisition_seed=15,
    )
    assert final0.key != final1.key


def test_multiple_retries_do_not_reuse_explicitly_invalid_pairs() -> None:
    _cfg, _split, allowed = _pick()
    first = select_retry_pair(
        split="train",
        base_latent_id=5,
        acquisition_variant_id=0,
        retry_index=1,
        allowed_pairs=allowed,
        forbidden_pairs={allowed[0].key},
        acquisition_seed=16,
    )
    second = select_retry_pair(
        split="train",
        base_latent_id=5,
        acquisition_variant_id=0,
        retry_index=2,
        allowed_pairs=allowed,
        forbidden_pairs={allowed[0].key, first.key},
        acquisition_seed=16,
    )
    assert second.key not in {allowed[0].key, first.key}


def test_resume_retry_selection_matches_continuous_inputs() -> None:
    _cfg, _split, allowed = _pick()
    kwargs = dict(
        split="train",
        base_latent_id=6,
        acquisition_variant_id=1,
        retry_index=3,
        allowed_pairs=allowed,
        forbidden_pairs={allowed[2].key, allowed[5].key},
        acquisition_seed=17,
    )
    assert select_retry_pair(**kwargs).key == select_retry_pair(**kwargs).key


def test_candidate_order_is_stable_across_processes() -> None:
    script = """
import json
from skin_optics_so1.camera_light_split import build_camera_light_split
from skin_optics_so1.retry_pairs import select_retry_pair
from skin_optics_so1.synthetic_config import load_config
cfg=load_config('configs/so1_synthetic_generation_v1.yaml')
split=build_camera_light_split(cfg.global_seed)
allowed=split.allowed_pairs('train')
p=select_retry_pair(split='train', base_latent_id=7, acquisition_variant_id=0, retry_index=2, allowed_pairs=allowed, forbidden_pairs={allowed[0].key}, acquisition_seed=18)
print(json.dumps({'pair': p.key}))
"""
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root / "src/skin_optics_so0/src")]
    )
    a = subprocess.check_output([sys.executable, "-c", script], text=True, cwd=root, env=env)
    b = subprocess.check_output([sys.executable, "-c", script], text=True, cwd=root, env=env)
    assert json.loads(a)["pair"] == json.loads(b)["pair"]


def test_failed_camera_light_pair_is_never_selected() -> None:
    allowed = [CameraLightPair("Point Grey Grasshopper 50S5C", "D65"), CameraLightPair("Canon 300D", "A")]
    selected = select_retry_pair(
        split="train",
        base_latent_id=8,
        acquisition_variant_id=0,
        retry_index=1,
        allowed_pairs=allowed,
        forbidden_pairs=set(),
        acquisition_seed=19,
    )
    assert (selected.camera_name, selected.light_name) not in FAILED_PAIRS
    assert selected.key == "Canon 300D / A"


def test_ood_retry_candidates_stay_inside_split_allowed_pairs() -> None:
    _cfg, split, allowed = _pick("camera_ood")
    selected = select_retry_pair(
        split="camera_ood",
        base_latent_id=9,
        acquisition_variant_id=0,
        retry_index=1,
        allowed_pairs=allowed,
        forbidden_pairs=set(),
        acquisition_seed=20,
    )
    allowed_keys = {p.key for p in split.allowed_pairs("camera_ood")}
    assert selected.key in allowed_keys
    assert selected.camera_name in split.unseen_cameras
    assert selected.light_name in split.seen_lights


def test_all_candidates_forbidden_raises_explicit_failure() -> None:
    _cfg, _split, allowed = _pick()
    with pytest.raises(RetryPairSelectionError):
        select_retry_pair(
            split="train",
            base_latent_id=10,
            acquisition_variant_id=0,
            retry_index=1,
            allowed_pairs=allowed[:3],
            forbidden_pairs={p.key for p in allowed[:3]},
            acquisition_seed=21,
        )


def test_initial_manifest_pair_protection_still_holds() -> None:
    cfg, split, _allowed = _pick()
    rows = build_manifest(cfg, split)
    assert_train_final_pair_invariants(rows)


def test_all_10000_train_base_latents_have_distinct_final_pairs_after_simulated_retry() -> None:
    cfg, split, allowed = _pick()
    rows = build_manifest(cfg, split)
    rows = deepcopy(rows)
    by_base = {}
    for row in rows:
        if row.split == "train":
            by_base.setdefault(row.base_latent_id, []).append(row)
    for base_latent_id in (5017, 5335, 5449, 9161):
        a, b = sorted(by_base[base_latent_id], key=lambda r: r.acquisition_variant_id)
        selected = select_retry_pair(
            split="train",
            base_latent_id=base_latent_id,
            acquisition_variant_id=a.acquisition_variant_id,
            retry_index=1,
            allowed_pairs=allowed,
            forbidden_pairs={a.initial_camera_light_pair, b.initial_camera_light_pair, b.final_camera_light_pair},
            acquisition_seed=a.acquisition_seed,
        )
        a.camera_name = selected.camera_name
        a.light_name = selected.light_name
        a.camera_light_pair = selected.key
        a.final_camera_name = selected.camera_name
        a.final_light_name = selected.light_name
        a.final_camera_light_pair = selected.key
    assert train_final_pair_collisions(rows) == []


def test_manifest_loader_preserves_uint64_seed_precision() -> None:
    seed = 12394173889755605833
    assert _coerce_manifest_value("m_seed", str(seed), 0) == seed
