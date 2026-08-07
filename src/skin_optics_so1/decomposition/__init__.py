"""SO-1 decomposition dataset, contract, and training utilities."""

from .contract import (
    ARRAY_SPECS,
    EXPECTED_GENERATOR_CONFIG_HASH,
    EXPECTED_SPLIT_COUNTS,
    SPLITS,
    build_dataset_contract,
)
from .synthetic_dataset import SO1DecompositionDataset
from .unet_decomposer import SO1UNetDecomposer

__all__ = [
    "ARRAY_SPECS",
    "EXPECTED_GENERATOR_CONFIG_HASH",
    "EXPECTED_SPLIT_COUNTS",
    "SPLITS",
    "SO1DecompositionDataset",
    "SO1UNetDecomposer",
    "build_dataset_contract",
]
