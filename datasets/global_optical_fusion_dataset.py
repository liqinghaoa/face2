"""Composition-based image plus optical-feature dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets.nyha_3class_face_dataset import NYHA3ClassFaceDataset
from utils.optical_feature_preprocessor import (
    FeatureScaler,
    VARIANT_AUX_DIM,
    VARIANT_FEATURE_COLUMNS,
    build_aux_features,
    validate_feature_values,
    validate_variant,
)


class GlobalOpticalFusionDataset(Dataset):
    """Preserve split order while joining an explicitly selected feature table by ID."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        variant: str,
        fold: int,
        split_role: str,
        transform: Any = None,
        image_root: str | Path | None = None,
        image_filename_template: str = "{ID}.png",
        feature_frame: pd.DataFrame | None = None,
        scaler: FeatureScaler | None = None,
    ) -> None:
        self.variant = validate_variant(variant)
        self.fold = int(fold)
        self.split_role = str(split_role).lower()
        if self.split_role not in {"train", "val"}:
            raise ValueError("split_role must be train or val")
        self.base_dataset = NYHA3ClassFaceDataset(
            csv_path=csv_path,
            transform=transform,
            image_root=image_root,
            image_filename_template=image_filename_template,
        )
        self.frame = self.base_dataset.frame
        split_ids = self.frame["ID"].astype(str).tolist()
        if len(split_ids) != len(set(split_ids)):
            raise ValueError("Split CSV contains duplicate IDs")

        if self.variant == "global_only":
            if feature_frame is not None or scaler is not None:
                raise ValueError("global_only must not receive optical data or a scaler")
            self.aux_features = np.empty((len(self.frame), 0), dtype=np.float32)
            self.feature_frame = None
        else:
            if feature_frame is None:
                raise ValueError(f"{self.variant} requires a feature_frame")
            if "ID" not in feature_frame.columns:
                raise ValueError("feature_frame is missing ID")
            features = feature_frame.copy()
            required_columns = {"ID", "forehead_available", *VARIANT_FEATURE_COLUMNS[self.variant]}
            missing_columns = sorted(required_columns.difference(features.columns))
            if missing_columns:
                raise ValueError(
                    f"feature_frame is missing the positive-allowlist columns {missing_columns}"
                )
            features["ID"] = features["ID"].astype(str)
            if features["ID"].duplicated().any():
                raise ValueError("feature_frame contains duplicate IDs")
            feature_ids = features["ID"].tolist()
            missing = sorted(set(split_ids) - set(feature_ids))
            extra = sorted(set(feature_ids) - set(split_ids))
            if missing or extra:
                raise ValueError(
                    f"feature_frame IDs must exactly match split IDs; missing={missing}, extra={extra}"
                )
            features = features.set_index("ID", drop=False).loc[split_ids].reset_index(drop=True)
            if self.variant in {"global_stage2a", "global_stage2b"}:
                required = {"fold", "split_role"}
                if not required.issubset(features.columns):
                    raise ValueError(f"Stage 2 features require columns {sorted(required)}")
                if not (pd.to_numeric(features["fold"], errors="coerce") == self.fold).all():
                    raise ValueError("Stage 2 fold does not match the classification fold")
                if not (features["split_role"].astype(str).str.lower() == self.split_role).all():
                    raise ValueError("Stage 2 split_role does not match the requested role")
            validate_feature_values(features, self.variant, "feature_frame")
            self.feature_frame = features
            self.aux_features = build_aux_features(features, self.variant, scaler)
        if self.aux_features.shape != (len(self.frame), VARIANT_AUX_DIM[self.variant]):
            raise ValueError("Auxiliary array shape does not match the variant")
        if self.aux_features.dtype != np.float32 or not np.isfinite(self.aux_features).all():
            raise ValueError("Auxiliary features must be finite float32")

    @property
    def labels(self) -> list[int]:
        return self.base_dataset.labels

    @property
    def ids(self) -> list[str]:
        return self.frame["ID"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base_dataset[index]
        return {
            "image": item["image"],
            "aux_features": torch.from_numpy(self.aux_features[index].copy()),
            "label": item["label"],
            "ID": item["ID"],
            "patient_group_id": item["patient_group_id"],
            "NYHA": item["NYHA"],
            "SEX": item["SEX"],
            "sex_name": item["sex_name"],
            "label_3class_name": item["label_3class_name"],
            # The train CSV contains samples whose held-out assignment is not
            # the current classifier fold.  The selected file path defines the
            # active outer fold, so expose that run-level fold consistently.
            "fold": self.fold,
            "split_role": self.split_role,
        }
