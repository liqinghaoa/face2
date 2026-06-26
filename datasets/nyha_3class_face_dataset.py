"""Dataset and transforms for the NYHA three-class face experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


REQUIRED_METADATA_COLUMNS = {
    "ID",
    "patient_group_id",
    "NYHA",
    "label_3class",
    "label_3class_name",
    "SEX",
    "sex_name",
    "fold",
}
IMAGE_PATH_COLUMN = "image_path"


def build_transforms(
    split: str,
    image_size: int = 224,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    horizontal_flip: bool = True,
) -> transforms.Compose:
    """Build the deterministic validation or flip-only training transform."""
    split = split.lower()
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got: {split!r}")

    operations: list[Any] = [transforms.Resize((image_size, image_size))]
    if split == "train" and horizontal_flip:
        operations.append(transforms.RandomHorizontalFlip(p=0.5))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )
    return transforms.Compose(operations)


class NYHA3ClassFaceDataset(Dataset):
    """Load fixed-fold face images and their NYHA three-class labels."""

    def __init__(
        self,
        csv_path: str | Path,
        transform: Any = None,
        image_root: str | Path | None = None,
        image_filename_template: str = "{ID}.png",
    ) -> None:
        self.csv_path = Path(csv_path).expanduser().resolve()
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Dataset CSV does not exist: {self.csv_path}")
        self.image_root = (
            Path(image_root).expanduser().resolve()
            if image_root not in {None, ""}
            else None
        )
        if self.image_root is not None and not self.image_root.is_dir():
            raise FileNotFoundError(f"Image root does not exist: {self.image_root}")
        self.image_filename_template = image_filename_template

        self.frame = pd.read_csv(
            self.csv_path,
            dtype={"ID": "string", "patient_group_id": "string"},
            encoding="utf-8-sig",
        )
        required_columns = set(REQUIRED_METADATA_COLUMNS)
        if self.image_root is None:
            required_columns.add(IMAGE_PATH_COLUMN)
        missing = sorted(required_columns.difference(self.frame.columns))
        if missing:
            raise ValueError(
                f"Dataset CSV is missing required columns {missing}: {self.csv_path}"
            )
        if self.frame.empty:
            raise ValueError(f"Dataset CSV contains no samples: {self.csv_path}")

        labels = pd.to_numeric(self.frame["label_3class"], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1, 2]).all():
            bad_rows = self.frame.index[labels.isna() | ~labels.isin([0, 1, 2])].tolist()
            raise ValueError(
                f"Invalid label_3class values at zero-based rows {bad_rows[:10]} "
                f"in {self.csv_path}"
            )
        self.frame["label_3class"] = labels.astype("int64")
        self.transform = transform

    def resolve_image_path(self, identifier: str) -> Path:
        """Resolve an image path from ID using image_root when configured."""
        if self.image_root is None:
            matches = self.frame.loc[self.frame["ID"].astype(str) == identifier]
            if matches.empty:
                raise KeyError(f"Unknown ID in dataset: {identifier}")
            return Path(str(matches.iloc[0][IMAGE_PATH_COLUMN])).expanduser()

        filename = self.image_filename_template.format(ID=identifier)
        return self.image_root / filename

    @property
    def labels(self) -> list[int]:
        return self.frame["label_3class"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        identifier = str(row["ID"])
        if self.image_root is None:
            image_path = Path(str(row[IMAGE_PATH_COLUMN])).expanduser()
        else:
            image_path = self.image_root / self.image_filename_template.format(
                ID=identifier
            )
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_tensor = self.transform(image) if self.transform else image.copy()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image for ID={identifier!r}, image_path={image_path}"
            ) from exc

        return {
            "image": image_tensor,
            "label": int(row["label_3class"]),
            "ID": identifier,
            "patient_group_id": str(row["patient_group_id"]),
            "image_path": str(image_path),
            "NYHA": int(row["NYHA"]),
            "SEX": int(row["SEX"]),
            "sex_name": str(row["sex_name"]),
            "label_3class_name": str(row["label_3class_name"]),
            "fold": int(row["fold"]),
        }
