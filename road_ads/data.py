from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import resolve_repo_path


IGNORE_INDEX = 255
CITYSCAPES_TO_SYNTHIA16 = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
    8: 8, 10: 9, 11: 10, 12: 11, 13: 12, 15: 13, 17: 14,
    18: 15,
}


def parse_split_line(line: str) -> Tuple[str, Optional[str]]:
    line = line.strip()
    if not line or line.startswith("#"):
        raise ValueError("empty split line")
    fields = [item for item in re.split(r"[\s,]+", line) if item]
    if len(fields) == 1:
        return fields[0], None
    return fields[0], fields[1]


def read_split(path: str | Path) -> List[Tuple[str, Optional[str]]]:
    path = resolve_repo_path(path)
    assert path is not None
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            samples.append(parse_split_line(line))
    if not samples:
        raise ValueError(f"No samples found in split: {path}")
    return samples


def _as_hw(value: Sequence[int] | None) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"Expected [height, width], got {value}")
    return int(value[0]), int(value[1])


def _pad_to_size(image: np.ndarray, label: Optional[np.ndarray], size: Tuple[int, int]):
    height, width = size
    pad_h = max(height - image.shape[0], 0)
    pad_w = max(width - image.shape[1], 0)
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        if label is not None:
            label = cv2.copyMakeBorder(
                label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                value=IGNORE_INDEX)
    return image, label


def _random_crop(image: np.ndarray, label: Optional[np.ndarray], size: Tuple[int, int]):
    image, label = _pad_to_size(image, label, size)
    height, width = size
    y = random.randint(0, image.shape[0] - height)
    x = random.randint(0, image.shape[1] - width)
    image = image[y:y + height, x:x + width]
    if label is not None:
        label = label[y:y + height, x:x + width]
    return image, label


def build_label_map(name: Optional[str]) -> Optional[np.ndarray]:
    if not name:
        return None
    if name != "cityscapes_to_synthia16":
        raise ValueError(f"Unknown label map: {name}")
    mapping = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for source, target in CITYSCAPES_TO_SYNTHIA16.items():
        mapping[source] = target
    mapping[IGNORE_INDEX] = IGNORE_INDEX
    return mapping


class RoadSegDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str | Path,
        *,
        train: bool,
        labeled: bool,
        resize: Sequence[int] | None,
        crop_size: Sequence[int] | None,
        mean: Sequence[float],
        std: Sequence[float],
        color_order: str = "rgb",
        flip_probability: float = 0.5,
        label_map: Optional[str] = None,
    ):
        self.root = resolve_repo_path(root)
        self.split_path = resolve_repo_path(split)
        assert self.root is not None and self.split_path is not None
        self.samples = read_split(self.split_path)
        self.train = train
        self.labeled = labeled
        self.resize = _as_hw(resize)
        self.crop_size = _as_hw(crop_size)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.color_order = color_order.lower()
        self.flip_probability = float(flip_probability)
        self.label_map = build_label_map(label_map)

        if self.color_order not in {"rgb", "bgr"}:
            raise ValueError(f"Unsupported color order: {color_order}")
        if self.labeled and any(label is None for _, label in self.samples):
            raise ValueError(f"Labeled split contains entries without masks: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve(self, relative: Optional[str]) -> Optional[Path]:
        if relative is None:
            return None
        path = Path(relative)
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_rel, label_rel = self.samples[index]
        image_path = self._resolve(image_rel)
        assert image_path is not None
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        original_size = tuple(image.shape[:2])

        label = None
        if self.labeled:
            label_path = self._resolve(label_rel)
            assert label_path is not None
            label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if label is None:
                raise FileNotFoundError(f"Could not read label: {label_path}")
            if self.label_map is not None:
                label = self.label_map[label]

        if self.resize is not None:
            height, width = self.resize
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            if label is not None:
                label = cv2.resize(
                    label, (width, height), interpolation=cv2.INTER_NEAREST)

        if self.train and self.crop_size is not None:
            image, label = _random_crop(image, label, self.crop_size)

        if self.train and random.random() < self.flip_probability:
            image = np.ascontiguousarray(image[:, ::-1])
            if label is not None:
                label = np.ascontiguousarray(label[:, ::-1])

        if self.color_order == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image.astype(np.float32) - self.mean) / self.std
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        if label is None:
            label = torch.full(image.shape[-2:], IGNORE_INDEX, dtype=torch.long)
        else:
            label = torch.from_numpy(label.astype(np.int64))

        return {
            "image": image,
            "label": label,
            "id": image_rel,
            "original_size": original_size,
        }


def build_dataset(config: Dict[str, object], *, train: bool, labeled: bool):
    input_cfg = config["input"]
    return RoadSegDataset(
        config["root"],
        config["split"],
        train=train,
        labeled=labeled,
        resize=config.get("resize", input_cfg.get("resize")),
        crop_size=(
            config.get("crop_size", input_cfg.get("crop_size"))
            if train else None
        ),
        mean=input_cfg["mean"],
        std=input_cfg["std"],
        color_order=input_cfg.get("color_order", "rgb"),
        flip_probability=config.get("flip_probability", 0.5 if train else 0.0),
        label_map=config.get("label_map"),
    )


def build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=workers > 0,
    )


def infinite_loader(loader: Iterable):
    while True:
        yield from loader
