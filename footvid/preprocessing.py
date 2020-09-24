from __future__ import annotations

from typing import Optional, Sequence, TYPE_CHECKING
import numpy as np
from torchvision import transforms

from footvid.utils.constants import MODELING_SIZE, RGB_CHANNEL_STATS

if TYPE_CHECKING:
    import torch


TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(
            size=tuple(
                (np.asarray(MODELING_SIZE) * (1.0 / (0.85 * 0.8))).astype(int)
            )
        ),
        transforms.CenterCrop(
            size=tuple((np.asarray(MODELING_SIZE) * (1.0 / 0.85)).astype(int))
        ),
        transforms.RandomCrop(size=MODELING_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(**RGB_CHANNEL_STATS),
    ]
)

TEST_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(
            size=tuple((np.asarray(MODELING_SIZE) * (1.0 / 0.8)).astype(int))
        ),
        transforms.CenterCrop(size=MODELING_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(**RGB_CHANNEL_STATS),
    ]
)


def denormalize(
    x: torch.Tensor,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> np.ndarray:
    if mean is None:
        mean = RGB_CHANNEL_STATS["mean"]
    if std is None:
        std = RGB_CHANNEL_STATS["std"]

    # Adjust to PIL Image format where RGB channel is the last dimension.
    x = x.numpy().transpose(1, 2, 0)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x * std + mean).clip(0, 1)

    return x
