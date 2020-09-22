import numpy as np
from torchvision import transforms

from footvid.utils.constants import MODELING_SIZE, RGB_CHANNEL_STATS


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
