from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import AbstractSet, Any, Mapping, Optional, Union, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def count_images_size(
    images_dir: Union[str, Path], extensions: Optional[AbstractSet[str]] = None
) -> Counter:
    images_dir = Path(images_dir)
    size_counter = Counter()

    if extensions is None:
        extensions = {".jpeg", ".jpg", ".png"}

    for file in images_dir.glob("[!.]*"):
        if file.suffix in extensions:
            image = Image.open(file)
            size_counter[image.size] += 1

    return size_counter


def plot_directories_representation(
    dirs: Mapping[str, Union[str, Path]],
    n_images: int = 10,
    extensions: Optional[AbstractSet[str]] = None,
    fig_params: Optional[Mapping[str, Any]] = None,
    text_params: Optional[Mapping[str, Any]] = None,
) -> Figure:
    if extensions is None:
        extensions = {".jpeg", ".jpg", ".png"}
    if fig_params is None:
        fig_params = {"figsize": (12, 12)}
    if text_params is None:
        text_params = {"fontsize": 10}

    size_params = {"nrows": len(dirs), "ncols": n_images + 1}
    fig_params = {**fig_params, **size_params}

    fig, axes = plt.subplots(**fig_params)

    for row_idx, (name, path) in enumerate(dirs.items()):
        axes[row_idx, 0].text(0, 0.5, name, **text_params)
        axes[row_idx, 0].axis("off")

        column_idx = 1
        for file in Path(path).glob("[!.]*"):
            if file.is_file() and file.suffix in extensions:
                try:
                    with Image.open(file) as image:
                        image = np.asarray(image)
                except UnidentifiedImageError:
                    continue
            plt.sca(axes[row_idx, column_idx])
            axes[row_idx, column_idx].imshow(image)
            axes[row_idx, column_idx].axis("off")

            if column_idx < n_images:
                column_idx += 1
            else:
                break

    return fig
