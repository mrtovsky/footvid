from typing import Dict, List


REPOSITORY_NAME: str = "footvid"

RGB_CHANNEL_STATS: Dict[str, List[float]] = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
