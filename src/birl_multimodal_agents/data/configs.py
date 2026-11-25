from dataclasses import dataclass
from pathlib import Path 
from typing import Literal

import yaml 

CropType = Literal["full", "target_union"]
Interpolation = Literal["nearest", "bilinear", "bicubic", "lanczos"]

@dataclass(frozen=True)
class LevelConfig:
    crop: CropType
    context_scale: float
    max_side: int | None 

    def __post_int__(self) -> None:
        if self.context_scale < 1.0:
            raise ValueError("context_scale must be at least 1.0")

        if self.max_side is not None and self.max_side <= 0:
            raise ValueError("max_side must be positive")

