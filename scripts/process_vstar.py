from dataclasses import dataclass
from pathlib import Path 
from typing import Literal
import PIL

bbox_xy_wh = tuple[float, float, float, float]

@dataclass(frozen=True)
class VStarExample:
    sample_id : str 
    category: Literal["direct_attributes", "relative_position"]
    image_path: Path 
    target_objects: tuple[str, ...]
    bboxes_xy_wh: tuple[bbox_xy_wh, ...]
    question: str 
    options: tuple[str, ...]
    answer_index: int 


@dataclass(frozen=True)
class EvidenceObservation:
    sample_id: str 
    level: int 
    image: "PIL.Image.Image"
    crop_xy_xy: tuple[int, int, int, int]
    original_size: tuple[int, int]
    transform_version: str