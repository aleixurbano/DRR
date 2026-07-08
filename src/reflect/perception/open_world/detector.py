"""YOLOe text-prompt detection - open-vocabulary detector that returns labeled
bboxes and initial masks. SAM3 refines the masks downstream.

The default vocabulary covers AI2-THOR sim kitchens + office/lab/warehouse
scenes typical of Robo2VLM-1 (Open-X-Embodiment). Caller-supplied vocab is
appended.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

ImageInput = Union[str, Path, np.ndarray, Image.Image]

# Path to the user's locally cached YOLOe weights. Override via env var.
_DEFAULT_YOLOE = os.environ.get(
    "REFLECT_YOLOE_WEIGHTS",
    "yoloe-26x-seg.pt",  # text-prompt variant (auto-downloads if absent)
)

# Background classes to keep but flag (per ConceptGraphs convention).
BG_CLASSES = {"wall", "floor", "ceiling"}

# Curated vocabulary covering REFLECT sim tasks + Robo2VLM scenes.
_DEFAULT_VOCAB: List[str] = [
    # AI2-THOR kitchen objects
    "pot", "pan", "frying pan", "kettle", "teapot",
    "faucet", "sink", "stove burner", "stove knob", "oven", "microwave",
    "fridge", "refrigerator", "freezer",
    "cabinet", "drawer", "shelf", "shelving unit", "countertop", "counter",
    "toaster", "coffee machine", "blender", "dishwasher",
    # Food
    "bread", "bread slice", "egg", "apple", "tomato", "tomato slice",
    "lettuce", "lettuce slice", "potato", "potato slice", "banana",
    "orange", "carrot", "onion",
    # Utensils / containers
    "knife", "butter knife", "fork", "spoon", "spatula", "ladle", "whisk",
    "cup", "mug", "bowl", "plate", "glass", "bottle", "glass bottle",
    "spray bottle", "salt shaker", "pepper shaker", "jar",
    "cutting board", "garbage can", "trash bin",
    "soap", "soap bar", "soap bottle", "dish sponge", "scrub brush",
    "paper towel", "paper towel roll", "hand towel", "towel",
    # Living-room / office (relevant to Robo2VLM)
    "table", "dining table", "side table", "coffee table", "desk",
    "chair", "stool", "bed", "sofa", "couch", "lamp", "desk lamp",
    "house plant", "vase", "alarm clock", "remote control",
    "television", "monitor", "laptop", "keyboard", "computer mouse",
    "phone", "book",
    # Robot / lab
    "box", "container", "bag", "tool", "robot arm", "gripper", "tape",
    # Background
    "wall", "floor", "ceiling", "window", "door",
    # People (occasional in Robo2VLM)
    "person", "hand", "arm",
]


@dataclass
class Detection:
    bbox_xyxy: np.ndarray   # (4,) float32: x1, y1, x2, y2
    mask: np.ndarray        # (H, W) bool - YOLOe's initial mask, refined by SAM3 later
    class_name: str
    confidence: float
    is_background: bool


@lru_cache(maxsize=4)
def _load_detector(weights: str, vocab_key: tuple):
    from ultralytics import YOLOE
    model = YOLOE(weights)
    classes = list(vocab_key)
    model.set_classes(classes, model.get_text_pe(classes))
    return model, classes


def _to_rgb_array(image: ImageInput) -> np.ndarray:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        return np.array(image)
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"image must be HxWx3 RGB, got {arr.shape}")
    return arr


def detect(
    image: ImageInput,
    *,
    vocabulary: Optional[Sequence[str]] = None,
    conf: float = 0.10,
    weights: Optional[str] = None,
    max_detections: int = 30,
) -> List[Detection]:
    """Run YOLOe in text-prompt mode and return a list of labeled detections."""
    rgb = _to_rgb_array(image)
    full_vocab = tuple(_DEFAULT_VOCAB) if vocabulary is None else tuple(
        list(_DEFAULT_VOCAB) + [v for v in vocabulary if v not in _DEFAULT_VOCAB]
    )
    model, classes = _load_detector(weights or _DEFAULT_YOLOE, full_vocab)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    res = model.predict(rgb, conf=conf, verbose=False, device=device)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []

    H, W = rgb.shape[:2]
    boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy().astype(np.float32)

    # YOLOe seg masks are at model resolution (640×640) - resize to image size.
    masks_raw = res.masks.data.cpu().numpy().astype(bool) if res.masks is not None else None

    detections: List[Detection] = []
    for i in range(len(boxes)):
        name = classes[cls_ids[i]]
        if masks_raw is not None and masks_raw[i].shape != (H, W):
            import cv2
            m = cv2.resize(masks_raw[i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            m = masks_raw[i] if masks_raw is not None else _bbox_to_mask(boxes[i], H, W)
        detections.append(Detection(
            bbox_xyxy=boxes[i],
            mask=m,
            class_name=name,
            confidence=float(confs[i]),
            is_background=name in BG_CLASSES,
        ))

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections[:max_detections]


def _bbox_to_mask(bbox: np.ndarray, H: int, W: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    m = np.zeros((H, W), dtype=bool)
    m[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
    return m
