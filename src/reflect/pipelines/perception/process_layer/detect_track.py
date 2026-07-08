"""YOLOe open-vocabulary detector + Ultralytics' built-in BoT-SORT tracker.

Two notes worth remembering:

1. Ultralytics' BoT-SORT association is already class-agnostic - `get_dists`
   uses pure IoU (see ultralytics/trackers/bot_sort.py:211, byte_tracker.py:409).
   The detector's class can flip between frames for the same physical object
   and the track-id will still persist, which is exactly the behaviour we want.

2. YOLOe's `Results.boxes` only exposes the *argmax* class + confidence for
   detection/segmentation tasks. We treat each detection as a **single vote**
   for the argmax class with weight = `yolo_conf` (see Dirichlet update in
   `tracks.py`). An earlier version smeared probability mass over all
   classes; that capped the posterior at ~`conf` and kept entropy high
   forever. A hard vote converges to the true class as observations
   accumulate, which is what the Dirichlet model is for.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLOE

from reflect.pipelines.perception.schemas import Detection2D


# Track buffer of 30 frames keeps a lost id alive for ~1 second at 30 Hz.
# `with_reid` stays False because our own CLIP fallback in `tracks.py`
# handles long-occlusion re-identification more robustly.
_DEFAULT_TRACKER = "botsort.yaml"


class DetectorTracker:
    """Wraps a single YOLOe instance with Ultralytics' stateful tracker."""

    def __init__(
        self,
        weights: str | Path,
        vocab: list[str],
        device: Optional[str] = None,
        conf: float = 0.10,
        tracker: str = _DEFAULT_TRACKER,
    ) -> None:
        self.vocab = list(vocab)
        self.conf = conf
        self.tracker = tracker
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = YOLOE(str(weights))
        self._model.set_classes(self.vocab, self._model.get_text_pe(self.vocab))

    def step(self, rgb: np.ndarray) -> list[Detection2D]:
        """Run one tracked detection frame and return per-detection records.

        `persist=True` is what makes Ultralytics keep the BoT-SORT state across
        successive `step()` calls instead of resetting per-frame.
        """
        results = self._model.track(
            rgb,
            persist=True,
            tracker=self.tracker,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []

        height, width = rgb.shape[:2]
        bboxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = res.boxes.conf.cpu().numpy().astype(np.float32)
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        # `boxes.id` is None until the tracker has assigned ids - happens on frame 1.
        ids = res.boxes.id
        track_ids = ids.cpu().numpy().astype(int) if ids is not None else [None] * len(bboxes)

        masks = self._extract_masks(res, height, width, n=len(bboxes))

        vocab_size = len(self.vocab)
        detections: list[Detection2D] = []
        for i in range(len(bboxes)):
            top_idx = int(cls_ids[i])
            top_label = self.vocab[top_idx] if 0 <= top_idx < vocab_size else "<unk>"
            detections.append(Detection2D(
                bbox_xyxy=bboxes[i],
                mask=masks[i],
                score_vector=_one_hot(top_idx, vocab_size),
                yolo_top_label=top_label,
                yolo_conf=float(confs[i]),
                track_id=int(track_ids[i]) if track_ids[i] is not None else None,
            ))
        return detections

    def _extract_masks(self, res, height: int, width: int, n: int) -> list[np.ndarray]:
        """Return a per-detection list of full-resolution boolean masks.

        YOLOe seg masks come at the model's 640×640 working resolution; we
        resize each one with nearest-neighbour interpolation. If the model
        somehow returned no masks (shouldn't happen with -seg weights) we
        fall back to a filled bbox mask so downstream code keeps working.
        """
        if res.masks is None:
            return [_bbox_to_mask(res.boxes.xyxy[i].cpu().numpy(), height, width)
                    for i in range(n)]
        masks_raw = res.masks.data.cpu().numpy().astype(bool)
        out = []
        for i in range(n):
            mi = masks_raw[i]
            if mi.shape != (height, width):
                mi = cv2.resize(mi.astype(np.uint8), (width, height),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            out.append(mi)
        return out


def _one_hot(top_idx: int, vocab_size: int) -> np.ndarray:
    """Hard one-hot vote for the detector's argmax class.

    Detector uncertainty is encoded via the `weight=yolo_conf` argument to
    `update_alpha` in `tracks.py`, not by smearing probability mass here.
    """
    vec = np.zeros(vocab_size, dtype=np.float32)
    if 0 <= top_idx < vocab_size:
        vec[top_idx] = 1.0
    return vec


def _bbox_to_mask(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rectangular fallback mask when the seg head returns nothing."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    mask = np.zeros((height, width), dtype=bool)
    mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = True
    return mask
