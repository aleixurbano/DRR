"""SAM3 mask refinement given YOLOe-detected bboxes + class labels.

SAM3 is concept-aware: it accepts both `text` (the class label) and
`input_boxes` (bbox prompts), producing higher-quality masks than YOLOe alone.

Requires `huggingface-cli login` with a token that has accepted the
facebook/sam3 license.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image

_SAM3_MODEL_ID = "facebook/sam3"


@lru_cache(maxsize=1)
def _load_sam3(device: str):
    from transformers import Sam3Model, Sam3Processor
    proc = Sam3Processor.from_pretrained(_SAM3_MODEL_ID)
    model = Sam3Model.from_pretrained(_SAM3_MODEL_ID).to(device)
    model.eval()
    return proc, model


def _refine_class_batch(rgb_pil: Image.Image, class_name: str, bboxes: list[list[float]],
                          proc, model, device: str, H: int, W: int) -> list[np.ndarray | None]:
    """Run SAM3 once on all bboxes for a single class. Returns one mask per box."""
    inputs = proc(
        images=rgb_pil,
        text=class_name,
        input_boxes=[bboxes],
        input_boxes_labels=[[1] * len(bboxes)],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = inputs.get("original_sizes").tolist() if "original_sizes" in inputs else [(H, W)]
    results = proc.post_process_instance_segmentation(
        outputs, threshold=0.3, mask_threshold=0.5, target_sizes=target_sizes,
    )[0]
    masks = results.get("masks")
    if masks is None or len(masks) == 0:
        return [None] * len(bboxes)

    out_masks_np = []
    for m in masks:
        if torch.is_tensor(m):
            m = m.cpu().numpy()
        out_masks_np.append(np.asarray(m).astype(bool))

    if len(out_masks_np) == len(bboxes):
        return out_masks_np

    # SAM3 may return more or fewer masks than input boxes (concept-aware
    # filtering). Match each input bbox to its highest-IoU SAM3 mask.
    matched: list[np.ndarray | None] = []
    for box in bboxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        best_iou, best_mask = 0.0, None
        for m in out_masks_np:
            if m.shape != (H, W):
                continue
            crop = m[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            inside = int(crop.sum())
            total = int(m.sum())
            if total == 0:
                continue
            iou = inside / total
            if iou > best_iou:
                best_iou, best_mask = iou, m
        matched.append(best_mask)
    return matched


def refine_masks(
    rgb: np.ndarray,
    detections: Sequence,           # list of detector.Detection
) -> List[np.ndarray]:
    """Refine each detection's mask using SAM3, batched by class for speed."""
    if not detections:
        return []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc, model = _load_sam3(device)
    pil = Image.fromarray(rgb)
    H, W = rgb.shape[:2]

    by_class: dict[str, list[int]] = {}
    for i, d in enumerate(detections):
        by_class.setdefault(d.class_name, []).append(i)

    out: list[np.ndarray | None] = [None] * len(detections)
    for class_name, idxs in by_class.items():
        bboxes = [detections[i].bbox_xyxy.tolist() for i in idxs]
        try:
            class_masks = _refine_class_batch(pil, class_name, bboxes, proc, model, device, H, W)
        except Exception as e:
            print(f"[sam3] error on class={class_name!r}: {type(e).__name__}: {str(e)[:100]}")
            class_masks = [None] * len(bboxes)
        for det_idx, m in zip(idxs, class_masks):
            out[det_idx] = m

    final: list[np.ndarray] = []
    for i, m in enumerate(out):
        if m is None or m.shape != (H, W) or int(m.sum()) < 25:
            m = detections[i].mask  # YOLOe fallback
        final.append(m.astype(bool))
    return final
