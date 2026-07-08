"""Depth estimation + synthesized pinhole intrinsics for the open-world pipeline.

When the caller supplies depth + intrinsics (e.g., RGB-D from a real robot), they
are returned unchanged. Otherwise we run Depth Anything v2 on the RGB and
synthesize a pinhole K from a default field of view.

Note on units: monocular depth is *relative*, not metric. Downstream relation
heuristics use scene-normalized thresholds, so absolute scale is irrelevant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from PIL import Image

_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"
# AI2-THOR's task_manager.py uses fieldOfView=60° (square frames so HFOV=VFOV=60).
# 60° is also a reasonable middle for typical real-world cameras (Robo2VLM).
_DEFAULT_HFOV_DEG = 60.0


@dataclass
class DepthResult:
    depth: np.ndarray            # (H, W) float32, larger = farther
    intrinsics: np.ndarray       # (3, 3) float32 pinhole K
    is_metric: bool              # True iff supplied by caller; False for monocular


@lru_cache(maxsize=1)
def _load_depth(device: str):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(_DEPTH_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(_DEPTH_MODEL_ID).to(device)
    model.eval()
    return processor, model


def synth_intrinsics(width: int, height: int, hfov_deg: float = _DEFAULT_HFOV_DEG) -> np.ndarray:
    """Pinhole K assuming centered principal point and square pixels."""
    fx = 0.5 * width / math.tan(math.radians(hfov_deg) / 2)
    fy = fx  # square pixels
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    return K


def estimate_depth(
    rgb: np.ndarray,
    *,
    depth: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    hfov_deg: float = _DEFAULT_HFOV_DEG,
) -> DepthResult:
    """Return depth + intrinsics. Uses Depth Anything v2 when caller has no depth."""
    H, W = rgb.shape[:2]
    if depth is not None:
        K = intrinsics if intrinsics is not None else synth_intrinsics(W, H, hfov_deg)
        return DepthResult(depth=depth.astype(np.float32), intrinsics=K.astype(np.float32), is_metric=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = _load_depth(device)

    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    pred = out.predicted_depth  # (1, h, w)

    # Resize to original resolution
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy().astype(np.float32)

    # Depth Anything v2 outputs relative inverse-depth-like values where larger =
    # closer. Invert + rescale so that "larger = farther" matches the convention
    # used by the existing depth_frame_to_camera_space_xyz utility.
    pred = pred.max() - pred
    # Normalize to [0.5, 5.0] range - keeps point clouds in a reasonable scale
    # band for the relation-heuristic AABB diagonals (real metric range varies).
    p_min, p_max = float(pred.min()), float(pred.max())
    if p_max - p_min > 1e-6:
        pred = 0.5 + 4.5 * (pred - p_min) / (p_max - p_min)
    else:
        pred = np.full_like(pred, 1.0)

    K = intrinsics if intrinsics is not None else synth_intrinsics(W, H, hfov_deg)
    return DepthResult(depth=pred, intrinsics=K.astype(np.float32), is_metric=False)


def backproject_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Backproject masked pixels to a (N, 3) camera-frame point cloud.

    Uses the standard pinhole model: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy.
    Y is flipped to match the Unity-style convention used by the rest of the
    perception module ("up" is +y).
    """
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[ys, xs]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xs - cx) * z / fx
    Y = -(ys - cy) * z / fy   # +y up
    Z = z
    return np.stack([X, Y, Z], axis=-1).astype(np.float32)
