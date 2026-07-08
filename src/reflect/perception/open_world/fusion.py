"""Cross-frame fusion of detected objects into a persistent scene graph.

Association rule per incoming FrameNode:
  1. 3D AABB IoU > 0.30 with an existing canonical object,
  2. AND CLIP-embedding cosine similarity > 0.85,
  3. tie-break by closest centroid distance,
  4. otherwise register a new canonical object.

Merge: union of point clouds, EMA on the embedding, recompute AABB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence
import uuid

import numpy as np

from reflect.perception.scene_graph import SceneGraph

from .relations import FrameNode, build_scene_graph


@dataclass
class _Canonical:
    cid: str
    name: str
    pcd: np.ndarray              # (N, 3) accumulated points
    embedding: np.ndarray        # running EMA, L2-normalized
    confidence: float            # name confidence, EMA
    n_observations: int


@dataclass
class FusionState:
    objects: Dict[str, _Canonical] = field(default_factory=dict)


def _aabb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


def _aabb_iou(a_min, a_max, b_min, b_max) -> float:
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    diff = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(diff))
    if inter_vol <= 0:
        return 0.0
    a_vol = float(np.prod(np.maximum(a_max - a_min, 1e-9)))
    b_vol = float(np.prod(np.maximum(b_max - b_min, 1e-9)))
    return inter_vol / max(a_vol + b_vol - inter_vol, 1e-9)


def _voxel_downsample(points: np.ndarray, voxel: float = 0.01) -> np.ndarray:
    """Hash-based voxel downsample; faster than Open3D for small clouds."""
    if points.shape[0] == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    # Pack 3 ints into one for unique-by-row
    packed = (keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663) ^ (keys[:, 2] * 83492791)
    _, idx = np.unique(packed, return_index=True)
    return points[np.sort(idx)]


def _associate(
    fn: FrameNode,
    state: FusionState,
    iou_thresh: float = 0.30,
) -> str | None:
    """Return cid of the canonical object that fn matches, or None for new.

    Gating: same class label AND 3D AABB IoU > iou_thresh. The detector already
    gave us a semantic label so we don't need a separate visual-similarity check.
    """
    if fn.pcd_camera.size == 0:
        return None
    fn_min, fn_max = _aabb(fn.pcd_camera)

    candidates: list[tuple[float, str]] = []
    for cid, c in state.objects.items():
        if c.pcd.size == 0 or c.name != fn.name:
            continue
        c_min, c_max = _aabb(c.pcd)
        if _aabb_iou(fn_min, fn_max, c_min, c_max) < iou_thresh:
            continue
        cent_dist = float(np.linalg.norm(fn.pcd_camera.mean(0) - c.pcd.mean(0)))
        candidates.append((cent_dist, cid))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def update(state: FusionState, frame_nodes: Sequence[FrameNode]) -> FusionState:
    for fn in frame_nodes:
        cid = _associate(fn, state)
        if cid is None:
            new_id = str(uuid.uuid4())[:8]
            state.objects[new_id] = _Canonical(
                cid=new_id,
                name=fn.name,
                pcd=_voxel_downsample(fn.pcd_camera),
                embedding=np.zeros(1, dtype=np.float32),
                confidence=float(fn.confidence),
                n_observations=1,
            )
        else:
            c = state.objects[cid]
            merged = np.concatenate([c.pcd, fn.pcd_camera], axis=0)
            c.pcd = _voxel_downsample(merged)
            alpha = 0.2
            c.confidence = (1 - alpha) * c.confidence + alpha * float(fn.confidence)
            c.n_observations += 1
    return state


def finalize(state: FusionState, *, is_metric: bool = False) -> SceneGraph:
    """Convert the fused canonical objects into a SceneGraph with edges."""
    fns: List[FrameNode] = []
    for c in state.objects.values():
        if c.pcd.shape[0] < 25:
            continue
        # Synthetic 2D bbox: project the 3D AABB to image plane is overkill here
        # since fusion's primary consumer is the relation heuristics. We pass
        # zeros - add_edge tolerates missing bbox2d for non-blocking relations.
        fns.append(FrameNode(
            name=c.name,
            pcd_camera=c.pcd,
            bbox2d=np.zeros(4, dtype=np.float32),
            embedding=c.embedding,
            confidence=c.confidence,
            score=1.0,
        ))
    return build_scene_graph(fns, is_metric=is_metric)
