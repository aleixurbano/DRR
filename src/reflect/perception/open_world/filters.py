"""Mask-list post-processing: BG handling, area filters, nested-mask subtraction.

Adapted from concept-graphs/conceptgraph/slam/utils.py:
  - filter_gobs: drop too-small / too-large / low-confidence detections
  - mask_subtract_contained: when one bbox is contained in another, subtract
    the inner mask from the outer (so the same surface isn't counted twice)
"""
from __future__ import annotations

from collections import deque
from typing import List, Sequence

import numpy as np

from .detector import Detection


def _bbox_contained(inner: np.ndarray, outer: np.ndarray, tol: int = 5) -> bool:
    return (
        inner[0] >= outer[0] - tol
        and inner[1] >= outer[1] - tol
        and inner[2] <= outer[2] + tol
        and inner[3] <= outer[3] + tol
    )


def mask_subtract_contained(detections: Sequence[Detection]) -> List[Detection]:
    """If detection j's bbox is contained inside detection i's bbox (i != j),
    subtract j's mask from i's mask. Mirrors ConceptGraphs' behaviour: it keeps
    the outer (larger-area) mask but punches out the contained inner masks so
    a countertop isn't counted twice once a knife is also detected on it.
    """
    if len(detections) < 2:
        return list(detections)
    out = [Detection(**{**d.__dict__, "mask": d.mask.copy()}) for d in detections]

    bboxes = np.stack([d.bbox_xyxy for d in out])
    for i in range(len(out)):
        for j in range(len(out)):
            if i == j:
                continue
            # If j is contained in i, subtract j's mask from i's
            if _bbox_contained(bboxes[j], bboxes[i]):
                out[i].mask = out[i].mask & ~out[j].mask
    return out


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    b_area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
    return inter / (a_area + b_area - inter)


def class_nms(detections: Sequence[Detection], iou_thresh: float = 0.4) -> List[Detection]:
    """Per-class NMS: keep highest-confidence detection per class when boxes
    overlap above `iou_thresh`. YOLOe already does cross-class NMS internally,
    but sometimes still emits multiple boxes for the same physical object when
    text-prompt scoring is ambiguous. This is the safety net.
    """
    if not detections:
        return []
    by_class: dict[str, list[Detection]] = {}
    for d in detections:
        by_class.setdefault(d.class_name, []).append(d)
    keep: list[Detection] = []
    for cls, items in by_class.items():
        items.sort(key=lambda d: d.confidence, reverse=True)
        survivors: list[Detection] = []
        for d in items:
            if any(_bbox_iou(d.bbox_xyxy, s.bbox_xyxy) >= iou_thresh for s in survivors):
                continue
            survivors.append(d)
        keep.extend(survivors)
    keep.sort(key=lambda d: d.confidence, reverse=True)
    return keep


def filter_detections(
    detections: Sequence[Detection],
    image_shape: tuple,
    *,
    min_mask_area: int = 200,
    max_bbox_area_ratio: float = 0.6,
    skip_bg: bool = True,
    min_conf: float = 0.05,
) -> List[Detection]:
    """Drop tiny / huge / low-confidence detections.

    `max_bbox_area_ratio` does NOT apply to background classes - those are
    legitimately allowed to span most of the image.
    """
    if not detections:
        return []
    H, W = image_shape[:2]
    image_area = H * W
    out: list[Detection] = []
    for d in detections:
        if skip_bg and d.is_background:
            continue
        if int(d.mask.sum()) < min_mask_area:
            continue
        if d.confidence < min_conf:
            continue
        if not d.is_background:
            x1, y1, x2, y2 = d.bbox_xyxy
            if (x2 - x1) * (y2 - y1) > max_bbox_area_ratio * image_area:
                continue
        out.append(d)
    return out


def _largest_voxel_connected_component(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Return points in the largest 26-connected occupied-voxel component."""
    if points.shape[0] == 0:
        return points

    vox = np.floor(points / voxel_size).astype(np.int64)
    uniq_vox, inv, counts = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
    if uniq_vox.shape[0] <= 1:
        return points

    key_to_idx = {tuple(v.tolist()): i for i, v in enumerate(uniq_vox)}
    visited = np.zeros(uniq_vox.shape[0], dtype=bool)
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    best_component: list[int] = []
    best_count = 0

    for start in range(uniq_vox.shape[0]):
        if visited[start]:
            continue
        q: deque[int] = deque([start])
        visited[start] = True
        comp: list[int] = []
        comp_count = 0

        while q:
            cur = q.popleft()
            comp.append(cur)
            comp_count += int(counts[cur])
            x, y, z = uniq_vox[cur]
            for dx, dy, dz in offsets:
                nei = key_to_idx.get((int(x + dx), int(y + dy), int(z + dz)))
                if nei is None or visited[nei]:
                    continue
                visited[nei] = True
                q.append(nei)

        if comp_count > best_count:
            best_count = comp_count
            best_component = comp

    if not best_component:
        return points

    keep_vox = np.zeros(uniq_vox.shape[0], dtype=bool)
    keep_vox[np.asarray(best_component, dtype=np.int64)] = True
    keep_points = keep_vox[inv]
    out = points[keep_points]
    return out if out.shape[0] >= 5 else points


def _statistical_outlier_filter(points: np.ndarray, *, n_neighbors: int = 12, z_thresh: float = 2.5) -> np.ndarray:
    """Drop sparse outliers using mean k-NN distance with bounded memory."""
    n = points.shape[0]
    if n <= max(5, n_neighbors + 1):
        return points
    try:
        from sklearn.neighbors import NearestNeighbors

        k = min(max(3, n_neighbors), n - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree", n_jobs=1)
        nbrs.fit(points)
        dists, _ = nbrs.kneighbors(points, return_distance=True)
        # Exclude self-distance in column 0.
        mean_d = dists[:, 1:].mean(axis=1)
        mu = float(mean_d.mean())
        sigma = float(mean_d.std())
        if sigma <= 1e-8:
            return points
        keep = mean_d <= (mu + z_thresh * sigma)
    except Exception:
        return points

    out = points[keep]
    return out if out.shape[0] >= 5 else points


def dbscan_largest_cluster(points: np.ndarray, eps: float = 0.05, min_points: int = 10) -> np.ndarray:
    """Keep the largest coherent object surface from a (N, 3) point cloud.

    This keeps the legacy API name for compatibility, but the implementation is
    intentionally non-DBSCAN to avoid DBSCAN's worst-case O(N^2) memory growth
    on dense masks. Steps:
      1) keep only finite xyz points,
      2) keep largest 3D connected component in voxel space,
      3) remove sparse outliers with bounded-memory k-NN statistics.

    `eps` is interpreted as voxel size in metres (or depth-relative units).
    """
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return points

    pts = np.asarray(points, dtype=np.float32)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] < max(5, min_points * 2):
        return pts

    voxel_size = max(float(eps), 1e-4)
    keep = _largest_voxel_connected_component(pts, voxel_size=voxel_size)
    if keep.shape[0] < max(5, min_points):
        return keep

    keep = _statistical_outlier_filter(
        keep,
        n_neighbors=max(8, min_points),
        z_thresh=2.5,
    )
    return keep
