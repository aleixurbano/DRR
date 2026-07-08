"""Lift a (mask, depth) pair into a 3D point cloud + oriented bounding box.

Depth from a consumer RGB-D sensor is noisy in two specific ways that
inflate OBBs:

  1. **Edge bleed** at object boundaries - the depth sensor averages over
     a neighbourhood that straddles fore/background, producing pixels with
     wrong depth right at the silhouette.
  2. **Mask leakage** - the seg mask sometimes includes a few background
     pixels, which then back-project as a "ghost" cluster behind the object.

We clean the cloud in four steps before fitting the OBB:

  * **Mask erosion** (kills edge bleed at the silhouette).
  * **Median-depth gate** (kills bimodal mask leakage by keeping only points
    within ±``depth_band_m`` of the mask's median depth).
  * **Statistical outlier removal** (knocks out the few remaining flyers).
  * **Percentile-trimmed OBB extent** (5-95% per local axis instead of
    full min/max, so any flyer that survives the filters can't inflate the box).
"""
from __future__ import annotations

import cv2
import numpy as np

from reflect.compat.open3d import o3d

from reflect.pipelines.perception.intrinsics import back_project, CameraIntrinsics
from reflect.pipelines.perception.schemas import Lifted3D




# Below this many valid depth pixels, we refuse to lift - too noisy to be useful.
_MIN_VALID_PIXELS = 50

# Pixels at the seg-mask boundary are usually depth-bleed; shrink the mask first.
# Scaled to mask size so we don't erode small objects out of existence.
_ERODE_PIXELS_DEFAULT = 3
_MIN_AREA_FOR_EROSION = 200          # below this many mask pixels, skip erosion

# Per-mask depth band - drop pixels whose Z differs from the mask median by more.
# 15 cm covers normal sensor noise for a single tabletop object without
# eating into the object itself.
_DEPTH_BAND_M_DEFAULT = 0.15

# Statistical outlier removal - knock out flyers from depth seams / shiny edges.
_OUTLIER_NEIGHBORS = 20
_OUTLIER_STD_RATIO = 1.5             # tightened from 2.0 (was letting flyers through)

# OBB extent uses a light 1-99% percentile trim along each local axis. With
# mask erosion + median-depth gating + statistical outlier removal upstream,
# the cloud is already clean by the time it reaches us; an aggressive 5-95%
# trim was visibly clipping ~30-50% of the object's real extent on rerun.
_OBB_PERCENTILE = 1.0


def lift_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    k: CameraIntrinsics,
    max_points: int = 2048,
    rng: np.random.Generator | None = None,
    t_cam_to_world: np.ndarray | None = None,
    erode_pixels: int = _ERODE_PIXELS_DEFAULT,
    depth_band_m: float = _DEPTH_BAND_M_DEFAULT,
) -> Lifted3D | None:
    """Lift one mask to a `Lifted3D`. Returns None if too few valid pixels remain.

    The defaults are tuned for RealSense D435i + YOLOe seg masks; bump
    `erode_pixels` if you still see boundary flyers, drop `depth_band_m`
    if you want a more aggressive depth gate.
    """
    cleaned_mask = _erode_mask(mask, erode_pixels) if erode_pixels > 0 else mask
    points = back_project(depth, cleaned_mask, k, t_cam_to_world=t_cam_to_world)
    if points.shape[0] < _MIN_VALID_PIXELS:
        return None

    points = _median_depth_gate(points, band_m=depth_band_m)
    if points.shape[0] < _MIN_VALID_PIXELS:
        return None

    points = _statistical_outlier_filter(points)
    if points.shape[0] < _MIN_VALID_PIXELS:
        return None

    points = _reservoir_sample(points, max_points, rng)
    centroid = points.mean(axis=0).astype(np.float32)
    obb_center, obb_extent, obb_rotation = _fit_obb_pca(points, percentile=_OBB_PERCENTILE)

    return Lifted3D(
        points=points,
        centroid=centroid,
        obb_center=obb_center,
        obb_extent=obb_extent,
        obb_rotation=obb_rotation,
        n_valid_depth=int(points.shape[0]),
    )


# ── Cleaning steps ──────────────────────────────────────────────────────────


def _erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Morphologically erode the mask; skip small masks so they don't vanish."""
    if mask.sum() < _MIN_AREA_FOR_EROSION:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    # If erosion ate everything, fall back to the original mask.
    return eroded.astype(bool) if eroded.any() else mask


def _median_depth_gate(points: np.ndarray, band_m: float) -> np.ndarray:
    """Keep only points whose Z is within ±band_m of the mask's median Z."""
    z = points[:, 2]
    median_z = float(np.median(z))
    return points[np.abs(z - median_z) <= band_m]


def _statistical_outlier_filter(points: np.ndarray) -> np.ndarray:
    """Drop depth flyers using Open3D's statistical outlier removal."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=_OUTLIER_NEIGHBORS,
        std_ratio=_OUTLIER_STD_RATIO,
    )
    return np.asarray(filtered.points, dtype=np.float32)


def _reservoir_sample(
    points: np.ndarray,
    max_points: int,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Uniform random sub-sample without replacement, up to `max_points`."""
    if points.shape[0] <= max_points:
        return points
    rng = rng or np.random.default_rng()
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


# ── OBB fitting ─────────────────────────────────────────────────────────────


def _fit_obb_pca(
    points: np.ndarray,
    percentile: float = _OBB_PERCENTILE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an oriented bounding box via PCA + percentile-trimmed extents.

    Using percentile rather than full min/max for the extents makes the
    box robust to the residual flyers that survive depth-band + SOR. PCA
    axes are a reasonable proxy for the true box axes on regular shapes
    and don't fail on near-coplanar masks the way Open3D's OBB fitter can.

    Returns ``(center, full_extent, rotation)`` matching Open3D's convention.
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    # SVD on the (N, 3) data matrix; columns of Vt.T are principal axes.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    rot = vt.T
    # SVD doesn't guarantee a right-handed frame - det(rot) can be -1 (a
    # reflection). Flip one axis to make it a proper rotation matrix; scipy
    # and rerun both reject reflections.
    if np.linalg.det(rot) < 0:
        rot = rot.copy()
        rot[:, -1] *= -1
    rot = rot.astype(np.float32)
    local = centred @ rot
    lo = np.percentile(local, percentile, axis=0)
    hi = np.percentile(local, 100.0 - percentile, axis=0)
    extent = (hi - lo).astype(np.float32)
    local_center = 0.5 * (lo + hi)
    center = (centroid + rot @ local_center).astype(np.float32)
    return center, extent, rot
