"""Camera intrinsics and depth back-projection.

Numpy-only. The `back_project` function takes an optional
`T_cam_to_world` so callers can plug in robot FK or a SLAM pose
later without changing the rest of the pipeline.
"""

import numpy as np

from reflect.core.constants import (
    MAX_REALSENSE_DEPTH,
    MIN_REALSENSE_DEPTH,
    REAL_FOCAL_LENGTH_X,
    REAL_FOCAL_LENGTH_Y,
    REAL_PRINCIPAL_POINT_X,
    REAL_PRINCIPAL_POINT_Y,
    REAL_SKEW,
)

from reflect.pipelines.perception.schemas import CameraIntrinsics


# Native capture resolution of the RealSense D435i used to collect the episodes.
_D435I_WIDTH = 1280
_D435I_HEIGHT = 720


def real_k() -> CameraIntrinsics:
    """Return the calibrated D435i intrinsics from `reflect.core.constants`."""
    return CameraIntrinsics(
        fx=REAL_FOCAL_LENGTH_X,
        fy=REAL_FOCAL_LENGTH_Y,
        cx=REAL_PRINCIPAL_POINT_X,
        cy=REAL_PRINCIPAL_POINT_Y,
        skew=REAL_SKEW,
        width=_D435I_WIDTH,
        height=_D435I_HEIGHT,
    )


def back_project(
    depth: np.ndarray,
    mask: np.ndarray,
    k: CameraIntrinsics,
    t_cam_to_world: np.ndarray | None = None,
) -> np.ndarray:
    """Lift the masked pixels of `depth` to 3D points.

    Args:
        depth:           (H, W) float32 metres. Zero / out-of-range pixels are dropped.
        mask:            (H, W) bool. True where we want to lift.
        k:               Camera intrinsics matching the depth resolution.
        t_cam_to_world:  Optional (4, 4) homogeneous transform. If given, the
                         output is in world frame; otherwise camera frame.

    Returns:
        (N, 3) float32 array of points. N may be 0 if the mask is empty / all-bad.
    """
    # Combine masks: user mask AND depth-validity. We do this in one pass so we
    # only ever index into `depth` once.
    valid = mask & (depth >= MIN_REALSENSE_DEPTH) & (depth <= MAX_REALSENSE_DEPTH)
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float32)

    vs, us = np.nonzero(valid)
    z = depth[vs, us].astype(np.float32)
    x = (us.astype(np.float32) - k.cx) * z / k.fx
    y = (vs.astype(np.float32) - k.cy) * z / k.fy
    pts_cam = np.stack([x, y, z], axis=1)            # (N, 3)

    if t_cam_to_world is None:
        return pts_cam

    # Apply the homogeneous transform: P_w = R @ P_c + t.
    rot = t_cam_to_world[:3, :3].astype(np.float32)
    trans = t_cam_to_world[:3, 3].astype(np.float32)
    return pts_cam @ rot.T + trans


def project(points_xyz: np.ndarray, k: CameraIntrinsics) -> np.ndarray:
    """Project 3D camera-frame points back to (u, v) pixel coordinates."""
    z = np.clip(points_xyz[:, 2], 1e-6, None)
    u = k.fx * (points_xyz[:, 0] / z) + k.cx
    v = k.fy * (points_xyz[:, 1] / z) + k.cy
    return np.stack([u, v], axis=1)
