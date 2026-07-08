"""Real-time visualization via rerun.io.

Entity tree layout:

    /world
        /camera              Pinhole + RGB + Depth
        /camera/dets         2D bounding boxes in image plane
        /tracks/<id>/points  per-track 3D point cloud
        /tracks/<id>/obb     per-track oriented bounding box
        /graph               final scene graph as a JSON text document

All entities are keyed on the ``step`` timeline so the viewer can scrub.
"""
from __future__ import annotations

import colorsys
from typing import Optional

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation

from reflect.pipelines.perception.schemas import (
    CameraIntrinsics,
    Detection2D,
    SceneGraphSnapshot,
    Track3D,
)

_TIMELINE = "step"


class RerunStream:
    """Open a rerun session and log frames as they are processed."""

    def __init__(
        self,
        app_id: str = "reflect_perception_pipeline",
        spawn_viewer: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        # Disconnect any leftover stream from a previous notebook run before
        # re-initialising - avoids "orphaned gRPC recording" errors in Jupyter.
        try:
            rr.disconnect()
        except Exception:
            pass
        try:
            rr.init(app_id, spawn=spawn_viewer)
        except RuntimeError:
            try:
                rr.disconnect()
            except Exception:
                pass
            rr.init(app_id, spawn=spawn_viewer)

        if save_path is not None:
            rr.save(save_path)

        # +Y-down, +Z-forward camera convention (OpenCV / RealSense).
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # ── Per-frame ──────────────────────────────────────────────────────────

    def log_frame(
        self,
        step: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> None:
        """Log RGB-D frame and camera intrinsics."""
        rr.set_time(_TIMELINE, sequence=step)
        rr.log("world/camera", rr.Pinhole(
            image_from_camera=_intrinsics_matrix(intrinsics),
            width=intrinsics.width,
            height=intrinsics.height,
        ))
        rr.log("world/camera/rgb", rr.Image(rgb))
        rr.log("world/camera/depth", rr.DepthImage(depth, meter=1.0))

    def log_detections(
        self,
        step: int,
        detections: list[Detection2D],
        tracks_by_id: dict[int, Track3D],
        vocab: list[str],
    ) -> None:
        """Log 2D bounding boxes with track labels in image space."""
        rr.set_time(_TIMELINE, sequence=step)
        if not detections:
            rr.log("world/camera/dets", rr.Clear(recursive=False))
            return

        bboxes = np.array([d.bbox_xyxy for d in detections], dtype=np.float32)
        xywh = np.stack([
            bboxes[:, 0],
            bboxes[:, 1],
            bboxes[:, 2] - bboxes[:, 0],
            bboxes[:, 3] - bboxes[:, 1],
        ], axis=1)

        labels, colors = [], []
        for det in detections:
            tid = det.track_id if det.track_id is not None else -1
            track = tracks_by_id.get(tid)
            if track is not None:
                labels.append(f"#{tid} {track.predicted_label(vocab)}  H={track.entropy:.2f}")
            else:
                labels.append(f"#? {det.yolo_top_label} ({det.yolo_conf:.2f})")
            colors.append(_rgba_for_id(tid))

        rr.log("world/camera/dets", rr.Boxes2D(
            array=xywh,
            array_format=rr.Box2DFormat.XYWH,
            labels=labels,
            colors=colors,
        ))

    def log_tracks_3d(
        self,
        step: int,
        tracks: list[Track3D],
        vocab: list[str],
    ) -> None:
        """Log per-track 3D point cloud and oriented bounding box."""
        rr.set_time(_TIMELINE, sequence=step)
        rr.log("world/tracks", rr.Clear(recursive=True))

        for track in tracks:
            color = _rgba_for_id(track.track_id)
            label = f"#{track.track_id} {track.predicted_label(vocab)}"

            rr.log(
                f"world/tracks/{track.track_id}/points",
                rr.Points3D(
                    positions=track.points_sample.astype(np.float32),
                    colors=color,
                    radii=0.005,
                ),
            )
            quat = Rotation.from_matrix(track.last_lifted.obb_rotation).as_quat().astype(np.float32)
            rr.log(
                f"world/tracks/{track.track_id}/obb",
                rr.Boxes3D(
                    centers=track.last_lifted.obb_center.reshape(1, 3).astype(np.float32),
                    half_sizes=(track.last_lifted.obb_extent / 2.0).reshape(1, 3).astype(np.float32),
                    quaternions=[rr.datatypes.Quaternion(xyzw=quat)],
                    colors=[color],
                    labels=[label],
                ),
            )

    def log_scene_graph(self, step: int, snapshot: SceneGraphSnapshot) -> None:
        """Log the final scene graph as a JSON text document."""
        rr.set_time(_TIMELINE, sequence=step)
        rr.log("world/graph", rr.TextDocument(snapshot.model_dump_json(indent=2)))


# ── Helpers ────────────────────────────────────────────────────────────────


def _intrinsics_matrix(k: CameraIntrinsics) -> np.ndarray:
    return np.array([
        [k.fx, k.skew, k.cx],
        [0.0,  k.fy,   k.cy],
        [0.0,  0.0,    1.0 ],
    ], dtype=np.float32)


def _rgba_for_id(track_id: int) -> list[int]:
    """Stable, visually distinct RGBA colour per track id."""
    if track_id < 0:
        return [128, 128, 128, 255]
    hue = (track_id * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return [int(r * 255), int(g * 255), int(b * 255), 255]
