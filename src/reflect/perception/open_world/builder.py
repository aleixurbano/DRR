"""End-to-end open-world perception following ConceptGraphs' detection-first
architecture: YOLOe (text-prompt) → SAM3 mask refinement → backproject →
SceneGraph relations.

Single-image entry: `perceive_image`
Multi-frame entry: `perceive_rgbd_sequence`
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

from reflect.core.utils import get_scene_text_util
from reflect.perception.scene_graph import SceneGraph

from . import depth as _depth
from . import detector as _det
from . import filters as _filt
from . import relations as _rel
from . import segmentor as _seg
from .fusion import FusionState, _voxel_downsample, update as _fusion_update, finalize as _fusion_finalize
from . import captioner as _cap
from . import edge_llm as _edge_llm

# Adaptive voxel target. Relation heuristics are O(N²) per pair; we keep object
# point clouds modest so a 25-detection scene is still ~1s.
_PCD_TARGET_POINTS = 300

ImageInput = Union[str, Path, np.ndarray, Image.Image]


@dataclass
class Frame:
    rgb: np.ndarray
    depth: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None  # (4, 4) world←camera; None = identity


def _to_rgb(image: ImageInput) -> np.ndarray:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        return np.asarray(image)
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"image must be HxWx3 RGB, got {arr.shape}")
    return arr


def _build_frame_nodes(
    rgb: np.ndarray,
    depth: Optional[np.ndarray],
    intrinsics: Optional[np.ndarray],
    vocabulary: Optional[Sequence[str]],
    max_detections: int,
    use_sam3: bool,
    hfov_deg: float,
    captioner_backend: Optional[str] = None,
) -> tuple[List[_rel.FrameNode], bool]:
    # 1. YOLOe detection (open-vocab, labeled bboxes + initial masks)
    detections = _det.detect(rgb, vocabulary=vocabulary, max_detections=max_detections)
    if not detections:
        return [], False

    # 2. Filter background, tiny, oversized, low-confidence
    detections = _filt.filter_detections(detections, rgb.shape)
    if not detections:
        return [], False

    # 3. Per-class NMS - safety net for YOLOe text-prompt mode emitting
    #    multiple low-confidence boxes for the same physical object.
    detections = _filt.class_nms(detections, iou_thresh=0.4)

    # 4. Subtract nested masks so a contained knife doesn't double-count the counter
    detections = _filt.mask_subtract_contained(detections)

    # 4. SAM3 refines each mask using bbox + class_name as prompts
    if use_sam3:
        refined_masks = _seg.refine_masks(rgb, detections)
        for d, m in zip(detections, refined_masks):
            if int(m.sum()) >= 25:
                d.mask = m

    # 5. Depth (provided or monocular Depth Anything v2) + backproject per mask
    depth_res = _depth.estimate_depth(rgb, depth=depth, intrinsics=intrinsics, hfov_deg=hfov_deg)
    out: List[_rel.FrameNode] = []
    for d in detections:
        pcd = _depth.backproject_mask(d.mask, depth_res.depth, depth_res.intrinsics)
        if pcd.shape[0] < 25:
            continue
        # 6. DBSCAN: keep largest cluster (drops backprojection noise from
        # monocular-depth boundaries).
        pcd = _filt.dbscan_largest_cluster(pcd, eps=0.05, min_points=10)
        if pcd.shape[0] < 25:
            continue
        # 7. Voxel-downsample to ~_PCD_TARGET_POINTS for tractable relation heuristics.
        if pcd.shape[0] > _PCD_TARGET_POINTS:
            extent = float(np.linalg.norm(pcd.max(0) - pcd.min(0))) + 1e-6
            voxel = extent / (_PCD_TARGET_POINTS ** (1 / 3))
            pcd = _voxel_downsample(pcd, voxel=voxel)
        x1, y1, x2, y2 = d.bbox_xyxy
        bbox2d = np.array([y1, x1, y2, x2], dtype=np.float32)  # SceneGraph: y1,x1,y2,x2

        # Phase 2: replace the YOLOe class label with a VLM caption per object.
        # Falls back to the YOLOe label if captioner_backend is None.
        node_name = d.class_name
        if captioner_backend is not None:
            try:
                cap_res = _cap.caption(rgb, d.mask, d.bbox_xyxy, backend=captioner_backend)
                if cap_res.text and cap_res.text != "unknown":
                    node_name = cap_res.text
            except Exception as e:
                print(f"[builder] captioner failed for {d.class_name}: {type(e).__name__}: {str(e)[:80]}")

        out.append(_rel.FrameNode(
            name=node_name,
            pcd_camera=pcd,
            bbox2d=bbox2d,
            embedding=np.zeros(1, dtype=np.float32),  # detector-driven; fusion uses class name + 3D IoU
            confidence=d.confidence,
            score=d.confidence,
        ))
    return out, depth_res.is_metric


def perceive_image(
    image: ImageInput,
    *,
    intrinsics: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
    vocabulary: Optional[Sequence[str]] = None,
    max_detections: int = 25,
    use_sam3: bool = True,
    hfov_deg: float = 60.0,
    use_llm_scene_graph: bool = False,
    captioner_backend: Optional[str] = None,
    edge_llm_model: Optional[str] = None,
    return_text: bool = True,
) -> Union[str, SceneGraph]:
    """Single-image open-world perception.

    `use_llm_scene_graph=True` switches on the Phase 2 path:
      - VLM captions per object (set `captioner_backend`, e.g. "qwen3.5:9b")
      - 3D-AABB MST + "near" candidate selection
      - GPT-5.4 batched relation labelling (set `edge_llm_model` to override)
    """
    rgb = _to_rgb(image)
    cap_backend = captioner_backend if use_llm_scene_graph else None
    frame_nodes, is_metric = _build_frame_nodes(
        rgb, depth, intrinsics, vocabulary, max_detections, use_sam3, hfov_deg,
        captioner_backend=cap_backend,
    )

    if use_llm_scene_graph and frame_nodes:
        captions = [fn.name for fn in frame_nodes]
        pcds = [fn.pcd_camera for fn in frame_nodes]
        candidates = _edge_llm.pick_edge_candidates(pcds)
        labelled = _edge_llm.label_edges_batched(
            captions, pcds, candidates, model=edge_llm_model,
        )
        graph = _rel.build_scene_graph_with_llm_edges(frame_nodes, labelled)
    else:
        graph = _rel.build_scene_graph(frame_nodes, is_metric=is_metric)

    return get_scene_text_util(graph) if return_text else graph


def perceive_rgbd_sequence(
    frames: Iterable[Frame],
    *,
    vocabulary: Optional[Sequence[str]] = None,
    max_detections: int = 25,
    use_sam3: bool = True,
    hfov_deg: float = 60.0,
    return_text: bool = False,
) -> Union[str, SceneGraph]:
    """Multi-frame perception with cross-frame fusion of objects.

    `pose` per Frame is a (4, 4) world←camera matrix. None = identity.
    """
    state = FusionState()
    last_metric: bool = False
    for frame in frames:
        rgb = _to_rgb(frame.rgb)
        frame_nodes, is_metric = _build_frame_nodes(
            rgb, frame.depth, frame.intrinsics, vocabulary, max_detections, use_sam3, hfov_deg,
        )
        if frame.pose is not None:
            R = frame.pose[:3, :3]
            t = frame.pose[:3, 3]
            for fn in frame_nodes:
                fn.pcd_camera = (fn.pcd_camera @ R.T) + t
        state = _fusion_update(state, frame_nodes)
        last_metric = is_metric

    graph = _fusion_finalize(state, is_metric=last_metric)
    return get_scene_text_util(graph) if return_text else graph
