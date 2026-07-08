"""Spatial relation inference via the existing SceneGraph heuristics.

We construct a minimal mock AI2-THOR event so we can reuse `SceneGraph.add_edge`
unchanged. Relation thresholds are scene-normalized in `build_scene_graph`
because monocular depth has arbitrary scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from reflect.perception import scene_graph as sg
from reflect.perception.scene_graph import Node, SceneGraph


class _MockEvent:
    """Just enough metadata to satisfy SceneGraph.__init__ and add_edge."""

    def __init__(self):
        self.metadata = {
            "agent": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "rotation": {"y": 0.0},
                "cameraHorizon": 0.0,
            },
            "objects": [],
        }
        self.frame = None


@dataclass
class FrameNode:
    """Per-mask product of segment + name + backproject, before fusion."""

    name: str
    pcd_camera: np.ndarray   # (N, 3) - camera-frame coords
    bbox2d: np.ndarray       # (4,) y1, x1, y2, x2  (note: SceneGraph uses this convention)
    embedding: np.ndarray    # (D,) CLIP feature for fusion association
    confidence: float
    score: float


def _nodes_from_frame(frame_nodes: Sequence[FrameNode]) -> List[Node]:
    """Convert FrameNodes into reflect.perception.Nodes ready for SceneGraph.add_node."""
    out: List[Node] = []
    seen: dict = {}
    for fn in frame_nodes:
        if fn.pcd_camera.size == 0:
            continue
        # Disambiguate duplicate names so SceneGraph.add_node doesn't try to
        # cross-frame-merge within a single image.
        n = seen.get(fn.name, 0)
        seen[fn.name] = n + 1
        unique_name = fn.name if n == 0 else f"{fn.name} {n+1}"

        pts = fn.pcd_camera
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        # Corner ordering MUST match Open3D's OrientedBoundingBox.get_box_points()
        # because SceneGraph.add_edge() indexes corner_pts[0] as the min corner
        # and corner_pts[4] as the max corner. Wrong ordering breaks the
        # "on top of" / "inside" containment heuristics.
        corner_pts = np.array([
            [mn[0], mn[1], mn[2]],   # 0: min - used by add_edge as box[0]
            [mx[0], mn[1], mn[2]],   # 1: +x
            [mn[0], mx[1], mn[2]],   # 2: +y
            [mn[0], mn[1], mx[2]],   # 3: +z
            [mx[0], mx[1], mx[2]],   # 4: max - used by add_edge as box[4]
            [mn[0], mx[1], mx[2]],   # 5: +y +z
            [mx[0], mn[1], mx[2]],   # 6: +x +z
            [mx[0], mx[1], mn[2]],   # 7: +x +y
        ], dtype=np.float32)

        node = Node(
            name=unique_name,
            object_id=unique_name,
            pos3d=pts.mean(axis=0).astype(np.float32),
            corner_pts=corner_pts,
            bbox2d=fn.bbox2d.astype(np.float32),
            pcd=torch.from_numpy(pts.astype(np.float32)),
            depth=pts[:, 2].astype(np.float32),
            global_node=False,
        )
        out.append(node)
    return out


def _scene_diagonal(nodes: Sequence[Node]) -> float:
    """Largest distance between any two node centroids; used to scale thresholds."""
    if len(nodes) < 2:
        return 1.0
    cents = np.stack([n.pos3d for n in nodes])
    mx = cents.max(axis=0)
    mn = cents.min(axis=0)
    diag = float(np.linalg.norm(mx - mn))
    return max(diag, 1e-3)


def build_scene_graph(
    frame_nodes: Sequence[FrameNode],
    *,
    is_metric: bool = False,
) -> SceneGraph:
    """Construct a SceneGraph and populate edges via the existing heuristics.

    For monocular (non-metric) depth, in_contact / close thresholds are
    rescaled by the scene-AABB diagonal so the heuristics' geometry still
    makes sense in arbitrary units.
    """
    nodes = _nodes_from_frame(frame_nodes)
    graph = SceneGraph(_MockEvent(), task={})

    # Stash & restore the module-level thresholds so we don't permanently mutate
    # the shared module if the caller later runs the AI2-THOR pipeline.
    saved = (sg.IN_CONTACT_DISTANCE, sg.CLOSE_DISTANCE)
    try:
        if not is_metric:
            # Monocular depth has high boundary noise - backprojected pcds of
            # touching objects often sit ~5-15% of the scene diagonal apart.
            # Be more permissive than ConceptGraphs' 0.1m for real RGB-D.
            diag = _scene_diagonal(nodes)
            sg.IN_CONTACT_DISTANCE = max(0.05 * diag, 0.05)
            sg.CLOSE_DISTANCE = max(0.20 * diag, 0.20)
        # else: keep the AI2-THOR-tuned defaults (0.1m in_contact, 0.4m close)
        for n in nodes:
            graph.add_node_wo_edge(n)
        for n in nodes:
            graph.add_node(n)
    finally:
        sg.IN_CONTACT_DISTANCE, sg.CLOSE_DISTANCE = saved

    return graph


def build_scene_graph_with_llm_edges(
    frame_nodes: Sequence[FrameNode],
    labelled_edges: Sequence,         # list of edge_llm.LabelledEdge
) -> SceneGraph:
    """Phase 2 path: package detections + LLM-supplied edges into a SceneGraph.

    Skips the geometric `add_edge` heuristics entirely. Each `LabelledEdge`
    carries `i`, `j`, `direction`, and `edge_type`. Direction `"ij"` means
    `start = nodes[i]`, `end = nodes[j]`. `edge_type=None` means "none of the
    above" - we drop the edge.
    """
    from reflect.perception.scene_graph import Edge

    nodes = _nodes_from_frame(frame_nodes)
    graph = SceneGraph(_MockEvent(), task={})
    for n in nodes:
        graph.add_node_wo_edge(n)
        graph.nodes.append(n)   # bypass heuristic add_node so we don't re-run add_edge

    for le in labelled_edges:
        if le.edge_type is None or le.direction not in ("ij", "ji"):
            continue
        if le.i >= len(nodes) or le.j >= len(nodes):
            continue
        a, b = (nodes[le.i], nodes[le.j]) if le.direction == "ij" else (nodes[le.j], nodes[le.i])
        if a.name == b.name:
            continue
        graph.edges[(a.name, b.name)] = Edge(a, b, le.edge_type)
    return graph
