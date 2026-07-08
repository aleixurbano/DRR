"""Shared value types for the perception pipeline.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for a single calibrated RGB stream."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    skew: float = 0.0


@dataclass
class Detection2D:
    """A single per-frame detection from the open-vocab detector.

    `score_vector` is a normalized distribution over the Layer-0 vocabulary;
    its index axis is the vocab order returned by `layer0_vocab.seed_vocab`.
    """
    bbox_xyxy: np.ndarray            # (4,) float32, image-pixel coords
    mask: np.ndarray                 # (H, W) bool, full image-resolution
    score_vector: np.ndarray         # (V,) float32, sums to ~1
    yolo_top_label: str              # diagnostic only - argmax of score_vector
    yolo_conf: float                 # detector's own confidence in [0, 1]
    track_id: Optional[int] = None   # None if BoT-SORT did not return an id


@dataclass
class Lifted3D:
    """3D representation of one detection: points + oriented bounding box."""
    points: np.ndarray               # (N, 3) float32, in camera frame
    centroid: np.ndarray             # (3,) float32
    obb_center: np.ndarray           # (3,) float32
    obb_extent: np.ndarray           # (3,) float32, half-extents in OBB frame
    obb_rotation: np.ndarray         # (3, 3) float32, columns = OBB axes
    n_valid_depth: int               # how many depth pixels survived filtering


@dataclass
class Track3D:
    """Persistent state for one tracked object across the episode."""
    track_id: int
    alpha: np.ndarray                # (V,) Dirichlet concentration
    last_lifted: Lifted3D
    points_sample: np.ndarray        # (K<=512, 3) reservoir-sampled cloud
    last_seen_step: int
    history: list[int] = field(default_factory=list)
    clip_embedding: Optional[np.ndarray] = None      # (D,) L2-normalized
    last_embed_step: int = -10_000                   # forces a refresh on first hit

    # vocab is shared across all tracks; the store keeps it. Predictive helpers
    # below take it as an argument so the dataclass stays self-contained.
    def predicted_label(self, vocab: list[str]) -> str:
        return vocab[int(np.argmax(self.alpha))]

    @property
    def entropy(self) -> float:
        """Normalized Shannon entropy of E[p] under the Dirichlet. 0 = certain."""
        p = self.alpha / self.alpha.sum()
        # log(V) is the max entropy of a V-dim categorical → normalize into [0, 1]
        eps = 1e-12
        return float(-(p * np.log(p + eps)).sum() / np.log(len(p)))
    
    def summary(self, vocab: list[str]) -> str:
        return f"Track {self.track_id}: '{self.predicted_label(vocab)}' (entropy {self.entropy:.2f}, last seen step {self.last_seen_step})"


# Pydantic models (serialized; used by LLM + JSON snapshot)

# Every edge carries a `source` so downstream consumers know what to trust:
#   "geometric"     - measured directly from 3D point clouds (high precision).
#                     Failure detection should use ONLY these edges.
#   "llm_spatial"   - LLM proposed a spatial relation (above/supports/inside)
#                     that the geometric module didn't fire. Plausible but
#                     not measured.
#   "llm_semantic"  - LLM proposed a functional/affordance edge (controls,
#                     can_heat, contains) that geometry cannot recover. Always
#                     LLM-derived.
EdgeSource = Literal["geometric", "llm_spatial", "llm_semantic"]


class GeomEdge(BaseModel):
    """A geometric scene-graph edge proposed without any LLM.

    Three asymmetric relations:
      * supports - b rests on a (contact + footprint containment).
      * inside   - a is contained inside b's volume.
      * above    - a is vertically above b with horizontal overlap, no contact.
    """
    src_track_id: int
    dst_track_id: int
    relation: Literal["supports", "inside", "above"]
    source: EdgeSource = "geometric"


class SemanticEdge(BaseModel):
    """A scene-graph edge produced by the LLM (spatial gap-fill or affordance).

    `source` distinguishes the two LLM modes:
      * "llm_spatial"  - LLM filled in a spatial relation the geometric module
                         missed (e.g., bowl on cabinet when the geometric
                         containment threshold didn't fire).
      * "llm_semantic" - LLM proposed a functional/affordance relation that no
                         amount of geometry could capture (e.g., faucet controls sink).
    """
    src_track_id: int
    dst_track_id: int
    relation: str = Field(description="Free-form verb phrase, e.g. 'controls'.")
    rationale: str = Field(description="Why the LLM proposed this edge.")
    source: EdgeSource = "llm_semantic"


class CanonicalLabel(BaseModel):
    """Canonical (LLM-normalized) label assigned to a track."""
    track_id: int
    label: str


class TrackMerge(BaseModel):
    """LLM-proposed merge directive: `drop_track_id` is absorbed into `keep_track_id`.

    Used when the LLM concludes two tracks describe the *same physical object*
    (typical case: BoT-SORT lost an id across occlusion, our 3D-IoU + CLIP
    re-id couldn't recover it, and the LLM sees two same-label tracks at
    nearly the same centroid).

    Subject to safety gates in `scene_graph.apply_merges` before being applied.
    """
    drop_track_id: int
    keep_track_id: int
    rationale: str = Field(description="Short reason the two tracks are the same object.")


class LabelProb(BaseModel):
    """One entry in a top-k posterior list."""
    label: str
    prob: float


class TrackRecord(BaseModel):
    """A track as the LLM sees it - only the fields useful for reasoning.

    Mirrors `Track3D` but drops the raw numpy arrays. Includes
    `observation_count` so the LLM can spot transient flicker tracks
    adjacent to mature ones.
    """
    track_id: int
    yolo_label: str
    top3_posterior: list[LabelProb]
    centroid_xyz_m: list[float]
    size_extent_m: list[float]
    label_entropy: float
    observation_count: int


class SceneInput(BaseModel):
    """The full scene payload sent to the LLM in `normalize_scene`."""
    tracks: list[TrackRecord]
    geometric_edges: list[GeomEdge]


class LLMSceneOutput(BaseModel):
    """Structured response from `semantic_layer.llm_edges.normalize_scene`."""
    canonical_labels: list[CanonicalLabel]
    semantic_edges: list[SemanticEdge]
    merges: list[TrackMerge] = Field(default_factory=list)


class SceneGraphSnapshot(BaseModel):
    """The final, serializable scene graph for one moment in time."""
    step: int
    tracks: list[dict]                       # serialized Track3D dicts
    geom_edges: list[GeomEdge]
    sem_edges: list[SemanticEdge]
    canonical_labels: dict[int, str]
