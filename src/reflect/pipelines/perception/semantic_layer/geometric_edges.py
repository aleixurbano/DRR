"""Geometric scene-graph edges proposed without any LLM.

Three asymmetric relations are inferred from the 3D point clouds + OBBs:

  * ``supports(a, b)``  - b rests *on* a (contact + xz-footprint containment).
  * ``inside(a, b)``    - most of a's points lie inside b's OBB.
  * ``above(a, b)``     - a is vertically above b with horizontal overlap,
                          but NOT in contact (otherwise it would be `supports`).

We deliberately do NOT emit a `near` relation. With centroids and OBB extents
already in the LLM payload, "near" carries no actionable information - the LLM
can infer proximity on its own. Direction (`above`), contact (`supports`) and
containment (`inside`) are the relations the LLM cannot recover from positions
alone, and so they are the only ones we emit.

Convention
----------
The pipeline runs in **camera frame** by default. In the OpenCV camera
convention, +Y points roughly downward (toward gravity for a typical
workspace camera). "Above" therefore means *smaller* Y values. If your
mount breaks this assumption, compose a ``T_cam_to_world`` upstream so
the +Y-down rule holds.
"""
from __future__ import annotations

import numpy as np

from reflect.pipelines.perception.schemas import GeomEdge, Track3D


# ── Tunables (defaults work for tabletop-distance objects at ~5-40 cm scale) ──
#
# These thresholds are tuned for *recall* over precision. The LLM augments
# this module by filling in spatial edges we miss (with `source="llm_spatial"`),
# so a few extra geometric edges in the snapshot are not a problem. False
# positives are rare even with loose gates because the predicates remain
# mathematically grounded.

# `supports`: how close b's bottom (max Y) must be to a's top (min Y), and how
# much of b's xz-footprint must overlap a's. The containment threshold is
# asymmetric on purpose - an apple sitting on a big table has tiny IoU but
# high containment.
_SUPPORTS_Z_TOLERANCE = 0.03            # 3 cm contact slack
_SUPPORTS_CONTAINMENT = 0.25            # ≥ 25 % of b's xz-footprint inside a's

# `inside`: fraction of a's sampled points that must fall inside b's OBB.
# We also require b's OBB volume to be meaningfully larger than a's - otherwise
# any pair of near-overlapping tracks (e.g. duplicate detections of the same
# physical object) produces bogus inside edges.
_INSIDE_POINT_FRACTION = 0.55
_INSIDE_VOLUME_RATIO = 1.5              # vol_b ≥ ratio * vol_a

# `above`: minimum vertical clearance for "above without contact" (anything
# tighter than this is `supports`'s territory), and how much horizontal
# overlap to require so that we don't call e.g. a far-corner cabinet
# "above" a centre-of-scene plate.
_ABOVE_MIN_GAP = _SUPPORTS_Z_TOLERANCE  # below this gap, the contact predicate owns it
_ABOVE_HORIZONTAL_OVERLAP = 0.10        # ≥ 10 % footprint containment either direction


# ── Public API ──────────────────────────────────────────────────────────────


def propose_edges(tracks: list[Track3D], min_observations: int = 3) -> list[GeomEdge]:
    """Compute all geometric edges for the given alive tracks.

    Tracks with fewer than ``min_observations`` are dropped before any
    predicate runs - flicker tracks with one or two observations have
    near-prior-flat posteriors, near-noise OBBs, and contribute almost
    exclusively spurious edges. Set to 0 to disable the filter.

    Returns a deduplicated list of ``GeomEdge`` records. Ordering is
    stable: `supports`, then `inside`, then `above`.
    """
    if min_observations > 1:
        tracks = [t for t in tracks if len(t.history) >= min_observations]
    if len(tracks) < 2:
        return []

    edges: list[GeomEdge] = []
    pairs = [(a, b) for a in tracks for b in tracks if a.track_id != b.track_id]

    # Cache per-track AABBs and volumes once - every predicate uses them.
    cache = {t.track_id: _track_geom(t) for t in tracks}

    for a, b in pairs:
        if _supports(a, b, cache):
            edges.append(_edge(a, b, "supports"))

    for a, b in pairs:
        if _inside(a, b, cache):
            edges.append(_edge(a, b, "inside"))

    for a, b in pairs:
        if _above(a, b, cache):
            edges.append(_edge(a, b, "above"))

    return edges


# ── Predicates ──────────────────────────────────────────────────────────────


def _supports(a: Track3D, b: Track3D, cache: dict) -> bool:
    """True if b sits in contact on top of a along the camera-frame Y axis."""
    ga, gb = cache[a.track_id], cache[b.track_id]

    # b must be physically above a (smaller centroid-Y in +Y-down).
    if gb["centroid_y"] >= ga["centroid_y"]:
        return False

    # Contact: b's bottom (max Y of b) within tolerance of a's top (min Y of a).
    a_top_y = ga["lo"][1]
    b_bot_y = gb["hi"][1]
    if abs(b_bot_y - a_top_y) > _SUPPORTS_Z_TOLERANCE:
        return False

    # b's footprint sits over a's. Containment, not IoU (an apple on a table
    # has near-zero IoU but ~1.0 containment).
    return _xz_containment(ga["lo"], ga["hi"], gb["lo"], gb["hi"]) >= _SUPPORTS_CONTAINMENT


def _inside(a: Track3D, b: Track3D, cache: dict) -> bool:
    """True if most of a's points fall inside b's OBB and b is plausibly larger."""
    if a.last_lifted.points.shape[0] == 0:
        return False

    ga, gb = cache[a.track_id], cache[b.track_id]
    # Reject if b isn't meaningfully bigger than a - kills the
    # cup-inside-cup symptom that appears when re-id splits a single
    # physical object into two near-overlapping tracks.
    if gb["volume"] < _INSIDE_VOLUME_RATIO * max(ga["volume"], 1e-9):
        return False

    centre = b.last_lifted.obb_center
    half = b.last_lifted.obb_extent / 2.0
    rot = b.last_lifted.obb_rotation                   # columns are OBB axes
    local = (a.last_lifted.points - centre) @ rot      # a's points in b's OBB frame
    inside_mask = np.all(np.abs(local) <= half, axis=1)
    return float(inside_mask.mean()) >= _INSIDE_POINT_FRACTION


def _above(a: Track3D, b: Track3D, cache: dict) -> bool:
    """True if a is vertically above b with horizontal overlap, but NOT touching.

    Captures structural verticality with a gap - e.g. a shelf above a counter,
    a pot held above a burner, a vent above a stove. `supports` owns the
    in-contact case.
    """
    ga, gb = cache[a.track_id], cache[b.track_id]

    # a's lowest point (max Y) must be at least _ABOVE_MIN_GAP above b's
    # highest point (min Y). In +Y-down this is `b.min_y - a.max_y > gap`.
    gap = gb["lo"][1] - ga["hi"][1]
    if gap < _ABOVE_MIN_GAP:
        return False

    # Horizontal overlap, either direction (so a small light above a big
    # counter and a big light above a small counter both register).
    cont_a_in_b = _xz_containment(gb["lo"], gb["hi"], ga["lo"], ga["hi"])
    cont_b_in_a = _xz_containment(ga["lo"], ga["hi"], gb["lo"], gb["hi"])
    return max(cont_a_in_b, cont_b_in_a) >= _ABOVE_HORIZONTAL_OVERLAP


# ── Helpers ─────────────────────────────────────────────────────────────────


def _track_geom(track: Track3D) -> dict:
    """Precompute per-track geometric quantities (cheap; called once per propose)."""
    points = track.last_lifted.points
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    extent = track.last_lifted.obb_extent
    return {
        "lo": lo,
        "hi": hi,
        "centroid_y": float(track.last_lifted.centroid[1]),
        "volume": float(extent[0] * extent[1] * extent[2]),
    }


def _xz_containment(a_lo: np.ndarray, a_hi: np.ndarray,
                    b_lo: np.ndarray, b_hi: np.ndarray) -> float:
    """Fraction of b's xz-footprint that overlaps a's. Asymmetric on purpose."""
    inter_lo = np.maximum(a_lo[[0, 2]], b_lo[[0, 2]])
    inter_hi = np.minimum(a_hi[[0, 2]], b_hi[[0, 2]])
    inter = float(np.clip(inter_hi - inter_lo, 0.0, None).prod())
    if inter <= 0.0:
        return 0.0
    area_b = max(1e-9, (b_hi[0] - b_lo[0]) * (b_hi[2] - b_lo[2]))
    return inter / area_b


def _edge(a: Track3D, b: Track3D, relation: str) -> GeomEdge:
    return GeomEdge(src_track_id=a.track_id, dst_track_id=b.track_id, relation=relation)
