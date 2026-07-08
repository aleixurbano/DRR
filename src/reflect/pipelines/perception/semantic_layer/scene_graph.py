"""Assemble Layer-1 tracks + Layer-2 edges into a serializable scene graph.

If the LLM proposed `merges` in its output, we apply them here (after
validating against safety gates) before producing the final snapshot.
Semantic edges and canonical labels are remapped through the merge map
so they reference surviving canonical ids only.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from reflect.pipelines.perception.schemas import (
    GeomEdge,
    LLMSceneOutput,
    SceneGraphSnapshot,
    SemanticEdge,
    Track3D,
    TrackMerge,
)


# Hard safety gates that LLM-proposed merges must satisfy. The LLM gets a
# soft guideline of ~30-40 cm in the prompt; these are the absolute floors
# below which we refuse to merge no matter what the LLM says.
_MAX_CENTROID_DIST = 0.60       # metres
_MAX_VOLUME_RATIO = 3.0         # bigger / smaller OBB volume
_DIRICHLET_PRIOR_TO_SUBTRACT = 0.5   # match TrackStore's init_alpha prior


def build_snapshot(
    tracks: list[Track3D],
    geom_edges: list[GeomEdge],
    llm_out: Optional[LLMSceneOutput],
    step: int,
    vocab: list[str],
) -> SceneGraphSnapshot:
    """Combine all three layers into one serializable snapshot.

    If `llm_out.merges` is non-empty, validated merges are applied to the
    track list before the snapshot is built. Both `canonical_labels` and
    `semantic_edges` are remapped through the merge map so they only
    reference surviving canonical ids.
    """
    canonical: dict[int, str]
    semantic_edges: list[SemanticEdge] = []
    geom_edges_out: list[GeomEdge] = list(geom_edges)

    if llm_out is not None and llm_out.merges:
        tracks, id_remap, rejections = apply_merges(tracks, llm_out.merges, vocab)
        if rejections:
            _log_rejections(rejections)
        geom_edges_out = _remap_geom_edges(geom_edges_out, id_remap)
    else:
        id_remap = {t.track_id: t.track_id for t in tracks}

    survivors = {t.track_id for t in tracks}
    canonical = {t.track_id: t.predicted_label(vocab) for t in tracks}

    if llm_out is not None:
        for cl in llm_out.canonical_labels:
            canonical_id = id_remap.get(cl.track_id, cl.track_id)
            if canonical_id in survivors:
                canonical[canonical_id] = cl.label

        for edge in llm_out.semantic_edges:
            src = id_remap.get(edge.src_track_id, edge.src_track_id)
            dst = id_remap.get(edge.dst_track_id, edge.dst_track_id)
            if src == dst or src not in survivors or dst not in survivors:
                continue
            semantic_edges.append(SemanticEdge(
                src_track_id=src,
                dst_track_id=dst,
                relation=edge.relation,
                rationale=edge.rationale,
            ))

    track_dicts = [_track_to_dict(t, vocab) for t in tracks]
    return SceneGraphSnapshot(
        step=step,
        tracks=track_dicts,
        geom_edges=geom_edges_out,
        sem_edges=semantic_edges,
        canonical_labels=canonical,
    )


def apply_merges(
    tracks: list[Track3D],
    merges: list[TrackMerge],
    vocab: Optional[list[str]] = None,
    *,
    max_centroid_dist: float = _MAX_CENTROID_DIST,
    max_volume_ratio: float = _MAX_VOLUME_RATIO,
    prior: float = _DIRICHLET_PRIOR_TO_SUBTRACT,
) -> tuple[list[Track3D], dict[int, int], list[tuple[TrackMerge, str]]]:
    """Validate and apply LLM-proposed merges to a track list.

    Args:
        tracks: alive tracks.
        merges: LLM-proposed merge directives.
        vocab:  if provided, enables a label-compatibility safety gate. Two
                tracks with different argmax-Dirichlet labels (e.g. a `pot`
                track and a `faucet` track) are NOT merged regardless of how
                close their centroids are. Skip this arg to disable that gate.

    Returns:
        merged_tracks: Tracks after applying every validated merge.
        id_remap:      Map from every input track_id to its surviving canonical id.
                       Use it to redirect downstream edges and labels.
        rejections:    Per-rejected-merge (merge, reason) for logging / debugging.
    """
    by_id = {t.track_id: t for t in tracks}
    accepted: list[TrackMerge] = []
    rejections: list[tuple[TrackMerge, str]] = []

    for m in merges:
        reason = _validate_merge(m, by_id, max_centroid_dist, max_volume_ratio, vocab)
        if reason is None:
            accepted.append(m)
        else:
            rejections.append((m, reason))

    # Union-find groups any chain of accepted merges into one equivalence class.
    parent = {tid: tid for tid in by_id}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for m in accepted:
        ra, rb = find(m.drop_track_id), find(m.keep_track_id)
        if ra != rb:
            # Honour the LLM's direction: drop -> keep.
            parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for tid in by_id:
        groups.setdefault(find(tid), []).append(tid)

    merged: list[Track3D] = []
    id_remap: dict[int, int] = {}
    for members in groups.values():
        if len(members) == 1:
            merged.append(by_id[members[0]])
            id_remap[members[0]] = members[0]
            continue
        # Fold all members into the most-observed one (the canonical).
        sorted_members = sorted(members, key=lambda tid: (-len(by_id[tid].history), tid))
        canonical = by_id[sorted_members[0]]
        for other_id in sorted_members[1:]:
            canonical = _fold_track(canonical, by_id[other_id], prior=prior)
        merged.append(canonical)
        for tid in members:
            id_remap[tid] = canonical.track_id

    merged.sort(key=lambda t: -len(t.history))
    return merged, id_remap, rejections


def to_networkx(snapshot: SceneGraphSnapshot):
    """Return a `networkx.DiGraph` view of the snapshot. Heavy import is local."""
    import networkx as nx

    graph = nx.DiGraph()
    for record in snapshot.tracks:
        graph.add_node(
            record["track_id"],
            label=snapshot.canonical_labels.get(record["track_id"], record["yolo_label"]),
            entropy=record["entropy"],
            centroid=record["centroid"],
        )
    for edge in snapshot.geom_edges:
        graph.add_edge(
            edge.src_track_id, edge.dst_track_id,
            relation=edge.relation, source=edge.source, kind="geometric",
        )
    for edge in snapshot.sem_edges:
        graph.add_edge(
            edge.src_track_id, edge.dst_track_id,
            relation=edge.relation, source=edge.source,
            kind="semantic", rationale=edge.rationale,
        )
    return graph


# ── Internals ───────────────────────────────────────────────────────────────


def _validate_merge(
    merge: TrackMerge,
    by_id: dict[int, Track3D],
    max_centroid_dist: float,
    max_volume_ratio: float,
    vocab: Optional[list[str]] = None,
) -> Optional[str]:
    """Return None if the merge passes safety gates, else a reason string."""
    if merge.drop_track_id == merge.keep_track_id:
        return "self-merge"
    if merge.drop_track_id not in by_id or merge.keep_track_id not in by_id:
        return f"unknown track id (have {sorted(by_id)})"

    drop_t = by_id[merge.drop_track_id]
    keep_t = by_id[merge.keep_track_id]

    # Label-compatibility gate. If both tracks have a clear Dirichlet argmax
    # and they disagree, refuse to merge - the LLM is trying to fold evidence
    # of one object into a track of a different object, which would silently
    # corrupt the alpha vector.
    if vocab is not None:
        drop_label = drop_t.predicted_label(vocab)
        keep_label = keep_t.predicted_label(vocab)
        if drop_label != keep_label:
            return f"label mismatch ({drop_label!r} vs {keep_label!r})"

    dist = float(np.linalg.norm(drop_t.last_lifted.centroid - keep_t.last_lifted.centroid))
    if dist > max_centroid_dist:
        return f"centroid distance {dist:.2f}m > {max_centroid_dist}m"

    vol_drop = float(drop_t.last_lifted.obb_extent.prod())
    vol_keep = float(keep_t.last_lifted.obb_extent.prod())
    ratio = max(vol_drop, vol_keep) / max(min(vol_drop, vol_keep), 1e-9)
    if ratio > max_volume_ratio:
        return f"volume ratio {ratio:.1f}x > {max_volume_ratio}x"
    return None


def _fold_track(keep: Track3D, drop: Track3D, *, prior: float) -> Track3D:
    """Return a new `Track3D` that combines `drop`'s evidence into `keep`.

    Mirrors the merge policy in `TrackStore._merge_into`:
      - Dirichlet alphas add up; one prior is subtracted to avoid double-counting.
      - `last_lifted` and `points_sample` come from the more-recently-seen side.
      - CLIP embedding from whichever side was re-encoded later.
      - History is the sorted union of both histories.
    """
    new_alpha = keep.alpha + drop.alpha - prior
    new_history = sorted(set(keep.history) | set(drop.history))

    if drop.last_seen_step > keep.last_seen_step:
        new_lifted = drop.last_lifted
        new_sample = drop.points_sample
        new_step = drop.last_seen_step
    else:
        new_lifted = keep.last_lifted
        new_sample = keep.points_sample
        new_step = keep.last_seen_step

    if drop.last_embed_step > keep.last_embed_step and drop.clip_embedding is not None:
        new_emb = drop.clip_embedding
        new_emb_step = drop.last_embed_step
    else:
        new_emb = keep.clip_embedding
        new_emb_step = keep.last_embed_step

    return Track3D(
        track_id=keep.track_id,
        alpha=new_alpha,
        last_lifted=new_lifted,
        points_sample=new_sample,
        last_seen_step=new_step,
        history=new_history,
        clip_embedding=new_emb,
        last_embed_step=new_emb_step,
    )


def _remap_geom_edges(
    edges: list[GeomEdge],
    id_remap: dict[int, int],
) -> list[GeomEdge]:
    """Redirect geometric edges through the merge map; drop self-loops + dupes."""
    out: list[GeomEdge] = []
    seen: set[tuple[int, int, str]] = set()
    for e in edges:
        src = id_remap.get(e.src_track_id, e.src_track_id)
        dst = id_remap.get(e.dst_track_id, e.dst_track_id)
        if src == dst:
            continue
        key = (src, dst, e.relation)
        if key in seen:
            continue
        seen.add(key)
        out.append(GeomEdge(src_track_id=src, dst_track_id=dst, relation=e.relation))
    return out


def _log_rejections(rejections: list[tuple[TrackMerge, str]]) -> None:
    """Print rejected merges so the caller can audit what the LLM wanted."""
    if not rejections:
        return
    print(f"[scene_graph] {len(rejections)} LLM-proposed merge(s) rejected:")
    for merge, reason in rejections:
        print(f"  drop #{merge.drop_track_id} → keep #{merge.keep_track_id}: {reason}")
        print(f"    rationale was: {merge.rationale}")


def _track_to_dict(track: Track3D, vocab: list[str]) -> dict:
    """Compact, JSON-safe view of a Track3D for the snapshot payload."""
    return {
        "track_id": int(track.track_id),
        "yolo_label": track.predicted_label(vocab),
        "entropy": float(track.entropy),
        "centroid": [float(v) for v in track.last_lifted.centroid],
        "obb_center": [float(v) for v in track.last_lifted.obb_center],
        "obb_extent": [float(v) for v in track.last_lifted.obb_extent],
        "last_seen_step": int(track.last_seen_step),
        "n_observations": int(len(track.history)),
    }
