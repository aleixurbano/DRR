"""TrackStore - the brain of Layer 1.

For every detection in every frame, the store decides:

  * which existing Track3D (if any) this detection belongs to,
  * how to update that track's Dirichlet posterior over labels,
  * whether to refresh the track's CLIP embedding,
  * when to open a fresh track or prune a stale one.

Local re-implementation of 3D AABB-IoU lives here on purpose - we are
forbidden from importing `reflect.perception.open_world`, and AABB IoU is
~15 lines anyway.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from reflect.pipelines.perception.process_layer.dirichlet import init_alpha, update_alpha
from reflect.pipelines.perception.process_layer.reid_embed import MaskedCLIPEncoder
from reflect.pipelines.perception.schemas import Detection2D, Lifted3D, Track3D


# Tunable association thresholds. Conservative defaults - tighten if you see
# spurious merges, loosen if you see runaway track-id creation.
_IOU3D_REASSOC_MIN = 0.20
_CLIP_REASSOC_MIN = 0.80
_REASSOC_MAX_AGE = 300                  # only re-associate against tracks unseen ≤ this
_POINTS_SAMPLE_MAX = 512                # per-track cloud kept for viz + re-id geometry
_DEFAULT_EMBED_REFRESH_EVERY = 30       # frames between CLIP re-encodes per track

# Defaults for consolidation (the post-hoc duplicate-merge pass).
# Two 10 cm objects sitting 8 cm apart have an AABB-IoU near zero, so the
# IoU gate alone misses duplicates of the same physical object when it has
# moved slightly between observations. The centroid-distance fallback
# catches that - if two tracks share a centroid within `dist_max` AND the
# CLIP gate is happy, we consider them the same.
_CONSOLIDATE_IOU_MIN = 0.40
_CONSOLIDATE_COS_MIN = 0.80
_CONSOLIDATE_DIST_M = 0.15


class TrackStore:
    """Persistent, ID-assigning store of 3D tracks across an episode."""

    def __init__(
        self,
        vocab: list[str],
        embed_refresh_every: int = _DEFAULT_EMBED_REFRESH_EVERY,
        prior: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.vocab = list(vocab)
        self.embed_refresh_every = embed_refresh_every
        self.prior = prior
        self._tracks: dict[int, Track3D] = {}
        self._next_synthetic_id = 10_000          # well above any BoT-SORT id range
        self._rng = rng or np.random.default_rng(0)

        # Maps an *external* track id (from BoT-SORT, or a stale synthetic id
        # that was later merged) to our current canonical internal id. Without
        # this, every frame in which BoT-SORT keeps emitting the lost id forces
        # a fresh re-association from scratch.
        self._alias: dict[int, int] = {}

    # ── Public API ──────────────────────────────────────────────────────────

    def update(
        self,
        rgb: np.ndarray,
        detections: list[Detection2D],
        lifteds: list[Optional[Lifted3D]],
        step_idx: int,
        encoder: Optional[MaskedCLIPEncoder] = None,
    ) -> None:
        """Fold one frame's detections into the store.

        Detections whose 3D lift failed (`lifteds[i] is None`) are dropped -
        we never open a track we can't anchor in 3D.
        """
        if len(detections) != len(lifteds):
            raise ValueError("detections and lifteds must be parallel lists")

        for det, lifted in zip(detections, lifteds):
            if lifted is None:
                continue
            track_id = self._resolve_track_id(det, lifted, rgb, encoder, step_idx)
            self._update_track(track_id, det, lifted, rgb, encoder, step_idx)

    def prune(self, max_age: int = _REASSOC_MAX_AGE) -> None:
        """Drop tracks not updated within the last `max_age` frames."""
        cutoff_step = self._latest_step() - max_age
        survivors = {tid for tid, t in self._tracks.items() if t.last_seen_step >= cutoff_step}
        self._tracks = {tid: self._tracks[tid] for tid in survivors}
        # Drop alias entries that pointed to evicted tracks.
        self._alias = {ext: ours for ext, ours in self._alias.items() if ours in survivors}

    def alive(self) -> list[Track3D]:
        """All current tracks, ordered by track_id."""
        return [self._tracks[k] for k in sorted(self._tracks)]

    def consolidate(
        self,
        iou_min: float = _CONSOLIDATE_IOU_MIN,
        cos_min: float = _CONSOLIDATE_COS_MIN,
        dist_max: float = _CONSOLIDATE_DIST_M,
    ) -> int:
        """Merge duplicate tracks pairwise by (3D-IoU OR centroid distance) + CLIP cosine.

        The IoU gate alone misses duplicates of the same physical object when
        it moved slightly between observations (two ~10 cm objects 8 cm apart
        have IoU near zero). The centroid-distance fallback covers that case.

        Run this every ~5-20 frames and once at the end. Each call is O(T^2)
        on the alive track count.

        Returns the number of tracks dropped.
        """
        ordered = sorted(self._tracks.values(),
                         key=lambda t: (-len(t.history), t.track_id))
        merged: set[int] = set()
        for i, keep in enumerate(ordered):
            if keep.track_id in merged:
                continue
            for drop in ordered[i + 1:]:
                if drop.track_id in merged:
                    continue
                iou = aabb_iou_3d(keep.last_lifted, drop.last_lifted)
                dist = float(np.linalg.norm(
                    keep.last_lifted.centroid - drop.last_lifted.centroid))
                if iou < iou_min and dist > dist_max:
                    continue
                if not _embeddings_match(keep, drop, cos_min):
                    continue
                self._merge_into(keep, drop)
                merged.add(drop.track_id)
        for tid in merged:
            self._tracks.pop(tid, None)
        return len(merged)

    # ── Internals ───────────────────────────────────────────────────────────

    def _resolve_track_id(
        self,
        det: Detection2D,
        lifted: Lifted3D,
        rgb: np.ndarray,
        encoder: Optional[MaskedCLIPEncoder],
        step_idx: int,
    ) -> int:
        """Pick a track id for `det`. Re-associate or open a new track if needed."""
        ext_id = det.track_id

        # Fast path: BoT-SORT id is already our id, or it's been aliased to one.
        if ext_id is not None:
            if ext_id in self._tracks:
                return ext_id
            if ext_id in self._alias and self._alias[ext_id] in self._tracks:
                return self._alias[ext_id]

        # Slow path: try re-association against recently-active tracks.
        reassoc = self._try_reassociate(lifted, rgb, det, encoder, step_idx)
        if reassoc is not None:
            if ext_id is not None:
                # Remember this so subsequent frames take the fast path.
                self._alias[ext_id] = reassoc
            return reassoc

        # Nothing matched - open a new track. Use BoT-SORT's id if available,
        # otherwise a synthetic one above its id range.
        if ext_id is not None:
            return ext_id
        synth = self._next_synthetic_id
        self._next_synthetic_id += 1
        return synth

    def _try_reassociate(
        self,
        lifted: Lifted3D,
        rgb: np.ndarray,
        det: Detection2D,
        encoder: Optional[MaskedCLIPEncoder],
        step_idx: int,
    ) -> Optional[int]:
        """Match `lifted` to an aging track via 3D AABB-IoU + (optional) CLIP cosine."""
        candidates: list[tuple[int, float]] = []
        for tid, track in self._tracks.items():
            age = step_idx - track.last_seen_step
            if age <= 0 or age > _REASSOC_MAX_AGE:
                continue
            iou = aabb_iou_3d(lifted, track.last_lifted)
            if iou >= _IOU3D_REASSOC_MIN:
                candidates.append((tid, iou))
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][0]

        # Multiple geometric candidates - disambiguate with CLIP if we can.
        if encoder is None:
            # No encoder available → pick the highest IoU. May be wrong; that's
            # the price of cheap re-id.
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        det_embed = encoder.encode(rgb, det.mask)
        if det_embed is None:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        best_id, best_cos = None, -1.0
        for tid, _ in candidates:
            t_emb = self._tracks[tid].clip_embedding
            if t_emb is None:
                continue
            cos = float(np.dot(det_embed, t_emb))
            if cos >= _CLIP_REASSOC_MIN and cos > best_cos:
                best_id, best_cos = tid, cos
        return best_id

    def _update_track(
        self,
        track_id: int,
        det: Detection2D,
        lifted: Lifted3D,
        rgb: np.ndarray,
        encoder: Optional[MaskedCLIPEncoder],
        step_idx: int,
    ) -> None:
        """Either refresh an existing track or create a brand new one."""
        if track_id in self._tracks:
            track = self._tracks[track_id]
            track.alpha = update_alpha(track.alpha, det.score_vector, weight=det.yolo_conf)
            track.last_lifted = lifted
            track.points_sample = _reservoir_sample(lifted.points, _POINTS_SAMPLE_MAX, self._rng)
            track.last_seen_step = step_idx
            track.history.append(step_idx)
            if encoder is not None and (step_idx - track.last_embed_step) >= self.embed_refresh_every:
                emb = encoder.encode(rgb, det.mask)
                if emb is not None:
                    track.clip_embedding = emb
                    track.last_embed_step = step_idx
            return

        # Brand new track. Initialize alpha at the prior, then apply the first observation.
        alpha = init_alpha(len(self.vocab), prior=self.prior)
        alpha = update_alpha(alpha, det.score_vector, weight=det.yolo_conf)
        embedding = encoder.encode(rgb, det.mask) if encoder is not None else None
        self._tracks[track_id] = Track3D(
            track_id=track_id,
            alpha=alpha,
            last_lifted=lifted,
            points_sample=_reservoir_sample(lifted.points, _POINTS_SAMPLE_MAX, self._rng),
            last_seen_step=step_idx,
            history=[step_idx],
            clip_embedding=embedding,
            last_embed_step=step_idx if embedding is not None else -10_000,
        )

    def _latest_step(self) -> int:
        if not self._tracks:
            return 0
        return max(t.last_seen_step for t in self._tracks.values())


    def _merge_into(self, keep: Track3D, drop: Track3D) -> None:
        """Fold `drop` into `keep` in-place. `drop` should be removed afterwards."""
        # Combine the Dirichlet evidence by adding pseudo-counts (Bayes by conjugacy).
        # Subtract one copy of the prior so we don't double-count it.
        keep.alpha = keep.alpha + drop.alpha - self.prior
        keep.history.extend(drop.history)
        keep.history.sort()

        if drop.last_seen_step >= keep.last_seen_step:
            keep.last_lifted = drop.last_lifted
            keep.points_sample = drop.points_sample
            keep.last_seen_step = drop.last_seen_step
        # CLIP embedding: keep whichever is newer.
        if drop.last_embed_step > keep.last_embed_step and drop.clip_embedding is not None:
            keep.clip_embedding = drop.clip_embedding
            keep.last_embed_step = drop.last_embed_step

        # Update the alias so any external id that still points at `drop`
        # (e.g., BoT-SORT will keep emitting it) is rerouted to `keep`.
        for ext_id, our_id in list(self._alias.items()):
            if our_id == drop.track_id:
                self._alias[ext_id] = keep.track_id
        self._alias[drop.track_id] = keep.track_id


# ── Free helpers ────────────────────────────────────────────────────────────


def _embeddings_match(a: Track3D, b: Track3D, cos_min: float) -> bool:
    """Pass if both CLIP embeddings exist and align ≥ cos_min, or if either is missing.

    Missing embeddings happen for very fresh tracks; we shouldn't refuse a
    geometry-strong merge just because one side hasn't been re-encoded yet.
    """
    if a.clip_embedding is None or b.clip_embedding is None:
        return True
    return float(np.dot(a.clip_embedding, b.clip_embedding)) >= cos_min


def aabb_iou_3d(a: Lifted3D, b: Lifted3D) -> float:
    """Axis-aligned 3D IoU on the two clouds. Used as a fast OBB-IoU proxy."""
    a_lo, a_hi = a.points.min(axis=0), a.points.max(axis=0)
    b_lo, b_hi = b.points.min(axis=0), b.points.max(axis=0)
    inter_lo = np.maximum(a_lo, b_lo)
    inter_hi = np.minimum(a_hi, b_hi)
    inter = np.clip(inter_hi - inter_lo, 0.0, None).prod()
    if inter <= 0.0:
        return 0.0
    vol_a = np.clip(a_hi - a_lo, 1e-9, None).prod()
    vol_b = np.clip(b_hi - b_lo, 1e-9, None).prod()
    return float(inter / (vol_a + vol_b - inter))


def _reservoir_sample(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= k:
        return points.astype(np.float32, copy=False)
    idx = rng.choice(points.shape[0], size=k, replace=False)
    return points[idx].astype(np.float32, copy=False)
