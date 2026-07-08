"""One batched LLM call to normalize track labels, propose semantic edges,
and (optionally) merge tracks that describe the same physical object.

ConceptGraphs uses pairwise LLM queries per edge; that gets expensive fast.
We do it in **one** call per scene-state-change instead. A small content
hash of (track_ids + predicted_labels + geometric edges) acts as the cache
key - if nothing changed, no LLM call.

The input payload is a `SceneInput` pydantic model and the output is an
`LLMSceneOutput`. Both live in `schemas.py` so the prompt, the LLM's
structured-output parser, and any downstream code share one source of truth.
"""
from __future__ import annotations

import hashlib
import json

from reflect.llm.prompter import PortkeyLLMPrompter

from reflect.pipelines.perception.process_layer.dirichlet import topk
from reflect.pipelines.perception.schemas import (
    GeomEdge,
    LabelProb,
    LLMSceneOutput,
    SceneInput,
    Track3D,
    TrackRecord,
)


_SYSTEM_PROMPT = (
    "You are the semantic normalizer of a robotic scene graph.\n"
    "\n"
    "The input `SceneInput` contains:\n"
    "  - `tracks`: 3D-tracked objects with position, size, label posterior, and "
    "an observation count.\n"
    "  - `geometric_edges`: VERIFIED spatial relations measured directly from "
    "3D geometry. These are ground truth and form the backbone of the scene "
    "graph. Do NOT contradict them; do NOT duplicate them.\n"
    "\n"
    "Your job is to produce an `LLMSceneOutput` with three pieces:\n"
    "\n"
    "1. `canonical_labels`: one clean kitchen-domain label per surviving track. "
    "Lowercase, singular, no colour/size unless needed for disambiguation. "
    "Example: `metal pot` → `pot`, `blue bowl` → `bowl`.\n"
    "\n"
    "2. `semantic_edges`: edges the geometric backbone is missing. Two kinds, "
    "distinguished by the `source` field on each edge:\n"
    "   (a) SPATIAL GAP-FILLERS - set `source=\"llm_spatial\"`. Use this when "
    "you can infer a clear spatial relation from the centroids and sizes but "
    "the geometric module didn't fire (it has tight thresholds). Use the same "
    "relation vocabulary as geometric edges: `supports`, `inside`, `above`, "
    "plus `beside` for objects at similar heights and close in xz.\n"
    "   (b) FUNCTIONAL / AFFORDANCE - set `source=\"llm_semantic\"`. Use this "
    "for relations no geometry could recover. Examples: `faucet controls sink`, "
    "`stove can_heat pot`, `drawer contains utensil`.\n"
    "\n"
    "3. `merges`: pairs of tracks that describe the SAME physical object. "
    "Propose a merge when ALL hold: same predicted label, centroids within "
    "~40 cm, and either (a) one is a brief high-entropy duplicate adjacent to "
    "a mature low-entropy track, or (b) the centroids are essentially "
    "co-located (within ~10 cm). Do NOT merge two stable long-lived tracks of "
    "the same label if their centroids are clearly separated (>50 cm) - they "
    "might genuinely be two identical objects.\n"
    "\n"
    "Hard rules:\n"
    "  - Every track_id you output MUST appear in the input.\n"
    "  - Every `source` MUST be exactly \"llm_spatial\" or \"llm_semantic\".\n"
    "  - Every `rationale` ≤ 12 words.\n"
    "  - Output strictly the requested JSON schema."
)


def normalize_scene(
    tracks: list[Track3D],
    geom_edges: list[GeomEdge],
    prompter: PortkeyLLMPrompter,
    vocab: list[str],
    min_observations: int = 3,
) -> LLMSceneOutput:
    """Single-shot LLM normalization. Returns the structured output directly.

    Tracks with fewer than ``min_observations`` are dropped before the LLM
    sees them. This matches the default of `propose_edges` so the two
    payloads stay in sync. Set to 0 to disable the filter.
    """
    if min_observations > 1:
        tracks = [t for t in tracks if len(t.history) >= min_observations]

    scene_input = _build_scene_input(tracks, geom_edges, vocab)

    user_text = (
        "Scene state (`SceneInput` schema). The `geometric_edges` list is your "
        "verified backbone - do not contradict, do not duplicate:\n"
        + scene_input.model_dump_json(indent=2)
        + "\n\nReturn JSON matching `LLMSceneOutput`:\n"
        + "  - `canonical_labels`: list of {track_id, label}\n"
        + "  - `semantic_edges`:   list of {src_track_id, dst_track_id, relation, rationale, source}\n"
        + "       where source ∈ {\"llm_spatial\", \"llm_semantic\"}.\n"
        + "  - `merges`:           list of {drop_track_id, keep_track_id, rationale}\n"
        + "Every track_id you reference MUST appear in the input above."
    )

    prompt = {"system": _SYSTEM_PROMPT, "user": user_text}
    # 8192 tokens gives the LLM head-room even with ~20 tracks and dense edges + merges.
    sampling_params = {"max_tokens": 8192, "temperature": 0.0}

    parsed, _ = prompter.query(
        prompt=prompt,
        sampling_params=sampling_params,
        response_model=LLMSceneOutput,
    )
    return parsed


def scene_hash(tracks: list[Track3D], geom_edges: list[GeomEdge], vocab: list[str]) -> str:
    """Content hash that flips iff the scene changed materially.

    Use this from the notebook to decide whether to call `normalize_scene`
    again or reuse the last `LLMSceneOutput`.
    """
    tids = sorted(t.track_id for t in tracks)
    labels = {t.track_id: t.predicted_label(vocab) for t in tracks}
    edges = sorted((e.src_track_id, e.dst_track_id, e.relation) for e in geom_edges)
    payload = json.dumps({"tids": tids, "labels": labels, "edges": edges}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ── Internals ───────────────────────────────────────────────────────────────


def _build_scene_input(
    tracks: list[Track3D],
    geom_edges: list[GeomEdge],
    vocab: list[str],
) -> SceneInput:
    """Construct the typed payload sent to the LLM."""
    track_records: list[TrackRecord] = []
    for t in tracks:
        top = [
            LabelProb(label=vocab[idx], prob=round(prob, 3))
            for idx, prob in topk(t.alpha, k=3)
        ]
        track_records.append(TrackRecord(
            track_id=int(t.track_id),
            yolo_label=t.predicted_label(vocab),
            top3_posterior=top,
            centroid_xyz_m=[float(round(v, 3)) for v in t.last_lifted.centroid],
            size_extent_m=[float(round(v, 3)) for v in t.last_lifted.obb_extent],
            label_entropy=round(t.entropy, 3),
            observation_count=len(t.history),
        ))
    return SceneInput(tracks=track_records, geometric_edges=list(geom_edges))
