"""Edge candidate selection (3D-AABB MST per component) + LLM relation labelling.

Replaces the brittle `SceneGraph.add_edge()` geometric heuristics in our
single-image perception flow.

Approach (mirrors ConceptGraphs):
  1. Compute pairwise 3D-AABB overlap between object point clouds.
  2. Treat all positively-overlapping pairs as graph edges; find connected
     components; per-component, take the minimum spanning tree.
  3. Add a "near" fallback: any pair whose centroid distance is below a fraction
     of the scene-diagonal is also a candidate (catches legitimately near-by
     objects whose AABBs don't overlap, e.g., two items on the same counter).
  4. Send the candidate list - together with object captions and bbox info -
     to Portkey GPT-5.4 in a single batched JSON-output call.
  5. Parse the response into a list of `(i, j, relation_type)` tuples ready to
     drop into a `SceneGraph`.

The function caches by hash of (caption-list, bbox-list, candidate-list) so
re-running an unchanged scene costs nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

# Relation taxonomy used in the LLM prompt and downstream SceneGraph edges.
# Maps the LLM output to the strings that `get_scene_text_util` already handles
# verbatim (these are the same strings our existing `Edge.edge_type` values use).
LLM_TO_EDGE_TYPE = {
    "on":      "on top of",   # "a on b"  → directed: a (start) on top of b (end)
    "in":      "inside",
    "near":    "near",
    "none":    None,
}
ALLOWED_LLM_RELATIONS = ("on", "in", "near", "none")


@dataclass
class EdgeCandidate:
    i: int
    j: int
    overlap: float        # 3D-AABB IoU (0 if added via "near" fallback only)
    centroid_dist: float  # Euclidean distance between AABB centers


@dataclass
class LabelledEdge:
    i: int                # caller's node index
    j: int                # caller's node index
    relation: str         # one of {"on", "in", "near", "none"} from the LLM
    direction: str        # "ij" means i→j (i on/in/near j); "ji" means j→i; "" for none
    reason: str           # short rationale (debugging / display)
    edge_type: Optional[str]  # final SceneGraph edge_type ("on top of", "inside", "near"); None if "none"


# ---------------------------------------------------------------------------
# Step 1-3: candidate selection
# ---------------------------------------------------------------------------

def _aabb_iou_3d(a_min, a_max, b_min, b_max) -> float:
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    diff = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(diff))
    if inter_vol <= 0:
        return 0.0
    a_vol = float(np.prod(np.maximum(a_max - a_min, 1e-9)))
    b_vol = float(np.prod(np.maximum(b_max - b_min, 1e-9)))
    return inter_vol / max(a_vol + b_vol - inter_vol, 1e-9)


def _scene_diagonal(centroids: np.ndarray) -> float:
    if centroids.shape[0] < 2:
        return 1.0
    mn, mx = centroids.min(axis=0), centroids.max(axis=0)
    return max(float(np.linalg.norm(mx - mn)), 1e-3)


def pick_edge_candidates(
    pcds: Sequence[np.ndarray],
    *,
    near_fraction: float = 0.30,
    overlap_min: float = 0.005,
) -> List[EdgeCandidate]:
    """Return MST edges per connected component plus a "near" fallback set.

    Parameters
    ----------
    pcds : list of (N, 3) arrays - one per object node.
    near_fraction : pairs with centroid distance ≤ near_fraction × scene_diagonal
        are added as candidates even if their 3D AABBs don't overlap.
    overlap_min : minimum 3D-AABB IoU to count an MST edge as connected.
    """
    n = len(pcds)
    if n < 2:
        return []

    aabbs = [(p.min(axis=0), p.max(axis=0)) for p in pcds]
    centroids = np.stack([p.mean(axis=0) for p in pcds])
    diag = _scene_diagonal(centroids)
    near_thresh = near_fraction * diag

    # Pairwise overlap + centroid distances
    overlap_mat = np.zeros((n, n), dtype=np.float32)
    cent_mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            ov = _aabb_iou_3d(aabbs[i][0], aabbs[i][1], aabbs[j][0], aabbs[j][1])
            cd = float(np.linalg.norm(centroids[i] - centroids[j]))
            overlap_mat[i, j] = overlap_mat[j, i] = ov
            cent_mat[i, j] = cent_mat[j, i] = cd

    # 1) MST per connected component on the overlap graph
    rows, cols, weights = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if overlap_mat[i, j] > overlap_min:
                # MST minimises weight; use 1/overlap so high-overlap pairs are preferred.
                rows += [i, j]; cols += [j, i]
                weights += [1.0 / overlap_mat[i, j]] * 2
    mst_edges: List[Tuple[int, int]] = []
    if rows:
        adj = csr_matrix((weights, (rows, cols)), shape=(n, n))
        n_comp, labels = connected_components(adj)
        for c in range(n_comp):
            comp_idx = np.where(labels == c)[0]
            if len(comp_idx) <= 1:
                continue
            sub = adj[comp_idx][:, comp_idx]
            tree = minimum_spanning_tree(sub)
            for u, v in zip(*tree.nonzero()):
                a, b = int(comp_idx[u]), int(comp_idx[v])
                if a != b:
                    mst_edges.append((min(a, b), max(a, b)))

    # 2) "Near" fallback - pairs that are physically close even without AABB overlap.
    near_edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if cent_mat[i, j] <= near_thresh and overlap_mat[i, j] <= overlap_min:
                near_edges.append((i, j))

    # Dedup, build candidate records
    seen: set = set()
    out: List[EdgeCandidate] = []
    for i, j in mst_edges + near_edges:
        if (i, j) in seen:
            continue
        seen.add((i, j))
        out.append(EdgeCandidate(i=i, j=j,
                                  overlap=float(overlap_mat[i, j]),
                                  centroid_dist=float(cent_mat[i, j])))
    return out


# ---------------------------------------------------------------------------
# Step 4-5: batched LLM relation labelling
# ---------------------------------------------------------------------------

_DEFAULT_LLM_MODEL = os.environ.get("REFLECT_EDGE_LLM", "gpt-5.4")


def _portkey_api_key() -> str:
    key = os.environ.get("PORTKEY_API_KEY")
    if not key:
        raise RuntimeError("PORTKEY_API_KEY is not set (see .env.example at the repo root).")
    return key


def _edge_cache_dir() -> Path:
    from reflect.core.paths import analysis_cache_dir

    return Path(os.environ.get("REFLECT_EDGE_CACHE", analysis_cache_dir("edge_llm_cache")))

EDGE_PROMPT = """Given the following list of object pairs from a 3D scene, decide the spatial \
relation between each pair. For every pair, output one of:
  - "a on b"   if object A is physically on top of object B
  - "b on a"   if object B is physically on top of object A
  - "a in b"   if object A is inside object B (e.g. inside a container, drawer, or fridge)
  - "b in a"   if object B is inside object A
  - "a near b" if A and B are close but neither is on/inside the other
  - "none"     if there is no clear physical relation between the two

Use the captions, 3D bbox extents (in meters), and centers to ground your reasoning.
A *small* object is more likely "on" a *larger* one. An object inside a container has a \
smaller bbox roughly contained in the larger one. Default to "near" when in doubt.

Reply with ONLY a JSON object of this exact shape (no markdown, no prose):
{
  "edges": [
    {"i": <int>, "j": <int>, "relation": "<one of: a on b, b on a, a in b, b in a, a near b, none>", "reason": "<short reason>"}
  ]
}
The "i" and "j" fields must echo back the IDs from the input. Output exactly one element \
per input pair, in the same order.
"""


def _hash_key(captions: Sequence[str], aabbs: Sequence, candidates: Sequence[EdgeCandidate]) -> str:
    payload = {
        "captions": list(captions),
        "aabbs": [(np.round(a[0], 3).tolist(), np.round(a[1], 3).tolist()) for a in aabbs],
        "candidates": [(c.i, c.j, round(c.overlap, 3), round(c.centroid_dist, 3)) for c in candidates],
        "model": _DEFAULT_LLM_MODEL,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _build_pair_payload(captions, aabbs, centroids, candidates) -> List[dict]:
    out = []
    for c in candidates:
        a_min, a_max = aabbs[c.i]
        b_min, b_max = aabbs[c.j]
        out.append({
            "i": c.i, "j": c.j,
            "object_a": {
                "caption": captions[c.i],
                "bbox_extent": np.round(a_max - a_min, 2).tolist(),
                "bbox_center": np.round(centroids[c.i], 2).tolist(),
            },
            "object_b": {
                "caption": captions[c.j],
                "bbox_extent": np.round(b_max - b_min, 2).tolist(),
                "bbox_center": np.round(centroids[c.j], 2).tolist(),
            },
            "overlap_3d": round(c.overlap, 3),
            "centroid_distance_m": round(c.centroid_dist, 3),
        })
    return out


def _parse_llm_relation(rel_str: str) -> tuple[str, str]:
    """'a on b' → ('on', 'ij'); 'b in a' → ('in', 'ji'); 'none' → ('none', '')."""
    s = (rel_str or "").strip().lower()
    if s == "none":
        return "none", ""
    m = re.match(r"^([ab])\s+(on|in|near)\s+([ab])$", s)
    if not m:
        return "none", ""
    src, rel, dst = m.group(1), m.group(2), m.group(3)
    if src == dst:
        return "none", ""
    return rel, ("ij" if (src == "a" and dst == "b") else "ji")


def label_edges_batched(
    captions: Sequence[str],
    pcds: Sequence[np.ndarray],
    candidates: Sequence[EdgeCandidate],
    *,
    model: Optional[str] = None,
    use_cache: bool = True,
) -> List[LabelledEdge]:
    """Send all candidate pairs to GPT-5.4 in one batched call."""
    if not candidates:
        return []
    aabbs = [(p.min(axis=0), p.max(axis=0)) for p in pcds]
    centroids = [p.mean(axis=0) for p in pcds]

    cache_dir = _edge_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _hash_key(captions, aabbs, candidates)
    cache_file = cache_dir / f"{cache_key}.json"
    if use_cache and cache_file.exists():
        with open(cache_file) as f:
            return [LabelledEdge(**d) for d in json.load(f)]

    pair_payload = _build_pair_payload(captions, aabbs, centroids, candidates)
    user_msg = json.dumps({"pairs": pair_payload}, indent=0)

    from reflect.llm.prompter import PortkeyLLMPrompter
    prompter = PortkeyLLMPrompter(
        portkey_api_key=_portkey_api_key(),
        model=model or _DEFAULT_LLM_MODEL,
        reasoning_effort="none",
    )
    t0 = time.time()
    resp = prompter._client.chat.completions.create(
        model=model or _DEFAULT_LLM_MODEL,
        messages=[
            {"role": "system", "content": EDGE_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=200 + 30 * len(candidates),
    )
    raw = resp.choices[0].message.content or "{}"
    elapsed = time.time() - t0

    try:
        data = json.loads(raw)
        edges_in = data.get("edges", [])
    except json.JSONDecodeError:
        # Tolerate stray prose around JSON
        m = re.search(r"\{.*\}", raw, re.S)
        edges_in = json.loads(m.group(0)).get("edges", []) if m else []

    by_pair = {(int(e["i"]), int(e["j"])): e for e in edges_in if "i" in e and "j" in e}
    out: List[LabelledEdge] = []
    for c in candidates:
        rec = by_pair.get((c.i, c.j))
        if rec is None:
            out.append(LabelledEdge(i=c.i, j=c.j, relation="none", direction="",
                                     reason="missing_in_response", edge_type=None))
            continue
        rel, direction = _parse_llm_relation(rec.get("relation", ""))
        out.append(LabelledEdge(
            i=c.i, j=c.j,
            relation=rel, direction=direction,
            reason=str(rec.get("reason", "")).strip()[:200],
            edge_type=LLM_TO_EDGE_TYPE.get(rel),
        ))

    print(f"[edge_llm] labelled {len(out)} edges in {elapsed:.2f}s "
           f"(prompt_tokens={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens})")

    with open(cache_file, "w") as f:
        json.dump([e.__dict__ for e in out], f, indent=1)
    return out
