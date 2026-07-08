"""Batch ConceptGraphs over Robo2VLM-1 to produce reference scene graphs.

Calibration workstream - the runtime never imports this. Output graphs feed
``analysis/notebooks/bench_llm_scene_uncertainty.ipynb`` so candidate LLMs can be
evaluated against a known-good scene representation.

The heavy per-scene ConceptGraphs pipeline (RAM → GroundingDINO → SAM → CLIP →
depth backproject → MapObjectList → LLaVA → GPT scene graph) lives in
``analysis/notebooks/perception_conceptgraphs_real.ipynb``. This script is
the batch driver: it loads Robo2VLM-1 from HuggingFace, manages the output
manifest, and invokes :func:`process_scene` once per sample. To swap in the
real pipeline, replace :func:`process_scene` with the notebook's logic - the
imports at the top of the notebook (cell 2) are reproduced below so the
script picks up the same third_party paths.

Usage:

    pip install -e .[calibration]
    python analysis/scripts/build_reference_scene_graphs.py \\
        --out-dir analysis/outputs/conceptgraphs_robo2vlm \\
        --limit 50
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

REFLECT_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REFLECT_ROOT / "third_party"

# Match the notebook's sys.path bootstrap so this script imports the same
# ConceptGraphs / Grounded-SAM / SAM modules.
_PATHS = [
    str(REFLECT_ROOT / "src"),
    str(THIRD_PARTY / "concept-graphs"),
    str(THIRD_PARTY / "Grounded-Segment-Anything"),
    str(THIRD_PARTY / "Grounded-Segment-Anything" / "segment_anything"),
]
for _p in _PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from reflect.core.paths import analysis_experiment_dir  # noqa: E402


def load_robo2vlm(limit: int | None = None) -> Iterable[dict]:
    """Stream Robo2VLM-1 samples. Each sample is the raw HF dict for the user
    to extract RGB/depth/intrinsics from."""
    from datasets import load_dataset

    ds = load_dataset("keplerccc/Robo2VLM-1", split="train", streaming=True)
    for idx, sample in enumerate(ds):
        if limit is not None and idx >= limit:
            return
        yield {"scene_idx": idx, "sample": sample}


def process_scene(scene_idx: int, sample: dict, out_path: Path) -> dict | None:
    """Build a ConceptGraphs scene map for one Robo2VLM-1 sample.

    The notebook's cells 7-21 implement this: RAM/GroundingDINO/SAM detection,
    OpenCLIP features, depth backproject, MapObjectList fusion, LLaVA captions,
    GPT scene-graph refinement, and gzipped-pickle save. To enable batched
    runs, port the relevant cells into this function and write to ``out_path``
    in the same ``scene_map.pkl.gz`` format the notebook produces.

    Returns a metadata dict (keys to be defined by the porting work) so the
    manifest can record per-scene status. Returns ``None`` if the scene is
    skipped.
    """
    raise NotImplementedError(
        "Port cells 7-21 of perception_conceptgraphs_real.ipynb here. "
        "This stub raises so an incomplete port is loud rather than silent."
    )


def _save_raw_inputs(scene_idx: int, sample: dict, raw_dir: Path) -> Path:
    """Persist RGB + depth + intrinsics for offline ConceptGraphs runs.

    We don't know the exact Robo2VLM-1 sample schema until we inspect a real
    record, so for now we just pickle the whole sample. The porting work in
    :func:`process_scene` can read this file back.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"scene_{scene_idx:06d}.pkl.gz"
    with gzip.open(path, "wb") as fh:
        pickle.dump(sample, fh)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(analysis_experiment_dir("perception_conceptgraphs_robo2vlm")))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip scenes whose output pkl.gz already exists.")
    parser.add_argument("--save-raw", action="store_true",
                        help="Cache raw Robo2VLM samples to disk before scene-graph build.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Iterate the dataset and update the manifest without calling process_scene.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    manifest_path = out_dir / "manifest.json"
    manifest: dict[str, Any] = {"entries": []}
    if manifest_path.exists():
        with open(manifest_path, "r") as fh:
            manifest = json.load(fh)
    seen = {entry["scene_idx"] for entry in manifest.get("entries", [])}

    for entry in load_robo2vlm(limit=args.limit):
        scene_idx = entry["scene_idx"]
        sample = entry["sample"]
        scene_path = out_dir / f"scene_{scene_idx:06d}.pkl.gz"
        if args.skip_existing and scene_path.exists():
            print(f"[skip] scene_{scene_idx:06d} (output exists)")
            continue

        if args.save_raw:
            _save_raw_inputs(scene_idx, sample, raw_dir)

        if args.dry_run:
            print(f"[dry-run] scene_{scene_idx:06d} keys={list(sample.keys())[:6]}")
            status = "dry-run"
            meta = None
        else:
            try:
                meta = process_scene(scene_idx, sample, scene_path)
                status = "ok"
            except NotImplementedError as exc:
                print(f"[stub] scene_{scene_idx:06d}: {exc}")
                status = "not-implemented"
                meta = None
            except Exception as exc:  # noqa: BLE001
                print(f"[err]  scene_{scene_idx:06d}: {exc}")
                status = f"error: {exc}"
                meta = None

        if scene_idx in seen:
            for entry_dict in manifest["entries"]:
                if entry_dict["scene_idx"] == scene_idx:
                    entry_dict.update({"status": status, "meta": meta})
                    break
        else:
            manifest["entries"].append({
                "scene_idx": scene_idx,
                "scene_path": str(scene_path),
                "status": status,
                "meta": meta,
            })
            seen.add(scene_idx)

    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"manifest written → {manifest_path} ({len(manifest['entries'])} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
