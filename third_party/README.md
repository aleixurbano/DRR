# `third_party/`

This directory contains vendored external research code needed by the cleaned hand-off repo.

## Vendored source (checked in)

- `audioclip/`: vendored AudioCLIP source used by the real-world audio path
- `mdetr/`: vendored MDETR source used by the real-world detector wrapper

## Git submodules (fetched on demand)

The following are pinned git submodules, not checked-in code. Their weights and
runtime artifacts are intentionally not part of this repo. Fetch them with
`git submodule update --init --recursive`:

- `Grounded-Segment-Anything/`: https://github.com/IDEA-Research/Grounded-Segment-Anything
- `gradslam/`: https://github.com/gradslam/gradslam
- `chamferdist/`: https://github.com/krrish94/chamferdist
- `concept-graphs/`: https://github.com/concept-graphs/concept-graphs

## Excluded On Purpose

- heavyweight model checkpoints
- local experiment caches
- generated detections and runtime artifacts

The goal is source-level reproducibility without turning the repo into a model-weight archive.
