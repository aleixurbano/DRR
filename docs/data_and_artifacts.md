# Data And Artifacts

## The REFLECT Dataset

This project builds on the dataset released with the REFLECT paper
(Liu, Bahety, Song - 2023). It is **not** distributed with this repository:

- Download: <https://www.cs.columbia.edu/~liuzeyi/reflect_data>
- Project page: <https://robot-reflect.github.io/>
- Upstream repository: <https://github.com/real-stanford/reflect>

## What Is Excluded From Git

This repo is intentionally lightweight. The following are excluded from version
control:

- Full simulation dataset
- Full real-world dataset
- Generated scene-graph caches and recovered videos
- Local environments and installers
- Model checkpoints (`*.pt`, `*.pth`, `*.ts` are gitignored; `models/` holds local copies)

### Model checkpoints

| Checkpoint | Used by | Where to get it |
|---|---|---|
| YOLOE weights (e.g. `yoloe-26x-seg.pt`) | open-world perception, `perception_yoloe_real.ipynb` | fetched automatically by `ultralytics` on first use |
| AudioCLIP checkpoint | real-world audio path | AudioCLIP releases (see `third_party/audioclip/README.md`) |
| MDETR checkpoints | real-world detection | downloaded via `torch.hub` from the vendored `hubconf.py` |
| RAM / Grounded-SAM weights | `perception_conceptgraphs_real.ipynb` (external experiment) | [recognize-anything](https://github.com/xinyu1205/recognize-anything) and [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) releases |

## Expected External Layout

Set (or put in `.env`, copied from `.env.example`):

```bash
export REFLECT_DATA_ROOT=/path/to/data_root
export REFLECT_OUTPUT_ROOT=/path/to/output_root
```

Expected layout:

```text
REFLECT_DATA_ROOT/
  sim_data/
  reflect_dataset/
    real_data/
```

## Reference Sizes From The Original Workspace

Approximate sizes observed in the internship workspace at cleanup time:

- `datasets/sim_data`: ~529 GB
- `datasets/reflect_dataset`: ~30 GB

Those figures explain why this repo does not try to be self-contained.

## Output Locations

All generated artifacts share a single root (`REFLECT_OUTPUT_ROOT`, default
`outputs/`):

- `REFLECT_OUTPUT_ROOT/sim_validation/` - `reflect-sim`
- `REFLECT_OUTPUT_ROOT/reasoning_sweep/` - `reflect-sweep`
- `REFLECT_OUTPUT_ROOT/real_world/` - `reflect-real`
- `REFLECT_OUTPUT_ROOT/llm/` - LLM traces/caches
- `REFLECT_OUTPUT_ROOT/analysis/` - notebook scratch, caches, and per-run dirs
- `REFLECT_OUTPUT_ROOT/results/` - curated evaluation artifacts

Within that tree, lightweight results (CSVs, figures, reports, summary JSON) are
tracked in git; large regenerable artifacts (raw scene graphs, pickles, model-call
caches, videos, backups) are gitignored. See [`outputs/README.md`](../outputs/README.md)
for the per-subfolder layout and the `reflect.core.paths` helper for each.

## Research Artifacts

This repo tracks the research notebooks and supporting scripts, plus the
lightweight curated results they produce:

- research notebooks (machine-independent: all paths and keys come from the
  environment) under [`analysis/`](../analysis/)
- archived side-experiment notes without raw media or checkpoints
- the curated CSV/JSON/figure evaluation outputs and annotations generated during
  the internship live under `outputs/results/` and `outputs/analysis/`, inventoried
  in [`analysis/README.md`](../analysis/README.md); the raw intermediates behind
  them (scene-graph dumps, pickles, caches) are gitignored and regenerated with the
  pipelines and `sim_*`/`real_*` notebooks

## Third-Party Assets

- Vendored AudioCLIP and MDETR source code is included for reproducibility
- Grounded-Segment-Anything, gradslam, chamferdist, and concept-graphs are pinned
  git submodules (fetch with `git submodule update --init --recursive`)
- Heavy checkpoints are intentionally excluded and must be supplied separately
  (see the checkpoint table above)
