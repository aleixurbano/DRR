# DRR - Detect, Reason, Re-plan

A research project from the Fontys High Tech Embedded Software (HTES) lectoraat on
embodied AI, developed during a research internship.

This repository is an adaptation and extension of the Stanford/Columbia REFLECT
project. DRR focuses on detecting runtime execution failures in robotic tasks,
reasoning about root causes, and proposing and testing recovery strategies, with
a strong focus on explainability and uncertainty.

- **Detect** - Gather, structure and utilize available data to evaluate (using a foundation model) whether an error occurred during robotic task execution.
- **Reason** - With perception history, reason about the causal error of the task execution.
- **Re-plan** - Propose a new valid plan given the current state of the robot, the environment, and affordances.

This is research code: it prioritizes clarity and reproducibility of the experiments
over production concerns, and ships without a test suite.

## Overview

DRR began as a reproduction of REFLECT and grew into a study of when an LLM's
reasoning can be trusted. The work has three parts.

1. Reproduce and improve the simulation pipeline. Two rounds of error-taxonomy
   annotation over the failed episodes drove targeted fixes: typed structured plans
   in place of free-form text, in-memory and cached validation for fast reruns,
   audio-segment labeling that avoids frame-level spillover, and masked depth
   projection with voxel-hash merging in the scene graph. Across the 100-episode
   simulated failure set, human-scored failure-explanation accuracy rose from 57.6%
   on the initial reproduction to 90.0% on the final pipeline, and the dominant
   error category shifted away from perception (from 31% to 10% of the annotated
   error mass) toward reasoning and context.

2. Make replanning closed-loop and uncertainty-aware. REFLECT generates a recovery
   plan in one shot, which propagates early mistakes. DRR replaces this with an
   iterative observe, propose, select, execute loop (after KnowLoop). A
   chain-of-thought model proposes several candidate next actions, including an
   explicit option to terminate the episode; a second, non-CoT model makes the final
   choice, so its answer-token log-probabilities are a clean confidence signal rather
   than a verbalized guess. Following KnowNo, that confidence feeds conformal
   prediction: a single-action prediction set is executed autonomously, a larger set
   halts and escalates to a human.

3. Measure calibration on the task that matters. A multiple-choice question engine
   turns REFLECT's ground-truth annotations into 821 context-necessary items across
   four failure-reasoning types, with a 2% human-audited degenerate rate. Each model
   is scored on MMLU-Pro and on this failure-reasoning target set with the same
   confidence signal, 1 - normalized entropy of the answer-token log-probabilities,
   and evaluated with ECE plus two strictly proper scoring rules, Brier and NLL.

Headline finding: accuracy transfers across distributions, calibration does not.
gpt-5.4's accuracy carries over from MMLU-Pro to the failure-reasoning set (0.688 to
0.665, a dead heat with qwen3.5:27b at 0.670), but it is the worst-calibrated model
on both distributions and its miscalibration explodes on the target set (ECE 0.165
to 0.281, NLL 0.93 to 1.97) with confidence pinned near 0.94 regardless of
correctness. The Brier score even reverses the model ranking between distributions,
so a standard benchmark can misrank which model's confidence is safe to act on,
while qwen3.5:27b stays accurate and well calibrated on both.
See [Key findings](#key-findings).

## Upstream Baseline

- Original project: [`real-stanford/reflect`](https://github.com/real-stanford/reflect) (MIT)
- Project page: [robot-reflect.github.io](https://robot-reflect.github.io/)
- Upstream scope: REFLECT failure explanation and correction in simulation
- More context: [docs/upstream_baseline.md](docs/upstream_baseline.md), [docs/fork_changes.md](docs/fork_changes.md)

## Project Structure

The codebase is organized as an installable Python package under `src/reflect/`:

```text
.
├── pyproject.toml              # Package metadata, CLI entry points, dependencies
├── .env.example                # Template for API keys and data paths (copy to .env)
├── src/reflect/                # Main package (pip install -e .)
│   ├── core/                   #   Paths, constants, utilities, episode store, exceptions
│   ├── models/                 #   Data models (ActionPrimitive, Plan, TaskData, intrinsics)
│   ├── perception/             #   Scene graphs, CLIP, audio, point clouds, projection,
│   │                           #   open-world perception (YOLOE, captioning, fusion)
│   ├── sim/                    #   AI2-THOR simulation: actions, tasks, recovery, data gen
│   ├── pipelines/              #   Validation, fast validation, reasoning sweeps,
│   │                           #   layered perception pipeline (+ rerun.io viz)
│   ├── llm/                    #   LLM prompters (Ollama, OpenAI, Portkey, Anthropic)
│   ├── real_world/             #   Real-world pipeline: detection, batch validation, GPU workers
│   ├── cli/                    #   CLI entry points (installed as console_scripts)
│   ├── compat/                 #   Compatibility shims (HuggingFace, Open3D)
│   └── configs/                #   Task definitions and prompt templates (JSON)
├── analysis/                   # Research notebooks, scripts, and result artifacts
├── docs/                       # Project documentation
├── environment/                # Conda env specs (environment.yml + lock)
├── third_party/                # Vendored AudioCLIP and MDETR, plus pinned submodules
├── models/                     # Local model weights (gitignored, fetched on demand)
└── sample_data/                # Placeholder for smoke-test fixtures
```

Full directory guide: [docs/repo_map.md](docs/repo_map.md)

## Installation

### Recommended: conda environment

The whole project runs in a self-contained conda environment named
`reflect-py314` (Python 3.14). Creating it installs the package and all extras in
one step, so you can start working right away. Run from the repository root:

```bash
git submodule update --init --recursive   # fetch third_party submodules (perception extras)
conda env create -f environment/environment.yml
conda activate reflect-py314
reflect-check                    # verify paths, keys, and dependencies
```

The submodules under `third_party/` (Grounded-Segment-Anything, gradslam,
chamferdist, concept-graphs) are only needed for the open-world perception and
concept-graph pipelines. The core simulation and LLM validation paths run
without them.

Two specs live under [`environment/`](environment/):

| File | Use it when |
|---|---|
| `environment.yml` | Default. Portable, tracks `pyproject.toml`, installs `[all]` extras + Git-only OpenAI CLIP. |
| `environment.lock.yml` | Bit-for-bit reproduction of the development machine (Linux/CUDA, fully pinned). |

To reproduce the exact environment instead, use the lock file:

```bash
conda env create -f environment/environment.lock.yml
conda activate reflect-py314
```

GPU real-world extras beyond `[all]` are listed in
[`environment/real_world.txt`](environment/real_world.txt).

### Alternative: pip into an existing environment

```bash
pip install -e .

# With optional dependency groups:
pip install -e ".[sim]"          # AI2-THOR simulation extras
pip install -e ".[real-world]"   # Real-world pipeline (GPU deps)
pip install -e ".[llm]"          # OpenAI / Ollama backends
pip install -e ".[analysis]"     # Notebook stack (pandas, jupyter, ipywidgets, ...)
pip install -e ".[dev]"          # Development tools (ruff)
pip install -e ".[all]"          # Everything
```

OpenAI CLIP (used by the real-world pipeline) has no PyPI release; install it
separately with `pip install git+https://github.com/openai/CLIP.git`.

## Configuration: keys and paths

All API keys and machine-specific paths come from environment variables - nothing
is hardcoded. Copy the template and fill it in:

```bash
cp .env.example .env             # .env is gitignored
```

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | OpenAI-backed validation and sweeps |
| `PORTKEY_API_KEY`, `PORTKEY_VIRTUAL_KEY` | Portkey gateway backends |
| `ANTHROPIC_API_KEY` | Anthropic-backed prompting |
| `REFLECT_DATA_ROOT` | Location of the datasets (see below) |
| `REFLECT_OUTPUT_ROOT` | Where generated artifacts are written |

Then verify your setup:

```bash
reflect-check
```

## Dataset

This project uses the **REFLECT dataset** (Liu et al., 2023), which is **not**
included in this repository:

- Download: <https://www.cs.columbia.edu/~liuzeyi/reflect_data>
- Project page: <https://robot-reflect.github.io/>

The expected layout under `REFLECT_DATA_ROOT` is:

```text
REFLECT_DATA_ROOT/
  sim_data/                  # AI2-THOR simulation episodes
  reflect_dataset/
    real_data/               # Real-robot recordings (RealSense + robot state)
```

Annotations and curated evaluation artifacts generated during the internship live
under `outputs/results/` and are documented in
[`analysis/README.md`](analysis/README.md). All generated artifacts share the single
[`outputs/`](outputs/) tree; lightweight results (CSVs, figures, reports, summary
JSON) are tracked in git, while large regenerable artifacts (raw scene graphs,
pickles, caches) are gitignored. See [docs/data_and_artifacts.md](docs/data_and_artifacts.md)
for the full data policy and [`outputs/README.md`](outputs/README.md) for the layout.

## CLI Commands

All entry points are installed as console scripts via `pyproject.toml`:

| Command | Description |
|---|---|
| `reflect-check` | Verify environment setup, paths, and dependencies |
| `reflect-sim` | Run simulation validation episodes |
| `reflect-real` | Run real-world validation |
| `reflect-sweep` | Run reasoning-effort / model comparison sweeps |

## Run Simulation Validation

Local Ollama-backed example:

```bash
reflect-sim \
  --tasks boilWater \
  --episodes boilWater-1 \
  --backend local \
  --model qwen3.5:9b
```

This writes summaries and validation outputs under:

```text
$REFLECT_OUTPUT_ROOT/sim_validation/
```

## Run Reasoning-Effort / Model Comparison

OpenAI-backed example:

```bash
reflect-sweep \
  --tasks boilWater \
  --max-episodes 2 \
  --model gpt-5.4 \
  --reasoning-efforts none,low,medium,high,xhigh
```

This writes sweep payloads under:

```text
$REFLECT_OUTPUT_ROOT/reasoning_sweep/
```

## Real-World Extension

The real-world pipeline extends simulation validation to physical robot setups:

```bash
reflect-real --tasks 1 2
```

Outputs are staged under:

```text
$REFLECT_OUTPUT_ROOT/real_world/
```

## Key findings

Numbers below come from the committed metrics CSVs under `outputs/analysis/`. Full
method and figure walkthrough:
[EXPERIMENT_16_REPORT.md](outputs/analysis/llm_calibration_target_distribution/EXPERIMENT_16_REPORT.md).

Calibration on MMLU-Pro (the standard distribution) versus the failure-reasoning
target set. Confidence is 1 - normalized entropy of the answer-token
log-probabilities; ECE is expected calibration error, Brier and NLL are strictly
proper scoring rules (for all three, lower is better):

| Model | Std acc | Std ECE | Std Brier | Std NLL | Tgt acc | Tgt ECE | Tgt Brier | Tgt NLL |
|---|---|---|---|---|---|---|---|---|
| gpt-5.4 | 0.688 | 0.165 | 0.182 | 0.929 | 0.665 | 0.281 | 0.281 | 1.968 |
| qwen3.5:27b | 0.644 | 0.062 | 0.157 | 0.482 | 0.670 | 0.053 | 0.178 | 0.529 |
| qwen3.5:9b | 0.523 | 0.088 | 0.188 | 0.569 | 0.446 | 0.051 | 0.214 | 0.616 |

- The Brier ranking reverses across distributions: gpt-5.4 beats qwen3.5:9b on
  MMLU-Pro (0.182 vs 0.188) but loses to it on failure reasoning (0.281 vs 0.214),
  and the reversal survives difficulty-balanced reweighting. ECE and NLL keep the
  same ordering on both sets (qwen3.5:27b best, gpt-5.4 worst), so the collapse is
  not a binning artifact.
- On the target set gpt-5.4's entropy-derived confidence saturates near 0.94 and is
  largely independent of correctness, so a confidence threshold would gate almost
  nothing.
- Miscalibration concentrates on the hard diagnostic tasks (failure localization
  accuracy 0.322, attribution 0.366) and grows with item difficulty; outcome
  verification is easy and well calibrated (0.845, ECE 0.041).
- More context hurts: full-trace items are both less accurate (0.449) and worse
  calibrated (ECE 0.323) than plan-only items (0.676, ECE 0.130).

Reproduction against REFLECT, over all 100 simulated failure episodes, split by
ground-truth failure type (numbers are percentages; REFLECT figures are from the
paper's Table 1). Exp is human-scored explanation correctness; Loc is failure-step
localization match; Co-plan is correction plans that executed successfully in
simulation. Loc and Co-plan are computed from the committed final-run outputs by
[`analysis/scripts/reproduction_metrics.py`](analysis/scripts/reproduction_metrics.py)
(see [`outputs/results/sim_data/reproduction_metrics.csv`](outputs/results/sim_data/reproduction_metrics.csv)).

| Metric | DRR exec | DRR plan | DRR all | REFLECT (exec / plan) |
|---|---|---|---|---|
| Explanation (Exp) | 92.4 | 85.3 | 90.0 | 88.4 / 84.2 |
| Localization (Loc) | 75.8 | 88.2 | 80.0 | 96.0 / 80.7 |
| Correction plan (Co-plan) | 45.5 | 52.9 | 48.0 | 79.1 / 80.7 |

Explanation matches or exceeds the paper, and localization is close (above the paper
on planning failures, below on execution). Correction planning is the clear gap (48%
vs about 80%), consistent with the calibration result where the diagnostic and
planning steps are the hard part. Human-scored explanation accuracy improved from
57.6% on the initial reproduction to 90.0% on the final pipeline. The closed-loop
pipeline was run on all 100 episodes; the reasoning-effort sweep (none through xhigh)
was run on a 17-episode subset.

## Results and Notebooks

- Research notebooks (indexed): [`analysis/notebooks/`](analysis/notebooks/) - see [`analysis/README.md`](analysis/README.md)
- Calibration and reproduction artifacts (CSVs, figures, reports): [`outputs/analysis/`](outputs/analysis/)
- Curated result artifacts: [`outputs/results/`](outputs/results/)
- Archived experiments: [`analysis/archive/`](analysis/archive/)

## Data, Artifacts, and Reproducibility Notes

- Full datasets are intentionally excluded from Git (download links above)
- Lightweight curated results (CSVs, figures, reports, summary JSON) are tracked; raw scene graphs, videos, pickles, and caches are gitignored and routed to `REFLECT_OUTPUT_ROOT`
- Vendored third-party code (AudioCLIP, MDETR) is included; the perception submodules and heavyweight checkpoints are fetched separately
- Data and exclusion policy: [docs/data_and_artifacts.md](docs/data_and_artifacts.md)

## Scope and status

The reasoning and calibration pipeline is the focus of the project. A second track,
an open-world real-world perception pipeline, was built to enable testing on larger
and more diverse datasets (for example OpenX-Embodiment) but was de-scoped to keep
the emphasis on reasoning reliability. It remains functional in the codebase under
`src/reflect/perception/open_world/` and `src/reflect/real_world/`. Based on
ConceptGraphs, it adds a context-generation step where a VLM proposes object classes
that YOLOE then uses for open-set detection and mask segmentation, and it tracks
objects with fused IoU and embedding association plus Dirichlet class vectors so
spatial and label uncertainty carry forward across frames. See
[docs/fork_changes.md](docs/fork_changes.md) for the full list of additions.

Possible extensions from here:

- Run the closed-loop, uncertainty-gated pipeline on real-robot and OpenX-Embodiment
  episodes through the perception track above.
- Broaden the calibration study beyond one frontier model and two local models.
- Close the loop on Adapt: persist corrected plans as reusable behavior trees.

## License

This project is released under the [MIT License](LICENSE). It builds on REFLECT
(MIT) - see the LICENSE file for attribution. Vendored code under `third_party/`
retains its original licenses.

## Citation

If you build on this work, please cite:

```bibtex
@misc{urbano2026drr,
  title  = {DRR: Detect, Reason, Re-plan --- When can we trust an LLM's failure reasoning?},
  author = {Urbano, Aleix and Mossavat, Iman},
  year   = {2026},
  url    = {https://github.com/aleixurbano/DRR}
}
```

DRR builds on [REFLECT (Liu, Bahety, and Song, 2023)](https://arxiv.org/abs/2306.15724);
please also cite the original framework where appropriate.
