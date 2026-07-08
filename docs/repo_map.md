# Repo Map

## Top-Level Layout

```text
.
├── pyproject.toml        # Package config, CLI entry points, dependencies
├── .env.example          # Template for API keys and data paths (copy to .env)
├── LICENSE               # MIT, with upstream REFLECT attribution
├── src/reflect/          # Installable Python package
├── analysis/             # Research notebooks, scripts, result artifacts
├── docs/                 # Project documentation and upstream figures
├── environment/          # Conda env specs (environment.yml, environment.lock.yml)
├── third_party/          # Vendored AudioCLIP and MDETR code, plus pinned submodules
├── models/               # Local model weights (gitignored, fetched on demand)
├── sample_data/          # Placeholder for smoke-test fixtures
└── outputs/              # Generated artifacts (lightweight results tracked, large files gitignored)
```

## Package Sub-Packages (`src/reflect/`)

| Sub-package | Purpose | Key modules |
|---|---|---|
| `core/` | Shared foundations | `paths.py` (all path routing, env-driven), `constants.py` (NAME_MAP, categories), `utils.py` (embeddings, pathfinding), `episode_store.py`, `data.py`, `exceptions.py` |
| `models/` | Data models | `action_primitive.py` (ActionPrimitive), `plan.py` (Plan with `to_legacy_strings()`), `task_data.py` (TaskData), `intrinsics.py` |
| `perception/` | Scene understanding | `scene_graph.py` (SceneGraph, Node), `clip.py`, `audio.py`, `point_cloud.py`, `local_graph.py`, `projection/` (camera projection ops), `open_world/` (YOLOE detection, segmentation, captioning, depth backprojection, fusion, LLM edge labelling) |
| `sim/` | AI2-THOR simulation | `actions.py` (action primitives), `task_manager.py` (task utils), `recovery.py` (replan execution), `data_gen.py`, `assets/` (sound WAVs) |
| `pipelines/` | End-to-end workflows | `validation.py` (full pipeline), `fast_validation.py` (cached/in-memory), `reasoning_sweep.py` (model comparison), `perception/` (layered perception pipeline: discovery/process/semantic layers, schemas, rerun.io viz) |
| `llm/` | LLM backends | `prompter.py` (LocalLLMPrompter, OpenAILLMPrompter, PortkeyLLMPrompter, AnthropicLLMPrompter, logprob scoring) |
| `real_world/` | Real-world pipeline | `batch_validation.py`, `batch_reasoning.py`, `detection.py` (MDETR), `scene_graph.py`, `local_graph.py`, `prompting.py`, `worker_pool.py`, `diagrams/` |
| `cli/` | Console entry points | `sim_validation.py`, `real_world_validation.py`, `reasoning_sweep.py`, `check_setup.py` |
| `compat/` | Compatibility shims | `hf.py` (HuggingFace hub), `open3d.py` (Open3D import) |
| `configs/` | Configuration files | `tasks_sim.json`, `tasks_real_world.json`, `prompts/` (sim and real-world prompt templates) |

## Import Convention

All imports use absolute paths from the package root:

```python
from reflect.core.paths import sim_data_root, validation_artifacts
from reflect.models.plan import Plan
from reflect.perception.scene_graph import SceneGraph
from reflect.llm.prompter import LocalLLMPrompter
from reflect.pipelines.fast_validation import process_episode_mem
```

Two deliberate exceptions inside `real_world/`: `hubconf` and `transforms` resolve
against the vendored MDETR checkout, which `reflect.cli.real_world_validation` puts
on `sys.path` before the pipeline imports run.

## CLI Entry Points

Installed via `pyproject.toml [project.scripts]`:

| Command | Module | Purpose |
|---|---|---|
| `reflect-sim` | `reflect.cli.sim_validation` | Run simulation validation |
| `reflect-real` | `reflect.cli.real_world_validation` | Run real-world validation |
| `reflect-sweep` | `reflect.cli.reasoning_sweep` | Reasoning-effort sweeps |
| `reflect-check` | `reflect.cli.check_setup` | Environment verification |

## Supporting Directories

- `analysis/notebooks/` - research notebooks (index in `analysis/README.md`)
- `analysis/scripts/` - batch drivers supporting the notebooks
- `analysis/archive/` - archived side experiments and the upstream demo notebook
- `outputs/` - single tree for all generated artifacts. Lightweight curated results (`outputs/results/`, `outputs/analysis/`: CSVs, figures, reports, summary JSON) are tracked; raw runtime output, notebook scratch, and caches are gitignored (see `outputs/README.md`)
- `environment/environment.yml` - recommended conda environment (`reflect-py314`, Python 3.14; portable, installs `[all]` extras)
- `environment/environment.lock.yml` - fully-pinned reproduction of the development machine (Linux/CUDA)
- `environment/real_world.txt` - additional pip requirements for the real-world GPU pipeline
- `third_party/audioclip/` - vendored AudioCLIP (real-world audio perception)
- `third_party/mdetr/` - vendored MDETR (real-world object detection)
- `third_party/{Grounded-Segment-Anything,gradslam,chamferdist,concept-graphs}/` - pinned git submodules (fetch with `git submodule update --init --recursive`)

## Source Ownership

- First-party code lives in `src/reflect/`
- Vendored research dependencies live in `third_party/`
- Generated outputs go to `$REFLECT_OUTPUT_ROOT`; only the lightweight curated results under `outputs/` are committed, and raw runtime artifacts are gitignored
