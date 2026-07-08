# Reproduction Additions

This document registers all changes and additions to the original REFLECT codebase.

## What Was Added

These code areas do not exist in upstream REFLECT as source modules:

| Added area | What it introduces | Current location |
|---|---|---|
| Structured plan models | Typed plan/action objects for direct executable LLM output | `src/reflect/models/action_primitive.py`, `src/reflect/models/plan.py` |
| Fast validation pipeline | In-memory plus cached episode processing for repeated evaluation | `src/reflect/pipelines/fast_validation.py` |
| Reasoning-effort evaluation | Batch sweeps for model-size and reasoning-effort comparison | `src/reflect/pipelines/reasoning_sweep.py` |
| Compatibility shims | Environment fixes for modern dependency stacks | `src/reflect/compat/hf.py`, `src/reflect/compat/open3d.py` |
| Real-world extension | Real-world scene-graph, prompting, batch validation, profiling, detector wrappers | `src/reflect/real_world/` |
| Open-world perception | YOLOE detection, segmentation, captioning, depth backprojection, 3D fusion, LLM edge labelling | `src/reflect/perception/open_world/` |
| Layered perception pipeline | Discovery / process / semantic perception layers with typed schemas and rerun.io streaming | `src/reflect/pipelines/perception/` |
| Episode and task primitives | Typed episode store and task-data models for episode handling | `src/reflect/core/episode_store.py`, `src/reflect/models/task_data.py` |
| Analysis package | Validation notebooks, error-taxonomy material, curated sweep outputs | `analysis/notebooks/`, `outputs/results/` |
| Calibration study (Experiment 16) | Failure-anchored MCQ engine over REFLECT ground truth plus an MMLU-Pro companion run; 1 - normalized-entropy confidence scored with ECE, Brier, and NLL | `analysis/notebooks/sim_uncertainty_calibration_{target,standard}.ipynb`, `outputs/analysis/llm_calibration_*/` |

## File Modifications

These upstream files still exist (at new locations), but the fork changes them substantially:

| Original upstream file | Current location | What changed | Why it matters |
|---|---|---|---|
| `LLM/prompt.py` | `src/reflect/llm/prompter.py` | Added `LocalLLMPrompter`, Ollama support, structured response parsing, `Plan`-based outputs, logprob scoring | Replanning is no longer limited to free-form text |
| `main/exp.py` | `src/reflect/pipelines/validation.py` | Reworked around explicit `data_path` / output-dir flow, memory-friendly helpers, plan generation hooks | Validation is easier to rerun and separate from repo-local artifacts |
| `main/execute_replan.py` | `src/reflect/sim/recovery.py` | Added support for normalizing legacy strings, dict plans, and typed plan objects before execution | Keeps old data compatible while enabling direct plan execution |
| `main/audio.py` | `src/reflect/perception/audio.py` | Rewritten around audio segments with canonical labels, scene-aware filtering, sidecar-audio support | Better reproduction alignment, avoids frame-level spillover |
| `main/get_local_sg.py` | `src/reflect/perception/local_graph.py` | Added masked world-coord computation, voxel-hash merging, cached device selection, Open3D compat | Reduces scene-graph cost and improves reproducibility |
| `main/utils.py` | `src/reflect/core/utils.py` | Added lazy SentenceTransformer loading, HF compat, KD-tree point-cloud distance, richer replanning prompt scaffolding | Fixes brittle upstream runtime assumptions |
| `main/task_utils.py` | `src/reflect/sim/task_manager.py` | Expanded failure-handling, navigation helpers, docstrings, null-safe task-state logic | Supports heavier failure injection and more robust replay |
| `main/gen_data.py` | `src/reflect/sim/data_gen.py` | Broadened failure injection logic and reproducibility handling | Better control over validation episode generation |
| `main/clip_utils.py` | `src/reflect/perception/clip.py` | Fixed single-candidate ranking behavior in CLIP/audio-text retrieval | Prevents shape-related ranking bugs during evaluation |

## Change Families

| Area | Upstream behavior | Fork behavior | Primary files |
|---|---|---|---|
| Direct structured replanning | Text-first: generate text, translate to admissible actions | `Plan`/`ActionPrimitive` models, structured responses, typed plan execution with legacy-string compat | `models/`, `llm/prompter.py`, `pipelines/validation.py`, `sim/recovery.py` |
| Faster repeated validation | Monolithic `exp.py`, recomputes expensive intermediates on rerun | In-memory execution plus persistent caches for scene graphs and summaries | `pipelines/fast_validation.py`, `pipelines/validation.py` |
| Scene-graph and perception optimizations | Original dependency stack, heavier per-step processing | Masked depth projection, voxel-hash merging, compat imports, lazy model loading, retrieval fixes | `perception/local_graph.py`, `core/utils.py`, `perception/clip.py`, `compat/` |
| Audio pipeline changes | Frame-level labeling from original simulation setup | Segment-based with canonical labels, scene-aware candidate filtering, sidecar-audio support | `perception/audio.py`, `third_party/audioclip/` |
| Failure injection and replay robustness | Original task path and failure logic | Extended failure selection, null-safe parsing, typed-plan replay, reset helpers | `sim/data_gen.py`, `sim/task_manager.py`, `sim/recovery.py`, `sim/actions.py` |
| Real-world extension | Simulation-only | Full real-world pipeline: scene-graph, AudioCLIP, MDETR detection, batch validation, GPU workers, profiling | `real_world/`, `third_party/` |
| Error taxonomy and traceability | No assessor-facing analysis | Notebooks and outputs for error analysis, annotation summaries, saved LLM traces | `analysis/`, `real_world/` |
| Model-size and reasoning-effort evaluation | No sweep harness | Batch sweep framework with curated JSON outputs | `pipelines/reasoning_sweep.py`, `outputs/results/` |

All file paths above are relative to `src/reflect/` unless otherwise noted.
