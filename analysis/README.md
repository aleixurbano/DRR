# `analysis/`

Research notebooks and supporting scripts produced during the DRR internship
(Fontys HTES). These are for interpretation and auditability - the runnable
workflows are the CLI commands (`reflect-sim`, `reflect-sweep`, `reflect-real`).
The artifacts they generate land under the top-level [`outputs/`](../outputs/) tree.

## Layout

- `notebooks/` - curated experiment and analysis notebooks (see index below)
- `scripts/` - batch drivers that support the notebooks
- `archive/` - preserved side experiments and the upstream REFLECT demo

All generated artifacts - notebook scratch, caches, and the curated result
artifacts (see inventory below) - live under the single top-level
[`outputs/`](../outputs/) tree. Lightweight results (CSVs, figures, reports, summary
JSON) are tracked in git; large regenerable artifacts (raw scene graphs, pickles,
caches) are gitignored. See [`outputs/README.md`](../outputs/README.md) for the full
layout.

## Setup

```bash
pip install -e ".[analysis]"      # pandas, matplotlib, ipywidgets, jupyter, ...
cp .env.example .env              # then fill in keys and data paths
```

Notebooks resolve all input and output locations through `reflect.core.paths`
(honoring `REFLECT_DATA_ROOT` / `REFLECT_OUTPUT_ROOT`) and read all API keys from
environment variables (`OPENAI_API_KEY`, `PORTKEY_API_KEY`). No paths are
hardcoded - experiment scratch goes to `analysis_experiment_dir(...)`, caches to
`analysis_cache_dir(...)`, and curated results to `sim_results_root()` /
`real_results_root()`.

## Notebook Index

Names follow `<domain>_<topic>.ipynb` with domains `sim` (AI2-THOR simulation),
`real` (real-robot episodes), `perception` (perception-pipeline experiments),
and `bench` (model benchmarks).

| Notebook | Purpose |
|---|---|
| `sim_validation.ipynb` | Full-dataset sim validation with multi-plan replanning; produces the `local_uncertainty` sweep outputs |
| `sim_error_taxonomy.ipynb` | Two-round error-taxonomy labelling over all failed simulated episodes |
| `sim_case_analysis_round1.ipynb` | Case-by-case analysis of failed episodes from early validation runs |
| `sim_case_analysis_round2.ipynb` | Extended case analysis: LLM impact, audio pipeline checks; produces the round-2 annotation summary |
| `sim_uncertainty_replanning.ipynb` | Uncertainty-aware multi-plan replanning experiments per episode |
| `sim_uncertainty_calibration_target.ipynb` | Experiment 16: confidence calibration on the failure-reasoning target set (1 - normalized-entropy confidence, ECE + Brier + NLL with difficulty-balanced weights, reliability, bootstrap) |
| `sim_uncertainty_calibration_standard.ipynb` | Companion calibration run on MMLU-Pro with the same confidence signal and metrics; includes the cross-notebook rank-reversal check against the target set |
| `sim_llm_world_tracking.ipynb` | Exploration of the LLM world-state tracking failure mode |
| `real_validation.ipynb` | Validation, EDA, and gap analysis of REFLECT claims on the real-world dataset |
| `real_error_taxonomy.ipynb` | Error-taxonomy triage and review for real-world episodes |
| `perception_conceptgraphs_real.ipynb` | ConceptGraphs open-vocabulary 3D scene-understanding pipeline on real robot recordings (needs external `concept-graphs` and `Grounded-Segment-Anything` checkouts; see notebook header) |
| `perception_yoloe_real.ipynb` | YOLOE-based perception pipeline on real episodes with rerun.io streaming |
| `bench_vlm_captioner.ipynb` | VLM captioner comparison on YOLOE+SAM crops (sim + Robo2VLM real frames) |
| `bench_llm_scene_uncertainty.ipynb` | Can candidate LLMs read the scene, and is their uncertainty trustworthy? |

Archived: `archive/upstream_reflect_demo.ipynb` is the original upstream REFLECT
demo notebook. It targets the upstream flat code layout and is kept for reference
only - it does not run against this package.

### Former names

Notebooks were renamed for the public release; older reports may reference:

| Old name | Current name |
|---|---|
| `demo_upstream.ipynb` | `archive/upstream_reflect_demo.ipynb` |
| `captioner_benchmark.ipynb` | `bench_vlm_captioner.ipynb` |
| `model_test_capabilities.ipynb` | `bench_llm_scene_uncertainty.ipynb` |
| `perception_pipeline_visualization.ipynb` | `perception_conceptgraphs_real.ipynb` |
| `perception_pipeline_yoloe_new.ipynb` | `perception_yoloe_real.ipynb` |
| `real_error_taxonomy_analysis.ipynb` | `real_error_taxonomy.ipynb` |
| `sim_error_analysis_all_simulated.ipynb` | `sim_error_taxonomy.ipynb` |
| `sim_individual_analysis.ipynb` | `sim_case_analysis_round1.ipynb` |
| `sim_individual_analysis_round2.ipynb` | `sim_case_analysis_round2.ipynb` |
| `sim_individual_analysis_uncertainty.ipynb` | `sim_uncertainty_replanning.ipynb` |
| `sim_individual_llm_world_tracking_exploration.ipynb` | `sim_llm_world_tracking.ipynb` |
| `sim_uncertainty_calibration_experiment_16.ipynb` | `sim_uncertainty_calibration_target.ipynb` |

## Results Inventory

These curated artifacts live under `outputs/results/` (tracked in git; resolve via
`sim_results_root()` / `real_results_root()`).

### `outputs/results/sim_data/`

| Artifact | What it is |
|---|---|
| `reflect_results_run{1..4}.json` | Per-episode pipeline outputs (predicted failure reason/step, summaries, traces) for validation runs 1-4 |
| `exp_evaluation_run{1..4}.csv` | Human evaluation of runs 1-4 (`human_score` against ground-truth failure reasons) |
| `gpt4_exp_evaluation_run1.csv`, `gpt4_subsample_results_run{1,2}.json` | GPT-4 subsample comparison runs and their evaluation |
| `error_taxonomy_full.csv` | Error-taxonomy labels over all annotated episodes (compact schema) |
| `error_taxonomy_all.csv`, `error_taxonomy_all_run4.csv` | Extended taxonomy annotations (predictions, localisation/explanation scores, plans); `_run4` is joined to run-4 results |
| `custom_taxonomy_codes.json` | Definitions of the custom taxonomy codes used above |
| `sim_case_analysis_round2_annotation_summary.{csv,md}` | Annotation summary produced by `sim_case_analysis_round2.ipynb` |
| `reasoning_effort_sweep/` | `reflect-sweep` outputs for `gpt-5.4` across reasoning efforts (`none`-`xhigh`) plus their evaluation CSV |

### `outputs/results/real_world/`

| Artifact | What it is |
|---|---|
| `final_results_with_explanations.csv` | Final real-world pipeline results with failure explanations per task |
| `failure_explanation_evaluation.csv` | Human scores of the predicted failure explanations |
| `error_taxonomy_real_round1.csv` | Round-1 taxonomy triage of real-world episodes |

Provenance: simulated artifacts were generated by the validation pipeline and the
`sim_*` notebooks above against the REFLECT simulation dataset; real-world artifacts
by the `reflect-real` pipeline and `real_*` notebooks against the REFLECT real-robot
recordings. See `docs/data_and_artifacts.md` for where to obtain the datasets.
