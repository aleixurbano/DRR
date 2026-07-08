from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional


# Project root: three levels up from src/reflect/core/
REPO_ROOT = Path(__file__).resolve().parents[3]

# Package data root for configs bundled inside the package
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _resolve_env_path(env_name: str, default: Path) -> Path:
    raw_value = os.environ.get(env_name)
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return default.resolve()


def config_dir() -> Path:
    return _PACKAGE_ROOT / "configs"


def prompts_dir() -> Path:
    return config_dir() / "prompts"


def sim_tasks_config() -> Path:
    return config_dir() / "tasks_sim.json"


def real_world_tasks_config() -> Path:
    return config_dir() / "tasks_real_world.json"


def data_root() -> Path:
    return _resolve_env_path("REFLECT_DATA_ROOT", REPO_ROOT / "sample_data")



def output_root() -> Path:
    return _resolve_env_path("REFLECT_OUTPUT_ROOT", REPO_ROOT / "outputs")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slugify_component(value: str) -> str:
    text = str(value).strip().lower()
    pieces: list[str] = []
    for char in text:
        if char.isalnum():
            pieces.append(char)
        elif char in {" ", "/", "\\", ".", ":", "_", "-"}:
            pieces.append("_")
    slug = "".join(pieces).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unknown"


def model_slug(model: str) -> str:
    return _slugify_component(model)


def effort_slug(effort: Optional[str]) -> str:
    return "none" if effort in (None, "", "default") else _slugify_component(effort)


@dataclass(frozen=True)
class EpisodeRef:
    task_name: str
    episode_name: str

    @property
    def identifier(self) -> str:
        return f"{self.task_name}__{self.episode_name}"


@dataclass(frozen=True)
class ValidationEpisodeArtifacts:
    episode: EpisodeRef

    @property
    def summary_dir(self) -> Path:
        return sim_output_root() / "state_summary" / self.episode.task_name / self.episode.episode_name

    @property
    def result_path(self) -> Path:
        return self.summary_dir / "validation_result.json"

    @property
    def trace_dir(self) -> Path:
        return analysis_output_root() / "llm_sim_validation_traces" / self.episode.task_name / self.episode.episode_name

    @property
    def trace_path(self) -> Path:
        return self.trace_dir / "llm_trace.json"


def validation_episode(task_name: str, episode_name: str) -> EpisodeRef:
    return EpisodeRef(task_name=task_name, episode_name=episode_name)


def validation_artifacts(task_name: str, episode_name: str) -> ValidationEpisodeArtifacts:
    return ValidationEpisodeArtifacts(validation_episode(task_name, episode_name))


def sim_validation_run_summary_path() -> Path:
    return ensure_dir(sim_output_root()) / "run_summary.json"


def sim_data_root() -> Path:
    return data_root() / "sim_data"



def sim_data_root_legacy() -> Path:
    return data_root_legacy() / "sim_data"


def real_world_data_root() -> Path:
    return data_root() / "reflect_dataset" / "real_data"


def sim_output_root() -> Path:
    return ensure_dir(output_root() / "sim_validation")


def reasoning_sweep_root() -> Path:
    return ensure_dir(output_root() / "reasoning_sweep")


def real_world_output_root() -> Path:
    return ensure_dir(output_root() / "real_world")


def analysis_output_root() -> Path:
    """Scratch root for notebook/script outputs (was ``analysis/outputs``)."""
    return ensure_dir(output_root() / "analysis")


def analysis_experiment_dir(name: str) -> Path:
    """Per-experiment scratch dir, e.g. ``exp16`` or ``model_capability``."""
    return ensure_dir(analysis_output_root() / name)


def analysis_runs_root() -> Path:
    """Timestamped per-run scratch dirs produced by notebooks."""
    return ensure_dir(analysis_output_root() / "runs")


def local_uncertainty_root() -> Path:
    """Local-uncertainty sim-validation sweep outputs (sim_validation notebook + CLI)."""
    return ensure_dir(analysis_output_root() / "llm_local_uncertainty_measurement")


def analysis_cache_dir(name: str) -> Path:
    """Named cache dir, e.g. ``edge_llm_cache`` or ``robo2vlm_cache``."""
    return ensure_dir(analysis_output_root() / "caches" / name)


def results_root() -> Path:
    """Curated result artifacts root."""
    return ensure_dir(output_root() / "results")


def sim_results_root() -> Path:
    return ensure_dir(results_root() / "sim_data")


def real_results_root() -> Path:
    return ensure_dir(results_root() / "real_world")


def sim_episode_summary_dir(task_name: str, episode_name: str) -> Path:
    return ensure_dir(validation_artifacts(task_name, episode_name).summary_dir)


def sim_validation_result_path(task_name: str, episode_name: str) -> Path:
    return validation_artifacts(task_name, episode_name).result_path


def sim_validation_trace_dir(task_name: str, episode_name: str) -> Path:
    return ensure_dir(validation_artifacts(task_name, episode_name).trace_dir)


def sim_validation_trace_path(task_name: str, episode_name: str) -> Path:
    return validation_artifacts(task_name, episode_name).trace_path


def reasoning_sweep_output_dir(results_root: Path | str) -> Path:
    return ensure_dir(Path(results_root) / "reasoning_effort_sweep")


def reasoning_sweep_result_path(results_root: Path | str, model: str, effort: Optional[str]) -> Path:
    return reasoning_sweep_output_dir(results_root) / f"reasoning_effort__{model_slug(model)}__{effort_slug(effort)}.json"


def real_world_runtime_root() -> Path:
    return ensure_dir(real_world_output_root() / "runtime")
