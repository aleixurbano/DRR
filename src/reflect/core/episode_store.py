from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class EpisodeArtifactStore:
    data_path: Path
    output_dir: Path
    llm_dir: Path | None = None

    @property
    def task_path(self) -> Path:
        return self.data_path / "task.json"

    @property
    def local_graph_dir(self) -> Path:
        return self.output_dir / "local_graphs"

    @property
    def global_scene_graph_path(self) -> Path:
        return self.output_dir / "global_sg.pkl"

    @property
    def key_frames_path(self) -> Path:
        return self.output_dir / "L1_key_frames.txt"

    @property
    def summary_l1_path(self) -> Path:
        return self.output_dir / "state_summary_L1.txt"

    @property
    def summary_l2_path(self) -> Path:
        return self.output_dir / "state_summary_L2.txt"

    @property
    def reasoning_path(self) -> Path:
        return self.output_dir / "reasoning.json"

    @property
    def replan_path(self) -> Path:
        return self.output_dir / "replan.json"

    @property
    def episode_id(self) -> str:
        # Always "/" so ids are stable across platforms (used as keys and llm_dir subpaths).
        return f"{self.data_path.parent.name}/{self.data_path.name}"

    def ensure_local_graph_dir(self) -> Path:
        self.local_graph_dir.mkdir(parents=True, exist_ok=True)
        return self.local_graph_dir

    def ensure_llm_dir(self) -> Path:
        if self.llm_dir is None:
            raise ValueError("llm_dir is not configured for this episode store")
        self.llm_dir.mkdir(parents=True, exist_ok=True)
        return self.llm_dir

    def ensure_episode_llm_dir(self) -> Path:
        if self.llm_dir is None:
            raise ValueError("llm_dir is not configured for this episode store")
        episode_dir = self.llm_dir / self.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        return episode_dir

    def load_task(self) -> dict[str, Any]:
        with open(self.task_path, "r") as fh:
            return json.load(fh)

    def load_pickle(self, path: Path) -> Any:
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def save_pickle(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(value, fh)

    def load_json(self, path: Path, default: Any = None) -> Any:
        if not path.exists():
            return default
        with open(path, "r") as fh:
            return json.load(fh)

    def save_json(self, path: Path, value: Any, *, indent: int | None = 2) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(value, fh, indent=indent, default=str)

    def load_text(self, path: Path) -> str:
        if not path.exists():
            return ""
        with open(path, "r") as fh:
            return fh.read()

    def save_text(self, path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(value)

    def save_key_frames(self, key_frames: Iterable[int]) -> None:
        self.save_text(self.key_frames_path, "".join(f"{frame}\n" for frame in key_frames))

    def load_key_frames(self) -> list[int]:
        if not self.key_frames_path.exists():
            return []
        with open(self.key_frames_path, "r") as fh:
            return [int(line.strip()) for line in fh if line.strip()]
