#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


from reflect.core.paths import prompts_dir, real_world_data_root, real_world_output_root, real_world_runtime_root, real_world_tasks_config

# Package root for locating third_party assets
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]  # src/reflect/cli -> project root


def _ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() and link_path.resolve() == target_path.resolve():
            return
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target_path)


def _prepare_runtime_workspace() -> Path:
    runtime_root = real_world_runtime_root()
    (runtime_root / "real_world").mkdir(parents=True, exist_ok=True)
    (runtime_root / "real_world" / "state_summary").mkdir(parents=True, exist_ok=True)
    (runtime_root / "real_world" / "scene").mkdir(parents=True, exist_ok=True)
    (runtime_root / "real_world" / "images").mkdir(parents=True, exist_ok=True)
    (runtime_root / "LLM").mkdir(parents=True, exist_ok=True)

    _ensure_symlink(runtime_root / "real_world" / "data", real_world_data_root())
    _ensure_symlink(runtime_root / "real_world" / "tasks_real_world.json", real_world_tasks_config())
    _ensure_symlink(runtime_root / "LLM" / "prompts-gpt4.json", prompts_dir() / "real_world_prompts_gpt4.json")
    _ensure_symlink(runtime_root / "LLM" / "prompts.json", prompts_dir() / "real_world_prompts_local.json")
    _ensure_symlink(runtime_root / "AudioCLIP", _PACKAGE_ROOT / "third_party" / "audioclip")
    return runtime_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the cleaned REFLECT real-world validation workflow.")
    parser.add_argument("--tasks", "--list", nargs="*")
    parser.add_argument("--folder_name", type=str, default="")
    parser.add_argument("--obj_det", type=str, default="mdetr")
    parser.add_argument("--audio_ver", type=int, default=1)
    parser.add_argument("--ablation_type", type=int, default=0)
    parser.add_argument("--mdetr_confidence_threshold", type=float, default=0.9)
    parser.add_argument("--force_rebuild_sg", action="store_true")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--artifact_level", type=str, default="final", choices=["final", "full"])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--task_workers", type=str, default="auto")
    parser.add_argument("--reasoning_workers", type=int, default=1)
    parser.add_argument("--outlier_filter_max_points", type=int, default=30000)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    runtime_root = _prepare_runtime_workspace()
    os.environ["REFLECT_REAL_WORLD_RUNTIME_ROOT"] = str(runtime_root)
    os.chdir(runtime_root)

    # Vendored MDETR resolves flat imports (`hubconf`, `transforms`); the runtime
    # workspace provides the `AudioCLIP` symlink used by reflect.real_world.prompting.
    sys.path.insert(0, str(_PACKAGE_ROOT / "third_party" / "mdetr"))
    sys.path.insert(0, str(runtime_root))

    from reflect.real_world.batch_validation import run_batch_validation  # noqa: E402

    with open(real_world_tasks_config(), "r") as fh:
        tasks_json = json.load(fh)

    run_batch_validation(args=args, tasks_json=tasks_json)
    print(f"[done] real-world outputs written under {real_world_output_root()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
