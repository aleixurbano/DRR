import copy
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from reflect.core.paths import real_world_tasks_config

from reflect.real_world.logging_utils import configure_logging, get_logger
from reflect.real_world.prompting import (
    _summary_dir,
    config_parser,
    run_real_world_pipeline,
    run_real_world_reasoning_only,
)


logger = get_logger(__name__)
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))


def _runtime_root():
    return os.environ.get("REFLECT_REAL_WORLD_RUNTIME_ROOT", os.getcwd())


def _namespace_to_dict(args):
    return {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }


def resolve_task_keys(tasks_json, requested_tasks):
    if not requested_tasks:
        return list(tasks_json.keys())

    resolved = []
    for raw_task in requested_tasks:
        if str(raw_task) == "0":
            return list(tasks_json.keys())
        if str(raw_task).startswith("Task "):
            resolved.append(str(raw_task))
        else:
            resolved.append(f"Task {int(raw_task)}")
    return resolved


def resolve_task_workers(requested_workers, task_count):
    if task_count <= 1:
        return 1

    if str(requested_workers).lower() == "auto":
        if torch.cuda.is_available():
            return max(1, min(task_count, torch.cuda.device_count()))
        cpu_count = os.cpu_count() or 1
        return max(1, min(task_count, min(4, max(1, cpu_count // 4))))

    return max(1, min(task_count, int(requested_workers)))


def _build_args_from_dict(args_dict):
    args = config_parser().parse_args([])
    for key, value in args_dict.items():
        setattr(args, key, value)
    return args


def _phase1_worker(task_key, task_info, args_dict, runtime_root):
    os.chdir(runtime_root)
    args = _build_args_from_dict(args_dict)
    timings = run_real_world_pipeline(args, task_info, include_reasoning=False)
    return {
        "task_key": task_key,
        "folder_name": task_info["general_folder_name"],
        "timings": timings,
    }


def _load_task_timings(folder_name):
    timings_path = os.path.join(_runtime_root(), _summary_dir(folder_name), "timings.json")
    if not os.path.exists(timings_path):
        return None
    with open(timings_path, "r") as f:
        return json.load(f)


def build_batch_profile(ordered_tasks, task_workers, args):
    task_profiles = {}
    aggregate_stage_totals = {}
    slowest_tasks = []

    for task_key, task_info in ordered_tasks:
        folder_name = task_info["general_folder_name"]
        task_profile = _load_task_timings(folder_name)
        if task_profile is None:
            continue
        task_profiles[folder_name] = task_profile
        slowest_tasks.append(
            {
                "task_key": task_key,
                "folder_name": folder_name,
                "total_wall_time_sec": task_profile.get("total_wall_time_sec", 0.0),
            }
        )
        for stage_name, duration in task_profile.get("stage_totals_sec", {}).items():
            aggregate_stage_totals[stage_name] = aggregate_stage_totals.get(stage_name, 0.0) + float(duration)

    slowest_tasks.sort(key=lambda item: item["total_wall_time_sec"], reverse=True)
    slowest_stages = [
        {"stage": stage_name, "total_sec": round(duration, 6)}
        for stage_name, duration in sorted(aggregate_stage_totals.items(), key=lambda item: item[1], reverse=True)
    ]

    summary = {
        "task_workers": task_workers,
        "reasoning_workers": getattr(args, "reasoning_workers", 1),
        "artifact_level": getattr(args, "artifact_level", "final"),
        "force_rebuild_sg": bool(getattr(args, "force_rebuild_sg", False)),
        "task_count": len(ordered_tasks),
        "task_order": [task_key for task_key, _ in ordered_tasks],
        "slowest_tasks": slowest_tasks,
        "slowest_stages": slowest_stages,
        "tasks": task_profiles,
    }

    out_path = os.path.join(_runtime_root(), "real_world", "state_summary", "batch_timings.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_batch_validation(args=None, task_infos=None, task_keys=None, tasks_json=None):
    os.chdir(_runtime_root())
    configure_logging(getattr(args, "log_level", "INFO") if args is not None else "INFO")

    if args is None:
        parser = config_parser()
        parser.set_defaults(force_rebuild_sg=True, artifact_level="final", profile=True, task_workers="auto", reasoning_workers=1)
        args = parser.parse_args([])

    if getattr(args, "reasoning_workers", 1) != 1:
        raise ValueError("Only reasoning_workers=1 is supported to preserve reasoning order and reproducibility.")

    if tasks_json is None:
        with open(real_world_tasks_config(), "r") as f:
            tasks_json = json.load(f)

    if task_infos is None:
        resolved_task_keys = resolve_task_keys(tasks_json, getattr(args, "tasks", None))
        ordered_tasks = [(task_key, tasks_json[task_key]) for task_key in resolved_task_keys]
    else:
        ordered_tasks = []
        for idx, task_info in enumerate(task_infos):
            task_key = task_keys[idx] if task_keys is not None else f"Task {task_info['task_idx']}"
            ordered_tasks.append((task_key, task_info))

    resolved_task_workers = resolve_task_workers(getattr(args, "task_workers", "auto"), len(ordered_tasks))
    logger.info(
        "[Batch] tasks=%s task_workers=%s reasoning_workers=%s artifact_level=%s",
        len(ordered_tasks),
        resolved_task_workers,
        getattr(args, "reasoning_workers", 1),
        getattr(args, "artifact_level", "final"),
    )

    phase1_args = copy.deepcopy(args)
    phase1_args.force_rebuild_sg = bool(getattr(args, "force_rebuild_sg", False))
    phase1_args_dict = _namespace_to_dict(phase1_args)

    if resolved_task_workers == 1:
        for task_key, task_info in ordered_tasks:
            _phase1_worker(task_key, task_info, phase1_args_dict, _runtime_root())
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=resolved_task_workers, mp_context=ctx) as executor:
            futures = [
                executor.submit(_phase1_worker, task_key, task_info, phase1_args_dict, _runtime_root())
                for task_key, task_info in ordered_tasks
            ]
            for future in as_completed(futures):
                result = future.result()
                logger.info("[Batch] phase1 completed %s", result["folder_name"])

    phase2_args = copy.deepcopy(args)
    phase2_args.force_rebuild_sg = False
    for task_key, task_info in ordered_tasks:
        run_real_world_reasoning_only(phase2_args, task_info)
        logger.info("[Batch] reasoning completed %s", task_info["general_folder_name"])

    summary = build_batch_profile(ordered_tasks, resolved_task_workers, args)
    logger.info("[Batch] summary written to real_world/state_summary/batch_timings.json")
    return summary


def main():
    parser = config_parser()
    parser.set_defaults(force_rebuild_sg=True, artifact_level="final", profile=True, task_workers="auto", reasoning_workers=1)
    args = parser.parse_args()
    run_batch_validation(args=args)


if __name__ == "__main__":
    main()
