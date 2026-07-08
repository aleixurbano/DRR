#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


from reflect.core.paths import (
    prompts_dir,
    local_uncertainty_root,
    sim_data_root,
    sim_episode_summary_dir,
    sim_validation_run_summary_path,
    sim_validation_trace_path,
    validation_episode,
)


def _dependency_guidance(exc: ModuleNotFoundError) -> str:
    guidance = [
        f"Missing Python dependency: {exc.name}.",
        f"Active Python: {sys.executable}",
        "Install the missing simulation dependencies and rerun the command.",
    ]
    return " ".join(guidance)


def _split_csv(raw_value: str | None) -> list[str] | None:
    if not raw_value:
        return None
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _discover_episodes(task_filters: list[str] | None, episode_filters: list[str] | None) -> list[tuple[str, str, Path]]:
    root = sim_data_root()
    if not root.exists():
        raise FileNotFoundError(f"Simulation dataset root not found: {root}")

    task_dirs = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    if task_filters:
        task_dirs = [path for path in task_dirs if path.name in set(task_filters)]

    episodes: list[tuple[str, str, Path]] = []
    for task_dir in task_dirs:
        episode_dirs = sorted(path for path in task_dir.iterdir() if path.is_dir())
        for episode_dir in episode_dirs:
            if episode_filters and episode_dir.name not in set(episode_filters):
                continue
            episodes.append((task_dir.name, episode_dir.name, episode_dir))
    return episodes


def _build_prompter(args: argparse.Namespace):
    from reflect.llm.prompter import LocalLLMPrompter, OpenAILLMPrompter

    if args.backend == "local":
        return LocalLLMPrompter(model_name=args.model, base_url=args.ollama_base_url, think=args.ollama_think)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"{args.api_key_env} is required for --backend=openai.")
    return OpenAILLMPrompter(api_key=api_key, model=args.model, reasoning_effort=args.reasoning_effort)


def _run_episode(item):
    (
        episode_index,
        data_path,
        task_name,
        episode_name,
        prompt_template,
        with_audio,
        prompter_config,
        two_pass_replan,
    ) = item

    from reflect.llm.prompter import LocalLLMPrompter, OpenAILLMPrompter
    from reflect.pipelines.fast_validation import EpisodeConfig, process_episode

    if prompter_config["backend"] == "local":
        prompter = LocalLLMPrompter(
            model_name=prompter_config["model"],
            base_url=prompter_config["ollama_base_url"],
            think=prompter_config["ollama_think"],
        )
    else:
        prompter = OpenAILLMPrompter(
            api_key=prompter_config["api_key"],
            model=prompter_config["model"],
            reasoning_effort=prompter_config["reasoning_effort"],
        )

    cfg = EpisodeConfig(
        data_path=data_path,
        task_name=task_name,
        episode_name=episode_name,
        llm_prompter=prompter,
        prompt_template=prompt_template,
        with_audio=with_audio,
        two_pass_replan=two_pass_replan,
    )
    return episode_index, process_episode(cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the cleaned REFLECT simulation validation workflow.")
    parser.add_argument("--tasks", help="Comma-separated task filters, e.g. boilWater,makeCoffee")
    parser.add_argument("--episodes", help="Comma-separated episode filters, e.g. boilWater-1,makeCoffee-3")
    parser.add_argument("--max-episodes", type=int, default=0, help="Optional cap on how many discovered episodes to run.")
    parser.add_argument("--with-audio", type=int, default=1, choices=[0, 1], help="Use detected audio (1) or task.json sounds (0).")
    parser.add_argument("--backend", choices=["local", "openai"], default="local")
    parser.add_argument("--model", default="qwen3.5:9b", help="Model tag for the selected backend.")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--ollama-think", action="store_true")
    parser.add_argument("--reasoning-effort", default=None, help="OpenAI reasoning effort (none/low/medium/high). Default: none for GPT-5 family.")
    parser.add_argument("--two-pass-replan", action="store_true", help="Use two-pass propose/select replan with entropy tracking.")
    parser.add_argument("--regenerate-artifacts", action="store_true", help="Regenerate missing AI2-THOR episode artifacts before validation to enable co-plan execution.")
    parser.add_argument("--max-workers", type=int, default=max(1, min(2, os.cpu_count() or 1)))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()

    prompt_path = prompts_dir() / "sim_prompts.json"
    with open(prompt_path, "r") as fh:
        prompt_template = json.load(fh)

    episodes = _discover_episodes(_split_csv(args.tasks), _split_csv(args.episodes))
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    if not episodes:
        raise SystemExit("No simulation episodes matched the requested filters.")

    # ── Regenerate missing raw artifacts for co-plan execution ─────────────
    if args.regenerate_artifacts:
        _regenerate_missing_artifacts(episodes)

    try:
        from reflect.pipelines.fast_validation import process_episode  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(_dependency_guidance(exc)) from exc

    prompter_config = {
        "backend": args.backend,
        "model": args.model,
        "ollama_base_url": args.ollama_base_url,
        "ollama_think": args.ollama_think,
        "reasoning_effort": args.reasoning_effort,
        "api_key": os.environ.get(args.api_key_env),
    }
    batch_results = []
    work_items = []
    for episode_index, (task_name, episode_name, episode_dir) in enumerate(episodes):
        print(f"[run] {task_name}/{episode_name}", flush=True)
        work_items.append(
            (
                episode_index,
                str(episode_dir),
                task_name,
                episode_name,
                prompt_template,
                args.with_audio,
                prompter_config,
                args.two_pass_replan,
            )
        )

    total_episodes = len(work_items)
    indexed_results = [None] * total_episodes
    completed = 0
    if args.max_workers == 1:
        for item in work_items:
            index, result = _run_episode(item)
            indexed_results[index] = result
            ep_task, ep_name, _ = episodes[index]
            completed += 1
            status = result.get("status", "?") if result else "error"
            print(f"[run] completed {ep_task}/{ep_name} - {completed}/{total_episodes} done (status={status})", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            future_map = {pool.submit(_run_episode, item): item[0] for item in work_items}
            for future in as_completed(future_map):
                try:
                    index, result = future.result()
                    indexed_results[index] = result
                    ep_task, ep_name, _ = episodes[index]
                    completed += 1
                    status = result.get("status", "?") if result else "error"
                    print(f"[run] completed {ep_task}/{ep_name} - {completed}/{total_episodes} done (status={status})", flush=True)
                except Exception as exc:
                    import traceback
                    print(f"[ERROR] Worker raised an exception: {exc}", flush=True)
                    traceback.print_exc()

    for (task_name, episode_name, episode_dir), result in zip(episodes, indexed_results):
        if result is None:
            try:
                print(f"[ERROR] No result for {task_name}/{episode_name} (worker may have crashed)", flush=True)
            except (ValueError, OSError):
                pass
            batch_results.append({
                "task": task_name, "episode": episode_name,
                "status": "error", "artifact_mode": None,
                "replay_available": False, "summary_dir": "",
            })
            continue
        episode_ref = validation_episode(task_name, episode_name)
        summary_dir = sim_episode_summary_dir(task_name, episode_name)
        summary_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_dir / "validation_result.json", "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        prompts_log = result.get("prompts_log", [])
        if prompts_log:
            trace_path = sim_validation_trace_path(task_name, episode_name)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with open(trace_path, "w") as fh:
                json.dump(prompts_log, fh, indent=2, default=str)
        batch_results.append(
            {
                "task": task_name,
                "episode": episode_name,
                "episode_id": episode_ref.identifier,
                "status": result.get("status"),
                "artifact_mode": result.get("artifact_mode"),
                "replay_available": result.get("replay_available"),
                "summary_dir": str(summary_dir),
            }
        )

    batch_out = sim_validation_run_summary_path()
    with open(batch_out, "w") as fh:
        json.dump(batch_results, fh, indent=2)

    # AI2-THOR Unity subprocesses can close our stdout fd; reopen if needed.
    if sys.stdout.closed:
        sys.stdout = open("/dev/stdout", "w")

    ok_count = sum(1 for row in batch_results if row["status"] == "ok")
    error_count = sum(1 for row in batch_results if row["status"] == "error")
    _msg = (f"[done] {ok_count}/{len(batch_results)} episodes task-success, "
            f"{len(batch_results) - ok_count - error_count} with detected failures, "
            f"{error_count} processing errors.")
    try:
        print(_msg, flush=True)
        print(f"[saved] {batch_out}", flush=True)
    except (ValueError, OSError):
        sys.stderr.write(_msg + "\n")
        sys.stderr.write(f"[saved] {batch_out}\n")

    # ── Export analysis CSVs for the notebook ──────────────────────────────
    try:
        _export_analysis_csvs(episodes, indexed_results, args.model)
    except Exception as exc:
        sys.stderr.write(f"[WARN] CSV export failed: {exc}\n")

    # Exit 0 as long as no processing errors - task failures are expected
    # for failure-injection episodes and are the subject of analysis.
    return 1 if error_count > 0 else 0


_GT_FAILURE_REASON_TO_CHOSEN = {
    "drop": "drop",
    "dropped": "drop",
    "missing": "missing_step",
    "failed to successfully execute": "failed_action",
    "failed": "failed_action",
    "blocking": "blocking",
    "occupied": "occupied",
    "wrong perception": "wrong perception",
}


def _infer_chosen_failure(gt_failure_reason: str) -> str | None:
    """Map a gt_failure_reason string to a chosen_failure value for data gen."""
    reason_lower = gt_failure_reason.lower()
    for keyword, failure_type in _GT_FAILURE_REASON_TO_CHOSEN.items():
        if keyword in reason_lower:
            return failure_type
    return None


def _validate_episode_artifacts(episode_dir: Path) -> tuple[bool, str]:
    """Return (is_valid, reason) for existing raw artifacts in *episode_dir*.

    Checks beyond mere file presence:
    - All pickle files are non-zero in size.
    - interact_actions.pickle and nav_actions.pickle can be loaded as dicts.
    - The first event pickle can be loaded without error.
    - The event count is >= the number of actions recorded in task.json.
    """
    import pickle

    data_path = str(episode_dir)
    events_dir = episode_dir / "events"
    interact_path = episode_dir / "interact_actions.pickle"
    nav_path = episode_dir / "nav_actions.pickle"

    # Basic size checks
    for p in [interact_path, nav_path]:
        if p.stat().st_size == 0:
            return False, f"{p.name} is zero bytes"

    event_pickles = sorted(events_dir.glob("*.pickle"))

    if not event_pickles:
        return False, "events/ dir is empty"

    if event_pickles[0].stat().st_size == 0:
        return False, f"{event_pickles[0].name} is zero bytes"

    # Loadability checks
    try:
        with open(interact_path, "rb") as fh:
            interact_data = pickle.load(fh)
        if not isinstance(interact_data, dict):
            return False, "interact_actions.pickle is not a dict"
    except Exception as exc:
        return False, f"interact_actions.pickle unreadable: {exc}"

    try:
        with open(nav_path, "rb") as fh:
            nav_data = pickle.load(fh)
        if not isinstance(nav_data, dict):
            return False, "nav_actions.pickle is not a dict"
    except Exception as exc:
        return False, f"nav_actions.pickle unreadable: {exc}"

    try:
        with open(event_pickles[0], "rb") as fh:
            pickle.load(fh)
    except Exception as exc:
        return False, f"{event_pickles[0].name} unreadable: {exc}"

    # Count check: event pickles should be >= number of actions in task.json
    task_json = episode_dir / "task.json"
    if task_json.exists():
        try:
            with open(task_json) as fh:
                task_meta = json.load(fh)
            expected_min = len(task_meta.get("actions", []))
            if len(event_pickles) < expected_min:
                return False, (
                    f"only {len(event_pickles)} event pickles "
                    f"but task has {expected_min} actions (truncated run)"
                )
        except Exception:
            pass  # if task.json is unreadable don't fail validation

    return True, "ok"


def _regenerate_missing_artifacts(episodes: list[tuple[str, str, Path]]):
    """Regenerate raw AI2-THOR artifacts for episodes that are missing or invalid.

    Uses run_data_gen() with a symlink trick: ``thor_tasks/`` → ``sim_data/``
    so that the generated files land in the correct episode directories.
    TaskUtil._get_folder_name is temporarily patched to prevent numeric suffixes.
    """
    from reflect.pipelines.fast_validation import _has_raw_episode_artifacts

    needs_regen = []
    print(f"[datagen] Checking {len(episodes)} episodes for valid raw artifacts...")
    for task_name, episode_name, episode_dir in episodes:
        if not _has_raw_episode_artifacts(str(episode_dir)):
            needs_regen.append((task_name, episode_name, episode_dir, "missing"))
            print(f"[datagen]   {task_name}/{episode_name}: MISSING")
        else:
            valid, reason = _validate_episode_artifacts(episode_dir)
            if not valid:
                needs_regen.append((task_name, episode_name, episode_dir, reason))
                print(f"[datagen]   {task_name}/{episode_name}: INVALID - {reason}")
            else:
                print(f"[datagen]   {task_name}/{episode_name}: ok")

    if not needs_regen:
        print("[datagen] All episodes have valid raw artifacts. Skipping regeneration.")
        return

    missing_count = sum(1 for *_, r in needs_regen if r == "missing")
    invalid_count = len(needs_regen) - missing_count
    print(
        f"[datagen] {len(needs_regen)}/{len(episodes)} episodes need regeneration "
        f"({missing_count} missing, {invalid_count} invalid). Regenerating..."
    )
    # unwrap to the original 3-tuple for the loop below
    missing = [(t, e, d) for t, e, d, _reason in needs_regen]

    from reflect.core.constants import TASK_DICT
    from reflect.sim.data_gen import run_data_gen
    from reflect.sim.task_manager import TaskUtil

    name_to_idx = {v: k for k, v in TASK_DICT.items() if v}

    # run_data_gen writes to both {repo_path}/thor_tasks/... (via save_data) and
    # CWD-relative thor_tasks/... (for action pickles, task.json).
    # We create a symlink thor_tasks → sim_data so both paths resolve to the
    # episode directory under sim_data/.
    sim_data_dir = missing[0][2].parent.parent          # .../sim_data
    datasets_root = sim_data_dir.parent                 # .../datasets
    thor_link = datasets_root / "thor_tasks"

    created_link = False
    if not thor_link.exists():
        os.symlink(sim_data_dir, thor_link)
        created_link = True
    elif not thor_link.is_symlink() or thor_link.resolve() != sim_data_dir.resolve():
        print(f"[datagen] WARNING: {thor_link} exists but doesn't point to {sim_data_dir}. "
              "Cannot regenerate artifacts.")
        return

    saved_cwd = os.getcwd()
    os.chdir(str(datasets_root))

    # Patch _get_folder_name to return the folder_name as-is (no "-N" suffix).
    # run_data_gen passes folder_name="taskName/episodeName" to TaskUtil.
    # Without the patch, TaskUtil appends "-1" → "taskName/episodeName-1".
    original_get_folder = TaskUtil._get_folder_name
    TaskUtil._get_folder_name = lambda self, name, idx: name

    try:
        for task_name, episode_name, episode_dir in missing:
            task_json_path = episode_dir / "task.json"
            if not task_json_path.exists():
                print(f"[datagen] SKIP {task_name}/{episode_name}: no task.json")
                continue

            with open(task_json_path) as fh:
                task_meta = json.load(fh)

            task_idx = task_meta.get("task_idx", name_to_idx.get(task_name))
            if task_idx is None:
                print(f"[datagen] SKIP {task_name}/{episode_name}: unknown task_idx")
                continue

            chosen_failure = _infer_chosen_failure(task_meta.get("gt_failure_reason", ""))

            # failure_injection is stored as the string "True" in some task.json files
            fi_raw = task_meta.get("failure_injection", True)
            failure_injection = str(fi_raw).lower() in ("true", "1", "yes")

            if chosen_failure in ("blocking", "occupied", "wrong perception") \
               and "failure_injection_params" not in task_meta:
                print(f"[datagen] SKIP {task_name}/{episode_name}: "
                      f"spatial failure '{chosen_failure}' but no failure_injection_params")
                continue

            gen_task = {
                "task_idx": task_idx,
                "folder_name": episode_name,
                "num_samples": 1,
                "scene": task_meta["scene"],
                "actions": task_meta["actions"],
                "failure_injection": failure_injection,
                "chosen_failure": chosen_failure,
            }
            if task_meta.get("specified_missing_steps"):
                gen_task["specified_missing_steps"] = task_meta["specified_missing_steps"]
            if "failure_injection_params" in task_meta:
                gen_task["failure_injection_params"] = task_meta["failure_injection_params"]
            if "preactions" in task_meta:
                gen_task["preactions"] = task_meta["preactions"]

            print(f"[datagen] Regenerating {task_name}/{episode_name} "
                  f"(scene={task_meta['scene']}, failure={chosen_failure})...")
            try:
                run_data_gen(data_path=str(datasets_root), task=gen_task)
                # Verify artifacts were created
                if _has_raw_episode_artifacts(str(episode_dir)):
                    print(f"[datagen] OK {task_name}/{episode_name}")
                else:
                    print(f"[datagen] WARNING {task_name}/{episode_name}: "
                          "run_data_gen completed but artifacts still missing")
            except Exception as exc:
                import traceback
                print(f"[datagen] FAILED {task_name}/{episode_name}: {exc}")
                traceback.print_exc()
    finally:
        TaskUtil._get_folder_name = original_get_folder
        os.chdir(saved_cwd)
        if created_link and thor_link.is_symlink():
            thor_link.unlink()


def _export_analysis_csvs(episodes, indexed_results, model: str):
    """Write episode_metrics and detector_trace CSVs for notebook consumption."""
    import pandas as pd

    model_slug = model.replace("/", "_").replace(".", "_")
    output_dir = local_uncertainty_root()
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_rows = []
    trace_rows = []
    for (task_name, episode_name, _), result in zip(episodes, indexed_results):
        if result is None:
            continue
        reasoning_dict = result.get("reasoning_dict") or {}
        correction_dict = result.get("correction_dict") or {}
        coplan_success = (
            correction_dict.get("success")
            if result.get("status") == "ok" and result.get("replay_available")
            else None
        )
        episode_rows.append({
            "task": task_name,
            "episode": episode_name,
            "status": result.get("status"),
            "artifact_mode": result.get("artifact_mode"),
            "replay_available": result.get("replay_available", False),
            "coplan_success": coplan_success,
            "pred_failure_step": reasoning_dict.get("pred_failure_step"),
            "gt_failure_step": reasoning_dict.get("gt_failure_step"),
            "pred_failure_reason": reasoning_dict.get("pred_failure_reason", ""),
            "gt_failure_reason": reasoning_dict.get("gt_failure_reason", ""),
            "error_type": reasoning_dict.get("error_type", ""),
            "gt_error_type": reasoning_dict.get("gt_error_type", ""),
        })

        detector_trace = reasoning_dict.get("detector_trace") or []
        for trace in detector_trace:
            score = trace.get("score") or {}
            predicted_label = trace.get("predicted_label")
            score_status = score.get("score_status", "unscored")
            constrained_parse_fail = (
                score_status not in ("available", "text_fallback") or predicted_label is None
            )
            trace_rows.append({
                "task": task_name,
                "episode": episode_name,
                "step": trace.get("step"),
                "subgoal": trace.get("subgoal", ""),
                "evaluation_active": bool(trace.get("evaluation_active")),
                "predicted_success": trace.get("predicted_success"),
                "oracle_success": trace.get("oracle_success"),
                "predicted_label": predicted_label,
                "uncertainty_metric": trace.get("uncertainty_metric", ""),
                "uncertainty_value": trace.get("uncertainty_value"),
                "confidence": score.get("confidence"),
                "entropy": score.get("entropy"),
                "score_status": score_status,
                "raw_score_status": score.get("raw_score_status", ""),
                "fallback_applied": score.get("fallback_applied", False),
                "confidence_label": score.get("confidence_label"),
                "constrained_parse_fail": constrained_parse_fail,
                "failed_response_text": (
                    trace.get("response_text", "") if constrained_parse_fail else ""
                ),
            })

    ep_path = output_dir / f"episode_metrics__{model_slug}.csv"
    dt_path = output_dir / f"detector_trace__{model_slug}.csv"
    pd.DataFrame(episode_rows).to_csv(ep_path, index=False)
    pd.DataFrame(trace_rows).to_csv(dt_path, index=False)
    try:
        print(f"[csv] {ep_path}")
        print(f"[csv] {dt_path}")
    except (ValueError, OSError):
        pass

    # --- plan trace (two-pass replan entropy) ---
    plan_trace_rows = []
    for (task_name, episode_name, _), result in zip(episodes, indexed_results):
        if result is None:
            continue
        replan_dict = result.get("replan_dict") or {}
        plan_trace = replan_dict.get("plan_trace") or []
        for entry in plan_trace:
            plan_trace_rows.append({
                "task": task_name,
                "episode": episode_name,
                "step_index": entry.get("step_index"),
                "candidates_json": json.dumps(entry.get("candidates", [])),
                "selected_index": entry.get("selected_index"),
                "selected_label": entry.get("selected_label"),
                "selected_action": entry.get("selected_action"),
                "confidence": entry.get("confidence"),
                "entropy": entry.get("entropy"),
                "score_status": entry.get("score_status"),
            })
    if plan_trace_rows:
        pt_path = output_dir / f"plan_trace__{model_slug}.csv"
        pd.DataFrame(plan_trace_rows).to_csv(pt_path, index=False)
        try:
            print(f"[csv] {pt_path}")
        except (ValueError, OSError):
            pass

    # --- worker stage timings ---
    timing_rows = []
    for (task_name, episode_name, _), result in zip(episodes, indexed_results):
        if result is None:
            continue
        for stage_name, elapsed_s in (result.get("stage_times") or []):
            timing_rows.append({
                "task": task_name,
                "episode": episode_name,
                "stage": stage_name,
                "elapsed_s": round(elapsed_s, 3),
            })
    if timing_rows:
        st_path = output_dir / f"stage_timings__{model_slug}.csv"
        pd.DataFrame(timing_rows).to_csv(st_path, index=False)
        try:
            print(f"[csv] {st_path}")
        except (ValueError, OSError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
