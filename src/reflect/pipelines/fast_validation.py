"""
fast_pipeline.py
----------------
Parallelisable version of the REFLECT validation pipeline.

This module started as an in-memory adaptation of `exp.py`, but now restores
episode-level persistence for expensive intermediates such as scene graphs and
summaries. That keeps the worker-friendly API while avoiding repeated scene
graph computation across reruns or machine restarts.
"""

import os
import io
import sys
import json
import pickle
import contextlib
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np

from reflect.core.paths import sim_episode_summary_dir


@dataclass
class EpisodeConfig:
    data_path: str
    task_name: str
    episode_name: str
    llm_prompter: Any
    prompt_template: dict
    with_audio: int
    status_dict: Any = None
    two_pass_replan: bool = False
    multi_plan_replan: bool = False
    gen_prompter: Any = None  # thinking proposer for multi_plan_replan
    sim_grounded_replan: bool = False  # use generate_replan_multi_plan_sim (propose+execute loop)


# ─── suppress noisy stdout/stderr from worker processes ──────────────────────
def _stream_fileno(stream):
    if stream is None:
        return None
    try:
        return stream.fileno()
    except (AttributeError, io.UnsupportedOperation, OSError):
        return None


@contextlib.contextmanager
def _quiet():
    """Silence Python, subprocess, and native-library stdout/stderr noise."""
    sink = open(os.devnull, "w")
    stdout_fd = _stream_fileno(getattr(sys, "__stdout__", None)) or _stream_fileno(sys.stdout)
    stderr_fd = _stream_fileno(getattr(sys, "__stderr__", None)) or _stream_fileno(sys.stderr)
    saved_fds = []
    try:
        if stdout_fd is not None:
            saved_fds.append((stdout_fd, os.dup(stdout_fd)))
            os.dup2(sink.fileno(), stdout_fd)
        if stderr_fd is not None:
            saved_fds.append((stderr_fd, os.dup(stderr_fd)))
            os.dup2(sink.fileno(), stderr_fd)

        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
    finally:
        for target_fd, saved_fd in reversed(saved_fds):
            try:
                os.dup2(saved_fd, target_fd)
            finally:
                os.close(saved_fd)
        sink.close()


def _load_scene_graph_cache(cache_dir):
    """Load cached scene graph artifacts if the on-disk cache is complete."""
    global_sg_path = os.path.join(cache_dir, "global_sg.pkl")
    key_frames_path = os.path.join(cache_dir, "L1_key_frames.txt")
    local_graph_dir = os.path.join(cache_dir, "local_graphs")

    if not (
        os.path.exists(global_sg_path)
        and os.path.exists(key_frames_path)
        and os.path.isdir(local_graph_dir)
    ):
        return None

    local_graph_files = sorted(
        file_name for file_name in os.listdir(local_graph_dir)
        if file_name.startswith("local_sg_") and file_name.endswith(".pkl")
    )
    if not local_graph_files:
        return None

    with open(global_sg_path, "rb") as fh:
        global_sg = pickle.load(fh)

    with open(key_frames_path, "r") as fh:
        key_frames = [int(line.strip()) for line in fh if line.strip()]

    local_sgs = {}
    for file_name in local_graph_files:
        step_idx = int(file_name[len("local_sg_"):-len(".pkl")])
        with open(os.path.join(local_graph_dir, file_name), "rb") as fh:
            local_sgs[step_idx] = pickle.load(fh)

    return local_sgs, global_sg, key_frames


def _save_scene_graph_cache(cache_dir, local_sgs, global_sg, key_frames):
    os.makedirs(cache_dir, exist_ok=True)
    local_graph_dir = os.path.join(cache_dir, "local_graphs")
    os.makedirs(local_graph_dir, exist_ok=True)

    # Replace prior cache contents so stale per-step files do not linger.
    for file_name in os.listdir(local_graph_dir):
        if file_name.startswith("local_sg_") and file_name.endswith(".pkl"):
            os.remove(os.path.join(local_graph_dir, file_name))

    for step_idx, local_sg in local_sgs.items():
        with open(os.path.join(local_graph_dir, f"local_sg_{step_idx}.pkl"), "wb") as fh:
            pickle.dump(local_sg, fh)

    with open(os.path.join(cache_dir, "L1_key_frames.txt"), "w") as fh:
        for frame in key_frames:
            fh.write(f"{frame}\n")

    with open(os.path.join(cache_dir, "global_sg.pkl"), "wb") as fh:
        pickle.dump(global_sg, fh)


def _read_summary_cache_bundle(cache_dir):
    l1_path = os.path.join(cache_dir, "state_summary_L1.txt")
    l2_path = os.path.join(cache_dir, "state_summary_L2.txt")
    if not (os.path.exists(l1_path) and os.path.exists(l2_path)):
        return None

    meta = {}
    meta_path = os.path.join(cache_dir, "summary_cache_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as fh:
            meta = json.load(fh)

    with open(l1_path, "r") as fh:
        l1_captions = fh.readlines()
    with open(l2_path, "r") as fh:
        l2_captions = fh.readlines()

    key_frames = []
    key_frames_path = os.path.join(cache_dir, "L1_key_frames.txt")
    if os.path.exists(key_frames_path):
        with open(key_frames_path, "r") as fh:
            key_frames = [int(line.strip()) for line in fh if line.strip()]

    return {
        "cache_dir": cache_dir,
        "with_audio": meta.get("with_audio"),
        "l1_summary": "".join(l1_captions),
        "l2_summary": "".join(l2_captions),
        "l1_captions": l1_captions,
        "l2_captions": l2_captions,
        "key_frames": key_frames,
    }


def _summary_cache_candidates(data_path, task_name, episode_name):
    candidates = [
        str(sim_episode_summary_dir(task_name, episode_name)),
        os.path.join(data_path, "state_summary", task_name, episode_name),
    ]

    deduped = []
    seen = set()
    for candidate in candidates:
        norm = os.path.abspath(candidate)
        if norm in seen:
            continue
        deduped.append(norm)
        seen.add(norm)
    return deduped


def _load_best_summary_cache(data_path, task_name, episode_name, with_audio):
    fallback_bundle = None
    for cache_dir in _summary_cache_candidates(data_path, task_name, episode_name):
        bundle = _read_summary_cache_bundle(cache_dir)
        if bundle is None:
            continue
        if bundle["with_audio"] == with_audio:
            bundle["audio_match"] = True
            return bundle
        if fallback_bundle is None:
            bundle["audio_match"] = False
            fallback_bundle = bundle
    return fallback_bundle


def _save_key_frames_cache(cache_dir, key_frames):
    if not key_frames:
        return
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "L1_key_frames.txt"), "w") as fh:
        for frame in key_frames:
            fh.write(f"{frame}\n")


def _persist_summary_bundle(cache_dir, summary_bundle):
    if summary_bundle is None:
        return
    _save_summary_cache(
        cache_dir,
        summary_bundle["l1_summary"],
        summary_bundle["l2_summary"],
        summary_bundle.get("with_audio"),
    )
    _save_key_frames_cache(cache_dir, summary_bundle.get("key_frames") or [])


def _count_event_pickles(data_path):
    events_dir = os.path.join(data_path, "events")
    if not os.path.isdir(events_dir):
        return 0
    return len([name for name in os.listdir(events_dir) if name.endswith(".pickle")])


def _has_raw_episode_artifacts(data_path):
    return (
        _count_event_pickles(data_path) > 0
        and os.path.exists(os.path.join(data_path, "interact_actions.pickle"))
        and os.path.exists(os.path.join(data_path, "nav_actions.pickle"))
    )


def _infer_num_events(data_path, l1_captions=None):
    event_count = _count_event_pickles(data_path)
    if event_count > 0:
        return event_count

    ego_dir = os.path.join(data_path, "ego_img")
    if os.path.isdir(ego_dir):
        ego_count = len([name for name in os.listdir(ego_dir) if name.endswith(".png")])
        if ego_count > 0:
            return ego_count

    if l1_captions:
        try:
            return max(_caption_step_seconds(caption) for caption in l1_captions)
        except Exception:
            return len(l1_captions)
    return 0


def _latest_visual_observation(l1_captions):
    for caption in reversed(l1_captions or []):
        obs_idx = caption.find("Visual observation:")
        if obs_idx == -1:
            continue
        observation = caption[obs_idx + len("Visual observation:"):].strip()
        audio_idx = observation.find("Auditory observation:")
        if audio_idx != -1:
            observation = observation[:audio_idx].strip()
        if observation:
            return observation
    return ""


def _save_summary_cache(cache_dir, l1_summary, l2_summary, with_audio):
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "state_summary_L1.txt"), "w") as fh:
        fh.write(l1_summary)
    with open(os.path.join(cache_dir, "state_summary_L2.txt"), "w") as fh:
        fh.write(l2_summary)
    with open(os.path.join(cache_dir, "summary_cache_meta.json"), "w") as fh:
        json.dump({"with_audio": with_audio}, fh)


SUBGOAL_VERIFIER_CHOICES = {"A": "Yes", "B": "No"}


def _flatten_steps(value):
    if value is None:
        return []
    if isinstance(value, (int, float, str)):
        return [str(value)]

    flat = []
    for item in value:
        if isinstance(item, list):
            flat.extend(str(x) for x in item)
        else:
            flat.append(str(item))
    return flat


def _ts_to_sec(ts):
    minutes, seconds = str(ts).strip().split(":")
    return int(minutes) * 60 + int(seconds)


def _caption_step_seconds(caption):
    return _ts_to_sec(caption.split(".", 1)[0])


def resolve_oracle_failure_index(task, l2_captions):
    if task.get("chosen_failure"):
        return None

    gt_steps = _flatten_steps(task.get("gt_failure_step"))
    if not gt_steps or not l2_captions:
        return None

    try:
        gt_seconds = [_ts_to_sec(step) for step in gt_steps]
        caption_seconds = [_caption_step_seconds(caption) for caption in l2_captions]
    except Exception:
        return None

    if len(gt_seconds) == 1:
        target = gt_seconds[0]
        for idx, step_seconds in enumerate(caption_seconds):
            if step_seconds == target:
                return idx
        for idx, step_seconds in enumerate(caption_seconds):
            if step_seconds >= target:
                return idx
        return len(caption_seconds) - 1

    start, end = gt_seconds[0], gt_seconds[-1]
    for idx, step_seconds in enumerate(caption_seconds):
        if start <= step_seconds <= end:
            return idx
    for idx, step_seconds in enumerate(caption_seconds):
        if step_seconds >= start:
            return idx
    return len(caption_seconds) - 1


def oracle_label_for_subgoal(task, idx, oracle_failure_index):
    if task.get("chosen_failure"):
        return True, True
    if oracle_failure_index is None:
        return None, False
    if idx < oracle_failure_index:
        return True, True
    if idx == oracle_failure_index:
        return False, True
    return None, False


def uncertainty_value(score_metadata, metric):
    if not score_metadata:
        return None
    if metric == "token_prob":
        return score_metadata.get("uncertainty")
    return score_metadata.get("entropy")


def is_success_label(choice_label):
    if choice_label is None:
        return None
    return str(choice_label).strip().upper() == "A"


def parse_verifier_result(answer_text, score_metadata):
    from reflect.llm.prompter import extract_choice_label

    predicted_label = (score_metadata or {}).get("predicted_label") or extract_choice_label(
        answer_text,
        SUBGOAL_VERIFIER_CHOICES,
    )
    predicted_success = is_success_label(predicted_label)
    if predicted_success is None:
        normalized_answer = str(answer_text or "").strip().lower()
        if normalized_answer.startswith("yes"):
            predicted_success = True
        elif normalized_answer.startswith("no"):
            predicted_success = False
    return predicted_label, predicted_success


def get_robot_plan_mem_local(l2_captions, l1_captions, step=None, with_obs=False):
    captions = l1_captions if with_obs else l2_captions
    robot_plan = ""
    for caption in captions:
        if step is not None and step in caption:
            break
        if with_obs:
            robot_plan += caption
        else:
            robot_plan += caption[:caption.find("Visual observation") - 1] + "\n"
    return robot_plan


# ─── in-memory scene-graph generation ────────────────────────────────────────
def scene_graphs_mem(events, object_list,
                      nav_actions, interact_actions,
                      with_audio, detected_sounds, task):
    """
    Pure in-memory equivalent of exp.generate_scene_graphs().
    Returns:
        local_sgs  : dict  {step_idx -> SceneGraph}
        global_sg  : SceneGraph
        key_frames : list[int]
    """
    # late-import so each worker process loads its own copy
    from reflect.perception.scene_graph import SceneGraph, Node as SceneGraphNode
    from reflect.perception.local_graph import get_scene_graph
    from reflect.core.utils import get_label_from_object_id

    key_frames = []
    local_sgs: dict = {}
    prev_graph = SceneGraph(event=None, task=task)
    total_points_dict, bbox3d_dict = {}, {}
    obj_held_prev = None
    cnt, interval = 0, 2
    nav_actions_end_indices = [idx[1] for idx in nav_actions.keys()]

    for step_idx, event in enumerate(events):
        # skip uninformative navigation frames (same logic as original)
        if (step_idx + 1) not in interact_actions and (step_idx + 1) not in nav_actions_end_indices:
            cnt += 1
            if with_audio == 1:
                if step_idx not in detected_sounds and cnt % interval == 0:
                    continue
            elif with_audio == 0:
                if str(step_idx) not in task.get('sounds', {}) and cnt % interval == 0:
                    continue

        # Must be OUTSIDE the if-block so action/nav-end frames also get a
        # fresh scene graph (matches original exp.py indentation).
        local_sg, total_points_dict, obj_held_prev, bbox3d_dict = get_scene_graph(
            step_idx, event, object_list,
            total_points_dict, bbox3d_dict,
            obj_held_prev, task
        )

        local_sgs[step_idx] = local_sg

        # keyframe: scene-graph changed
        if local_sg != prev_graph:
            if (step_idx + 1) not in key_frames:
                key_frames.append(step_idx + 1)
                prev_graph = local_sg

        # keyframe: action frame
        if (step_idx + 1) in interact_actions or (step_idx + 1) in nav_actions_end_indices:
            if (step_idx + 1) not in key_frames:
                key_frames.append(step_idx + 1)

        # keyframe: audio
        if with_audio == 0:
            if str(step_idx) in task.get('sounds', {}):
                if (step_idx + 1) not in key_frames:
                    key_frames.append(step_idx + 1)
        elif with_audio == 1:
            if step_idx in detected_sounds:
                if (step_idx + 1) not in key_frames:
                    key_frames.append(step_idx + 1)

    # build global scene graph from final accumulated point clouds
    global_sg = SceneGraph(events[-1], task)
    for label in total_points_dict:
        name = get_label_from_object_id(label, events, task)
        if name is not None:
            new_node = SceneGraphNode(
                name=name, object_id=label,
                pos3d=bbox3d_dict[label].get_center(),
                corner_pts=np.array(bbox3d_dict[label].get_box_points()),
                pcd=total_points_dict[label], global_node=True
            )
            global_sg.add_node_wo_edge(new_node)

    for label in total_points_dict:
        object_name = label.split("|")[0]
        if object_name in object_list:
            name = get_label_from_object_id(label, events, task)
            if name is not None:
                for node in global_sg.total_nodes:
                    if node.name == name:
                        global_sg.add_node(node)

    global_sg.add_agent()
    return local_sgs, global_sg, key_frames


# ─── in-memory summary generation ────────────────────────────────────────────
def summary_mem(events, nav_actions, interact_actions,
                 with_audio, detected_sounds, task,
                 local_sgs, key_frames, output_dir=None):
    """
    Pure in-memory equivalent of exp.generate_summary().
    Returns:
        l1_summary : str
        l2_summary : str
    """
    from reflect.core.utils import convert_step_to_timestep, convert_timestep_to_step

    try:
        from reflect.pipelines.validation import get_scene_text
    except (ImportError, OSError):
        # validation.py imports audio at top-level; if torchaudio/CUDA are
        # missing, fall back to the local get_scene_text_util.
        from reflect.core.utils import get_scene_text_util as get_scene_text

    try:
        from reflect.perception.audio import audio2label
    except (ImportError, OSError):
        def audio2label(sounds):
            """Stub when audio stack is unavailable."""
            return [s if isinstance(s, str) else str(s) for s in (sounds or [])]

    def _get_held_object_mem(step_idx):
        """walk backwards until we find a local_sg that has a held object"""
        while step_idx >= 0:
            sg = local_sgs.get(step_idx)
            if sg is not None:
                for key in sg.edges:
                    if "robot gripper" in key and key[0] != "nothing":
                        return key[0]
            step_idx -= 1
        return None

    state_summary_L1 = ""
    L1_captions = []

    for step_idx, event in enumerate(events):
        if step_idx not in local_sgs:
            continue
        if (step_idx + 1) not in key_frames:
            continue

        caption = ""

        # action label
        if (step_idx + 1) in interact_actions:
            caption += (f"{convert_step_to_timestep(step=step_idx+1, video_fps=1)}. "
                        f"Action: {interact_actions[step_idx+1]}.")
        else:
            for key in nav_actions:
                min_step, max_step = key
                if min_step <= (step_idx + 1) <= max_step:
                    caption += (f"{convert_step_to_timestep(step=step_idx+1, video_fps=1)}. "
                                f"Action: {nav_actions[key]}.")

        if not caption:
            continue

        # visual observation
        scene_text = get_scene_text(local_sgs[step_idx])
        caption += f" Visual observation: {scene_text}"

        # audio
        if with_audio == 0:
            sounds = task.get('sounds', {})
            if str(step_idx) in sounds:
                held = _get_held_object_mem(step_idx - 1)
                if 'drop' in sounds[str(step_idx)] and held is not None:
                    caption += " Auditory observation: something drops."
                else:
                    caption += f" Auditory observation: {audio2label[sounds[str(step_idx)]]}."
        elif with_audio == 1:
            if step_idx in detected_sounds:
                caption += f" Auditory observation: {detected_sounds[step_idx]}."

        caption += "\n"
        state_summary_L1 += caption
        L1_captions.append(caption)

    # L2: subgoal-level (only interact-action frames)
    L2_captions = []
    for caption in L1_captions:
        step_num = convert_timestep_to_step(caption.split(".")[0], video_fps=1)
        if step_num in interact_actions:
            L2_captions.append(caption.replace("Action", "Goal"))

    state_summary_L2 = "".join(L2_captions)
    return state_summary_L1, state_summary_L2, L1_captions, L2_captions


# ─── top-level worker ─────────────────────────────────────────────────────────
def process_episode(cfg: EpisodeConfig) -> dict:
    """Process a single episode. Suitable as a target for ProcessPoolExecutor.

    Returns a dict with keys: task, episode, status ('ok'|'error'),
    l1_summary, l2_summary, num_keyframes, num_events, reasoning_dict,
    replan_dict, correction_dict, artifact_mode, replay_available, error.
    """
    data_path = cfg.data_path
    task_name = cfg.task_name
    episode_name = cfg.episode_name
    llm_prompter = cfg.llm_prompter
    prompt_template = cfg.prompt_template
    with_audio = cfg.with_audio
    status_dict = cfg.status_dict
    two_pass_replan = cfg.two_pass_replan
    multi_plan_replan = cfg.multi_plan_replan
    gen_prompter = cfg.gen_prompter

    import time as _time
    worker_key = f"{task_name}/{episode_name}"
    _stage_times = []
    _stage_start = _time.monotonic()

    def _stage(s):
        _stage_times.append((s, _time.monotonic() - _stage_start))
        if status_dict is not None:
            status_dict[worker_key] = s
        try:
            print(f"[stage] {worker_key}: {s}", flush=True)
        except (ValueError, OSError):
            pass

    result = dict(task=task_name, episode=episode_name,
                  status='error', l1_summary='', l2_summary='',
                  num_keyframes=0, num_events=0, global_sg=None,
                  reasoning_dict=None, replan_dict=None, correction_dict=None,
                  artifact_mode='full', replay_available=True,
                  summary_source_dir='', summary_audio_match=True,
                  prompts_log=[], stage_times=[],
                  error='')
    try:
        _stage('loading data')
        with open(os.path.join(data_path, 'task.json')) as f:
            task_detail = json.load(f)
        # Merge master task config (name, success_condition, etc.) if missing from per-episode task.json
        if 'name' not in task_detail or 'success_condition' not in task_detail:
            from reflect.core.paths import sim_tasks_config
            with open(sim_tasks_config()) as _tf:
                _master_tasks = json.load(_tf)
            _tid = task_detail.get('task_idx')
            _master_entry = next(
                (v for v in _master_tasks.values() if v.get('task_idx') == _tid), None
            )
            if _master_entry:
                for _k in ('name', 'success_condition'):
                    if _k not in task_detail:
                        task_detail[_k] = _master_entry.get(_k, '')
            else:
                # Fallback: derive name from folder_name / task_name parameter
                if 'name' not in task_detail:
                    task_detail['name'] = task_name
                if 'success_condition' not in task_detail:
                    task_detail['success_condition'] = ''

        cache_dir = str(sim_episode_summary_dir(task_name, episode_name))
        summary_bundle = _load_best_summary_cache(data_path, task_name, episode_name, with_audio)
        raw_artifacts_available = _has_raw_episode_artifacts(data_path)

        result['summary_source_dir'] = (summary_bundle or {}).get('cache_dir', '')
        result['summary_audio_match'] = (summary_bundle or {}).get('audio_match', True)

        if not raw_artifacts_available and summary_bundle is None:
            raise FileNotFoundError(
                "Episode is missing raw simulator artifacts and no cached summaries were found. "
                "Expected either events/interact_actions/nav_actions for a full run or "
                "state_summary_L1.txt + state_summary_L2.txt for summary-only reasoning."
            )

        if not raw_artifacts_available:
            _stage('summaries (cached)')
            _persist_summary_bundle(cache_dir, summary_bundle)

            l1 = summary_bundle["l1_summary"]
            l2 = summary_bundle["l2_summary"]
            l1_captions = list(summary_bundle["l1_captions"])
            l2_captions = list(summary_bundle["l2_captions"])
            key_frames = summary_bundle.get("key_frames") or []

            result.update(
                l1_summary=l1,
                l2_summary=l2,
                num_keyframes=len(key_frames) if key_frames else len(l1_captions),
                num_events=_infer_num_events(data_path, l1_captions),
                artifact_mode='summary_only',
                replay_available=False,
            )

            _stage('reasoning (LLM)')
            reasoning_dict, reasoning_prompts = run_reasoning_mem(
                task_detail,
                llm_prompter,
                global_sg=None,
                prompt_info=prompt_template,
                L2_captions=l2_captions,
                L1_captions=l1_captions,
                data_path=data_path,
                task_name=task_name,
                episode_name=episode_name,
            )

            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, 'reasoning.json'), 'w') as _rf:
                json.dump(reasoning_dict, _rf, default=str)

            # Two-pass replan can run from summaries (no simulator needed)
            if two_pass_replan:
                _stage('replan (two-pass summary)')
                replan_dict, replan_prompts = generate_replan_two_pass_summary(
                    task_detail, llm_prompter, l2_captions,
                    reasoning_dict.get('pred_failure_reason', ''),
                    prompt_template,
                )
                reasoning_prompts.extend(replan_prompts)
            else:
                skip_reason = (
                    "Raw simulator events/actions are unavailable and two-pass replan was not requested."
                )
                replan_dict = {"skipped": True, "reason": skip_reason}

            correction_dict = {
                "skipped": True,
                "success": None,
                "reason": "Raw simulator events/actions are unavailable, so correction execution was skipped.",
            }

            _stage('done')
            result.update(
                status='ok',
                global_sg=None,
                reasoning_dict=reasoning_dict,
                replan_dict=replan_dict,
                correction_dict=correction_dict,
                prompts_log=reasoning_prompts,
            )
            return result

        # late-imports so each worker process loads its own copy
        from reflect.core.data import load_episode_data
        from reflect.sim.recovery import run_correction_mem

        with _quiet():
            events, task_detail, object_list, interact_actions, nav_actions = load_episode_data(data_path, task_detail)

        result.update(
            num_events=len(events),
            artifact_mode='full',
            replay_available=True,
        )

        # sound detection
        _stage('sound detection')
        detected_sounds = []
        if with_audio == 1:
            try:
                from reflect.perception.audio import process_sound, warn_audio_stack_unavailable_once
                with _quiet():
                    detected_sounds = process_sound(data_path, object_list)
            except (ImportError, OSError):
                try:
                    from reflect.perception.audio import warn_audio_stack_unavailable_once
                    warn_audio_stack_unavailable_once()
                except (ImportError, OSError):
                    pass
            except Exception:
                pass

        # scene graphs (persistent cache preferred)
        _stage('scene graphs')
        scene_graph_cache = _load_scene_graph_cache(cache_dir)
        if scene_graph_cache is None:
            with _quiet():
                local_sgs, global_sg, key_frames = scene_graphs_mem(
                    events, object_list,
                    nav_actions, interact_actions,
                    with_audio, detected_sounds, task_detail
                )
            _save_scene_graph_cache(cache_dir, local_sgs, global_sg, key_frames)
        else:
            local_sgs, global_sg, key_frames = scene_graph_cache

        # summaries (persistent cache preferred)
        _stage('summaries')
        summary_bundle = _read_summary_cache_bundle(cache_dir)
        if summary_bundle is None or summary_bundle.get('with_audio') != with_audio:
            with _quiet():
                l1, l2, l1_captions, l2_captions = summary_mem(
                    events, nav_actions, interact_actions,
                    with_audio, detected_sounds, task_detail,
                    local_sgs, key_frames
                )
            _save_summary_cache(cache_dir, l1, l2, with_audio)
        else:
            l1 = summary_bundle['l1_summary']
            l2 = summary_bundle['l2_summary']
            l1_captions = summary_bundle['l1_captions']
            l2_captions = summary_bundle['l2_captions']

        _stage('reasoning (LLM)')
        reasoning_dict, reasoning_prompts = run_reasoning_mem(
            task_detail,
            llm_prompter,
            global_sg,
            prompt_template,
            list(l2_captions),
            list(l1_captions),
            data_path=data_path,
            task_name=task_name,
            episode_name=episode_name,
        )

        last_frame = len(events) - 1

        # reasoning.json is read by action_primitives.pick_up() via taskUtil
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, 'reasoning.json'), 'w') as _rf:
            json.dump(reasoning_dict, _rf, default=str)

        sim_grounded_replan = cfg.sim_grounded_replan

        if sim_grounded_replan and multi_plan_replan and gen_prompter is not None:
            # Propose, select, and execute each step in the sim in one loop.
            # Returns replan_dict with 'success' already included; no separate
            # correction pass needed.
            _stage('replan+correction (sim-grounded)')
            with _quiet():
                replan_dict, correction_prompts = generate_replan_multi_plan_sim(
                    data_path=data_path,
                    task=task_detail,
                    final_event=events[-1],
                    last_frame=last_frame,
                    object_list=object_list,
                    gen_prompter=gen_prompter,
                    score_prompter=llm_prompter,
                    prompt_info=prompt_template,
                    global_sg=global_sg,
                    pred_failure_reason=reasoning_dict['pred_failure_reason'],
                    task_name=task_name,
                )
            correction_dict = {'success': replan_dict.get('success')}
        else:
            _stage('replan (LLM)')
            with _quiet():
                if multi_plan_replan and gen_prompter is not None:
                    replan_dict, correction_prompts = generate_replan_multi_plan(
                        task_detail, gen_prompter, llm_prompter, global_sg,
                        events[-1], object_list,
                        reasoning_dict['pred_failure_reason'], prompt_template,
                        task_name=task_name,
                    )
                elif two_pass_replan:
                    replan_dict, correction_prompts = generate_replan_two_pass(
                        task_detail, llm_prompter, global_sg,
                        events[-1], object_list,
                        reasoning_dict['pred_failure_reason'], prompt_template
                    )
                else:
                    replan_dict, correction_prompts = generate_replan_mem(
                        task_detail, llm_prompter, global_sg,
                        events[-1], object_list,
                        reasoning_dict['pred_failure_reason'], prompt_template
                    )

            _stage('correction (sim)')
            with _quiet():
                correction_dict = run_correction_mem(
                    data_path, task_detail, events[-1], last_frame, replan_dict
                )

        # Combine all LLM call logs (reasoning passes first, then replan/correction)
        prompts_log = reasoning_prompts + correction_prompts

        _stage('done')
        result.update(
            status='ok',
            l1_summary=l1,
            l2_summary=l2,
            num_keyframes=len(key_frames),
            global_sg=None,
            reasoning_dict=reasoning_dict,
            replan_dict=replan_dict,
            correction_dict=correction_dict,
            artifact_mode='full',
            replay_available=True,
            prompts_log=prompts_log,
        )

    except Exception as e:
        _stage(f'ERROR: {type(e).__name__}')
        result['error'] = traceback.format_exc()

    result['stage_times'] = _stage_times
    return result


def process_episode_mem(args):
    """Backward-compatible wrapper: accepts the old positional-tuple calling convention."""
    if len(args) < 6:
        raise ValueError("process_episode_mem expects at least 6 positional tuple items")
    cfg = EpisodeConfig(
        data_path=args[0],
        task_name=args[1],
        episode_name=args[2],
        llm_prompter=args[3],
        prompt_template=args[4],
        with_audio=args[5],
        status_dict=args[6] if len(args) > 6 else None,
        two_pass_replan=args[7] if len(args) > 7 else False,
    )
    return process_episode(cfg)


def generate_replan_mem(task, llm_prompter, global_sg, last_event, task_object_list,
                        pred_failure_reason, prompt_info):
    """
    In-memory equivalent of generate_replan().

    Parameters
    ----------
    task             : dict  - already-loaded task.json contents
    llm_prompter     : LLM query interface
    global_sg        : SceneGraph
    last_event       : final AI2-THOR event (provides the global object catalogue)
    task_object_list : list[str] - object types relevant to the task
    pred_failure_reason : str - predicted failure reason from run_reasoning_mem
    prompt_info      : dict - already-loaded prompts.json contents

    Returns
    -------
    replan_dict : dict
        task_plan     : list[str]
        llm_plan_raw  : list[dict]
        llm_plan      : list[dict]
        num_steps     : int
    """
    from reflect.models.plan import Plan
    from reflect.core.utils import get_scene_text_util, get_replan_prefix, get_initial_plan, get_admissible_actions

    curr_state = get_scene_text_util(global_sg)

    global_object_list = list(
        set([obj["objectType"] for obj in last_event.metadata["objects"]])
        | set(task_object_list)
    )

    available_actions = get_admissible_actions(global_object_list, last_event)
    available_objects = sorted({token.strip("() ") for action in available_actions for token in action.split(", ")[1:]})

    corr_info = prompt_info['correction']  # cache sub-dict
    prompt = {
        'system': corr_info['template-system'].replace(
            "[PREFIX]",
            get_replan_prefix(available_actions=available_actions, available_objects=available_objects),
        ),
        'user': (
            corr_info['template-user']
            .replace("[TASK_NAME]", task['name'])
            .replace("[PLAN]", get_initial_plan(task['actions']))
            .replace("[FAILURE_REASON]", pred_failure_reason)
            .replace("[CURRENT_STATE]", curr_state)
            .replace("[SUCCESS_CONDITION]", task['success_condition'])
        ),
    }

    plan, _ = llm_prompter.query(
        prompt=prompt,
        sampling_params=corr_info['params'],
        response_model=Plan,
    )

    raw_plan = plan.model_dump()["actions"]

    correction_prompts = [{"call": "correction", "prompt": dict(prompt), "response": raw_plan}]

    return (
        {
            "task_plan":     list(task['actions']),   # original task actions from task.json
            "llm_plan_raw":  raw_plan,                 # raw structured LLM output
            "llm_plan":      raw_plan,                 # executable structured plan
            "num_steps":     len(raw_plan),
        },
        correction_prompts,
    )


PLAN_SELECT_CHOICES = {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D", "E": "Option E"}


def _format_candidates_for_selection(candidates):
    labels = list(PLAN_SELECT_CHOICES.keys())
    lines = []
    for i, candidate in enumerate(candidates):
        obj_str = candidate.obj1
        if candidate.obj2:
            obj_str += f", {candidate.obj2}"
        lines.append(f"{labels[i]}) ({candidate.action}, {obj_str})")
    return "\n".join(lines)


def _propose_and_select_step(
    step_index, llm_prompter, propose_info, select_info,
    task_name, initial_plan_str, pred_failure_reason,
    curr_state, success_condition, available_actions, available_objects,
    actions_so_far_str, prompts_log,
):
    """Run one propose+select step.

    Returns (candidates, selected_index, selected_label, select_score).
    candidates is empty if the propose call returned nothing.
    """
    from reflect.models.plan import PlanStepCandidates

    propose_prompt = {
        'system': propose_info['template-system'],
        'user': (
            propose_info['template-user']
            .replace("[TASK_NAME]", task_name)
            .replace("[PLAN]", initial_plan_str)
            .replace("[FAILURE_REASON]", pred_failure_reason)
            .replace("[CURRENT_STATE]", curr_state)
            .replace("[SUCCESS_CONDITION]", success_condition)
            .replace("[ACTIONS_SO_FAR]", actions_so_far_str)
            .replace("[AVAILABLE_ACTIONS]", ", ".join(available_actions[:50]))
            .replace("[AVAILABLE_OBJECTS]", ", ".join(available_objects[:50]))
        ),
    }

    candidates_model, _ = llm_prompter.query(
        prompt=propose_prompt,
        sampling_params=propose_info['params'],
        response_model=PlanStepCandidates,
    )
    prompts_log.append({
        "call": "plan-propose",
        "step_index": step_index,
        "prompt": dict(propose_prompt),
        "response": candidates_model.model_dump(),
    })

    candidates = list(candidates_model.candidates[:5])
    if not candidates:
        return candidates, None, None, None

    candidates_text = _format_candidates_for_selection(candidates)
    select_prompt = {
        'system': select_info['template-system'],
        'user': (
            select_info['template-user']
            .replace("[TASK_NAME]", task_name)
            .replace("[FAILURE_REASON]", pred_failure_reason)
            .replace("[CURRENT_STATE]", curr_state)
            .replace("[SUCCESS_CONDITION]", success_condition)
            .replace("[ACTIONS_SO_FAR]", actions_so_far_str)
            .replace("[CANDIDATES]", candidates_text)
        ),
    }

    labels = list(PLAN_SELECT_CHOICES.keys())[:len(candidates)]
    choice_spec = {label: f"Option {label}" for label in labels}

    select_text, select_score = llm_prompter.query(
        prompt=select_prompt,
        sampling_params=select_info['params'],
        choice_spec=choice_spec,
    )
    prompts_log.append({
        "call": "plan-select",
        "step_index": step_index,
        "prompt": dict(select_prompt),
        "response": select_text,
        "score": select_score,
    })

    selected_label = (select_score or {}).get("predicted_label")
    if selected_label is None:
        from reflect.llm.prompter import extract_choice_label
        selected_label = extract_choice_label(select_text, choice_spec)
    selected_index = labels.index(selected_label) if selected_label in labels else 0

    return candidates, selected_index, selected_label, select_score


def generate_replan_two_pass(task, llm_prompter, global_sg, last_event, task_object_list,
                              pred_failure_reason, prompt_info, max_steps=6, task_name=None):
    """Generate a correction plan (single-shot) then score each step for entropy.

    Pass 1 - Plan generation: use the single-shot 'correction' prompt to get a
    coherent multi-step plan (same as generate_replan_mem).

    Pass 2 - Per-step scoring: for each step in the generated plan, ask the LLM
    to propose 5 alternative candidates, insert the real step among them, then
    ask the LLM to select. The logprobs on the selection give per-step entropy.
    """
    from reflect.models.plan import Plan, PlanStepCandidates
    from reflect.models.action_primitive import ActionPrimitive
    from reflect.core.utils import (
        get_scene_text_util, get_replan_prefix,
        get_initial_plan, get_admissible_actions,
    )

    curr_state = get_scene_text_util(global_sg)
    global_object_list = list(
        set(obj["objectType"] for obj in last_event.metadata["objects"])
        | set(task_object_list)
    )
    available_actions = get_admissible_actions(global_object_list, last_event)
    available_objects = sorted({
        token.strip("() ")
        for action in available_actions
        for token in action.split(", ")[1:]
    })
    initial_plan_str = get_initial_plan(task['actions'])

    prompts_log = []

    # ── Pass 1: single-shot plan generation ──────────────────────────────
    corr_info = prompt_info['correction']
    corr_prompt = {
        'system': corr_info['template-system'].replace(
            "[PREFIX]",
            get_replan_prefix(
                available_actions=available_actions,
                available_objects=available_objects,
            ),
        ),
        'user': (
            corr_info['template-user']
            .replace("[TASK_NAME]", task_name or task['name'])
            .replace("[PLAN]", initial_plan_str)
            .replace("[FAILURE_REASON]", pred_failure_reason)
            .replace("[CURRENT_STATE]", curr_state)
            .replace("[SUCCESS_CONDITION]", task['success_condition'])
        ),
    }

    plan_model, _ = llm_prompter.query(
        prompt=corr_prompt,
        sampling_params=corr_info['params'],
        response_model=Plan,
    )
    raw_plan = plan_model.model_dump()["actions"]
    prompts_log.append({
        "call": "correction",
        "prompt": dict(corr_prompt),
        "response": raw_plan,
    })

    # Build ActionPrimitive list from the generated plan
    plan_actions = [
        ActionPrimitive(action=s['action'], obj1=s.get('obj1'), obj2=s.get('obj2'))
        for s in raw_plan
    ]

    # ── Pass 2: per-step scoring via propose + select ────────────────────
    propose_info = prompt_info['plan-propose']
    select_info = prompt_info['plan-select']
    plan_trace = []

    for step_index, real_action in enumerate(plan_actions[:max_steps]):
        actions_so_far_str = ", ".join(
            f"({a.action}, {a.obj1}" + (f", {a.obj2}" if a.obj2 else "") + ")"
            for a in plan_actions[:step_index]
        ) or "(none)"

        candidates, selected_index, selected_label, select_score = _propose_and_select_step(
            step_index, llm_prompter, propose_info, select_info,
            task_name or task['name'], initial_plan_str, pred_failure_reason,
            curr_state, task['success_condition'], available_actions, available_objects,
            actions_so_far_str, prompts_log,
        )
        if not candidates:
            break

        # Ensure the real action from the generated plan is among the candidates
        real_is_present = any(
            c.action == real_action.action
            and c.obj1 == real_action.obj1
            and c.obj2 == real_action.obj2
            for c in candidates
        )
        if not real_is_present:
            candidates[-1] = PlanStepCandidates.model_validate({
                "candidates": [{
                    "action": real_action.action,
                    "obj1": real_action.obj1,
                    "obj2": real_action.obj2,
                }]
            }).candidates[0]

        plan_trace.append({
            "step_index": step_index,
            "candidates": [c.model_dump() for c in candidates],
            "selected_index": selected_index,
            "selected_label": selected_label,
            "selected_action": real_action.model_dump(),
            "confidence": (select_score or {}).get("confidence"),
            "entropy": (select_score or {}).get("entropy"),
            "score_status": (select_score or {}).get("score_status"),
        })

    return (
        {
            "task_plan": list(task['actions']),
            "llm_plan_raw": raw_plan,
            "llm_plan": raw_plan,
            "num_steps": len(raw_plan),
            "plan_trace": plan_trace,
        },
        prompts_log,
    )


def generate_replan_two_pass_summary(task, llm_prompter, l2_captions,
                                     pred_failure_reason, prompt_info, max_steps=6):
    """Two-pass propose/select replan using summary data (no simulator state required)."""
    from reflect.models.plan import Plan
    from reflect.models.action_primitive import ActionPrimitive
    from reflect.core.utils import get_initial_plan

    # Derive current state from the last L2 caption
    last_caption = l2_captions[-1].strip() if l2_captions else "(unknown)"
    # Extract visual observation portion as current state description
    if "Visual observation:" in last_caption:
        curr_state = last_caption[last_caption.index("Visual observation:"):]
    else:
        curr_state = last_caption

    # Derive available objects from task object_list and original plan actions
    task_objects = list(task.get('object_list', []))
    # Also extract objects mentioned in the plan actions
    for action_str in task.get('actions', []):
        parts = action_str.strip("()").split(", ")
        for part in parts[1:]:
            obj = part.strip()
            if obj and obj not in task_objects:
                task_objects.append(obj)
    available_objects = sorted(set(task_objects))

    # Derive available actions from task object types and common action primitives
    action_types = ["pick_up", "put_in", "put_on", "toggle_on", "toggle_off",
                    "open_obj", "close_obj", "slice_obj", "crack_obj", "pour"]
    available_actions = []
    for obj in available_objects:
        for act in action_types:
            available_actions.append(f"({act}, {obj})")

    propose_info = prompt_info['plan-propose']
    select_info = prompt_info['plan-select']
    initial_plan_str = get_initial_plan(task['actions'])

    selected_actions = []
    plan_trace = []
    prompts_log = []

    for step_index in range(max_steps):
        actions_so_far_str = ", ".join(
            f"({a.action}, {a.obj1}" + (f", {a.obj2}" if a.obj2 else "") + ")"
            for a in selected_actions
        ) or "(none)"

        candidates, selected_index, selected_label, select_score = _propose_and_select_step(
            step_index, llm_prompter, propose_info, select_info,
            task['name'], initial_plan_str, pred_failure_reason,
            curr_state, task['success_condition'], available_actions, available_objects,
            actions_so_far_str, prompts_log,
        )
        if not candidates:
            break

        selected_candidate = candidates[selected_index]
        selected_action = ActionPrimitive(
            action=selected_candidate.action,
            obj1=selected_candidate.obj1,
            obj2=selected_candidate.obj2,
        )
        selected_actions.append(selected_action)

        plan_trace.append({
            "step_index": step_index,
            "candidates": [c.model_dump() for c in candidates],
            "selected_index": selected_index,
            "selected_label": selected_label,
            "selected_action": selected_action.model_dump(),
            "confidence": (select_score or {}).get("confidence"),
            "entropy": (select_score or {}).get("entropy"),
            "score_status": (select_score or {}).get("score_status"),
        })

    plan = Plan(actions=selected_actions)
    raw_plan = plan.model_dump()["actions"]

    return (
        {
            "task_plan": list(task['actions']),
            "llm_plan_raw": raw_plan,
            "llm_plan": raw_plan,
            "num_steps": len(raw_plan),
            "plan_trace": plan_trace,
        },
        prompts_log,
    )


def _update_state_from_action(state_str, action):
    """Return state_str with the gripper / toggle effects of action applied.

    Handles pick_up, put_in, put_on, toggle_on, toggle_off.
    Other actions (pour, slice, crack,...) leave the state string unchanged.
    """
    import re
    act  = action.action
    obj1 = action.obj1 or ""
    obj2 = action.obj2 or ""

    if act == "pick_up":
        state_str = re.sub(
            r'nothing is inside robot gripper',
            f'{obj1} is inside robot gripper',
            state_str,
        )
        # defensive: overwrite whatever was in the gripper
        state_str = re.sub(
            r'[\w][\w\s,\-]* is inside robot gripper',
            f'{obj1} is inside robot gripper',
            state_str,
        )
    elif act in ("put_in", "put_on"):
        state_str = re.sub(
            r'[\w][\w\s,\-]* is inside robot gripper',
            'nothing is inside robot gripper',
            state_str,
        )
        if obj2:
            state_str = state_str.rstrip('. ') + f", {obj1} is now placed in/on {obj2}."
    elif act == "toggle_on":
        state_str = re.sub(
            rf'{re.escape(obj1)}\s*\(turned off\)',
            f'{obj1} (turned on)',
            state_str,
            flags=re.IGNORECASE,
        )
    elif act == "toggle_off":
        state_str = re.sub(
            rf'{re.escape(obj1)}\s*\(turned on\)',
            f'{obj1} (turned off)',
            state_str,
            flags=re.IGNORECASE,
        )
    return state_str


_ACTION_VERBS = (
    "pick_up", "put_in", "put_on", "toggle_on", "toggle_off",
    "open", "close", "slice", "pour", "crack_obj",
)


def _build_dynamic_candidate_schema(scene_objects):
    """Return a Pydantic model whose action/obj1/obj2 are Literal-constrained
    to the actual scene contents.

    Grounding lives in the schema (enforced at token-sampling time via JSON
    schema enum), so the thinking proposer can reason freely from state +
    success condition without being handed a distractor-filled object menu.
    """
    from typing import Literal, Optional, List
    from pydantic import create_model

    if not scene_objects:
        raise ValueError("scene_objects must be non-empty for Literal enum")

    unique_objects = tuple(dict.fromkeys(scene_objects))
    ActionEnum = Literal[_ACTION_VERBS]
    ObjectEnum = Literal[unique_objects]

    StepCandidate = create_model(
        "DynamicPlanStepCandidate",
        action=(ActionEnum, ...),
        obj1=(ObjectEnum, ...),
        obj2=(Optional[ObjectEnum], None),
    )
    StepCandidates = create_model(
        "DynamicPlanStepCandidates",
        candidates=(List[StepCandidate], ...),
    )
    return StepCandidates


def generate_replan_multi_plan(
    task,
    gen_prompter,
    score_prompter,
    global_sg,
    last_event,
    task_object_list,
    pred_failure_reason,
    prompt_info,
    num_candidates: int = 4,
    max_steps: int = 50,
    task_name: str = None,
):
    """Per-step multi-candidate replan with thinking-mode generation and fast scoring.

    For each step (up to max_steps):
      - gen_prompter (reasoning_effort="medium") proposes num_candidates diverse
        candidate actions using structured output.
      - score_prompter (reasoning_effort="none") selects the best from A/B/C/D,
        or E to signal the plan is complete. Logprobs give per-step confidence/entropy.

    Returns (replan_dict, prompts_log).
    """
    from reflect.models.action_primitive import ActionPrimitive
    from reflect.core.utils import (
        get_scene_text_util, get_initial_plan, get_admissible_actions,
    )

    # curr_state is updated each iteration via _update_state_from_action
    curr_state = get_scene_text_util(global_sg)
    global_object_list = list(
        set(obj["objectType"] for obj in last_event.metadata["objects"])
        | set(task_object_list)
    )
    available_actions = get_admissible_actions(global_object_list, last_event)
    available_objects = sorted({
        token.strip("() ")
        for action in available_actions
        for token in action.split(", ")[1:]
    })
    initial_plan_str = get_initial_plan(task['actions'])

    task_name = task_name or task['name']
    success_condition = task['success_condition']

    # Dynamic schema: Literal-constrain object names to real scene objects so
    # the proposer can't hallucinate names. Built once - scene composition
    # is fixed across the replan loop.
    CandidateSchema = _build_dynamic_candidate_schema(available_objects)

    propose_info = prompt_info['plan-propose-thinking']
    select_info = prompt_info['plan-select-done']
    # DONE label always occupies E; A..D are the candidate labels
    candidate_labels = list(PLAN_SELECT_CHOICES.keys())[:num_candidates]
    done_label = "E"

    prompts_log = []
    plan_trace = []
    selected_actions = []

    for step_index in range(max_steps):
        print(f"Replan step {step_index}: proposing candidates...", flush=True)
        actions_so_far_str = ", ".join(
            f"({a.action}, {a.obj1}" + (f", {a.obj2}" if a.obj2 else "") + ")"
            for a in selected_actions
        ) or "(none)"

        # ── Propose (thinking model) ──────────────────────────────────────
        propose_prompt = {
            'system': propose_info['template-system'],
            'user': (
                propose_info['template-user']
                .replace("[TASK_NAME]", task_name)
                .replace("[PLAN]", initial_plan_str)
                .replace("[FAILURE_REASON]", pred_failure_reason)
                .replace("[CURRENT_STATE]", curr_state)
                .replace("[SUCCESS_CONDITION]", success_condition)
                .replace("[ACTIONS_SO_FAR]", actions_so_far_str)
            ),
        }

        candidates_model, _ = gen_prompter.query(
            prompt=propose_prompt,
            sampling_params=propose_info['params'],
            response_model=CandidateSchema,
        )
        prompts_log.append({
            "call": "plan-propose-thinking",
            "step_index": step_index,
            "prompt": dict(propose_prompt),
            "response": candidates_model.model_dump(),
        })

        candidates = list(candidates_model.candidates[:num_candidates])
        if not candidates:
            break

        # ── Select (fast model) ───────────────────────────────────────────
        print(f"Selecting from candidates...", flush=True)
        labels = candidate_labels[:len(candidates)]
        choice_spec = {lbl: f"Option {lbl}" for lbl in labels}
        choice_spec[done_label] = "Done"

        candidates_text = _format_candidates_for_selection(candidates)
        select_prompt = {
            'system': select_info['template-system'],
            'user': (
                select_info['template-user']
                .replace("[TASK_NAME]", task_name)
                .replace("[PLAN]", initial_plan_str)
                .replace("[FAILURE_REASON]", pred_failure_reason)
                .replace("[SUCCESS_CONDITION]", success_condition)
                .replace("[ACTIONS_SO_FAR]", actions_so_far_str)
                .replace("[CANDIDATES]", candidates_text)
                .replace("[CURRENT_STATE]", curr_state)
            ),
        }

        select_text, select_score = score_prompter.query(
            prompt=select_prompt,
            sampling_params=select_info['params'],
            choice_spec=choice_spec,
        )
        prompts_log.append({
            "call": "plan-select-done",
            "step_index": step_index,
            "prompt": dict(select_prompt),
            "response": select_text,
            "score": select_score,
        })

        selected_label = (select_score or {}).get("predicted_label")

        print(f"Selected label: {selected_label}", flush=True)

        if selected_label is None:
            from reflect.llm.prompter import extract_choice_label
            selected_label = extract_choice_label(select_text, choice_spec)

        # DONE or unresolvable → plan is complete
        if selected_label == done_label or selected_label not in labels:
            plan_trace.append({
                "step_index": step_index,
                "candidates": [c.model_dump() for c in candidates],
                "selected_label": selected_label,
                "selected_action": None,
                "confidence": (select_score or {}).get("confidence"),
                "entropy": (select_score or {}).get("entropy"),
                "option_probs": (select_score or {}).get("option_probs"),
                "score_status": (select_score or {}).get("score_status"),
                "done": True,
            })
            break
        else:
            print("Not done, updating plan and state and continuing...", flush=True)

        selected_index = labels.index(selected_label)
        selected_candidate = candidates[selected_index]
        selected_action = ActionPrimitive(
            action=selected_candidate.action,
            obj1=selected_candidate.obj1,
            obj2=selected_candidate.obj2,
        )
        selected_actions.append(selected_action)

        plan_trace.append({
            "step_index": step_index,
            "candidates": [c.model_dump() for c in candidates],
            "selected_label": selected_label,
            "selected_action": selected_action.model_dump(),
            "curr_state_after": curr_state,
            "confidence": (select_score or {}).get("confidence"),
            "entropy": (select_score or {}).get("entropy"),
            "option_probs": (select_score or {}).get("option_probs"),
            "score_status": (select_score or {}).get("score_status"),
            "done": False,
        })

    selected_plan_raw = [a.model_dump() for a in selected_actions]
    return (
        {
            "task_plan": list(task['actions']),
            "selected_plan": [a.to_legacy_string() for a in selected_actions],
            "selected_plan_raw": selected_plan_raw,
            # alias so run_correction_mem can find the plan
            "llm_plan": selected_plan_raw,
            "llm_plan_raw": selected_plan_raw,
            "num_steps": len(selected_actions),
            "plan_trace": plan_trace,
        },
        prompts_log,
    )


def generate_replan_multi_plan_sim(
    data_path,
    task,
    final_event,
    last_frame,
    object_list,
    gen_prompter,
    score_prompter,
    prompt_info,
    global_sg,
    pred_failure_reason,
    task_name=None,
    max_steps=10,
    num_candidates=4,
):
    """Sim-grounded variant of generate_replan_multi_plan.

    Same propose/select loop, but executes each selected action in the
    AI2-THOR simulator and updates curr_state from the real scene graph
    rather than applying text heuristics.

    Returns (replan_dict, prompts_log).
    replan_dict keys: task_plan, selected_plan, selected_plan_raw,
                      llm_plan, llm_plan_raw, num_steps, plan_trace, success.
    """
    import io
    import sys
    from reflect.models.action_primitive import ActionPrimitive
    from reflect.core.utils import (
        get_scene_text_util, get_initial_plan, get_admissible_actions,
        check_task_success,
    )
    from reflect.perception.local_graph import get_scene_graph
    import inspect as _inspect
    from reflect.sim.recovery import (
        make_controller, _restore_sim_state, _clear_recovery_artifacts,
        _normalize_instruction, _dispatch_action,
    )
    from reflect.sim.task_manager import TaskUtil
    from reflect.core.paths import sim_output_root

    folder_name = os.path.join(
        os.path.basename(os.path.dirname(os.path.abspath(data_path))),
        os.path.basename(os.path.abspath(data_path)),
    )
    runtime_root = str(sim_output_root())
    _clear_recovery_artifacts(runtime_root, folder_name)

    controller = make_controller(task['scene'])
    _restore_sim_state(controller, task, final_event)
    reachable_positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    taskUtil = TaskUtil(
        folder_name=folder_name,
        controller=controller,
        reachable_positions=reachable_positions,
        failure_injection=False,
        index=0,
        repo_path=runtime_root,
        chosen_failure=task.get('chosen_failure'),
        failure_injection_params=task.get('failure_injection_params'),
        counter=last_frame,
        replan=True,
    )

    object_set = list(
        set(obj["objectType"] for obj in final_event.metadata["objects"])
        | set(object_list)
    )
    available_actions = get_admissible_actions(object_set, final_event)
    available_objects = sorted({
        token.strip("() ")
        for action in available_actions
        for token in action.split(", ")[1:]
    })

    CandidateSchema = _build_dynamic_candidate_schema(available_objects)
    propose_info = prompt_info['plan-propose-thinking']
    select_info = prompt_info['plan-select-done']
    candidate_labels = list(PLAN_SELECT_CHOICES.keys())[:num_candidates]
    done_label = "E"

    task_name = task_name or task['name']
    success_condition = task['success_condition']
    initial_plan_str = get_initial_plan(task['actions'])
    curr_state = get_scene_text_util(global_sg)

    total_points_dict, bbox3d_dict, obj_held_prev = {}, {}, None
    selected_actions = []
    plan_trace = []
    prompts_log = []

    for step_index in range(max_steps):
        print(f"Replan step {step_index}: proposing candidates...", flush=True)
        actions_so_far_str = ", ".join(
            f"({a.action}, {a.obj1}" + (f", {a.obj2}" if a.obj2 else "") + ")"
            for a in selected_actions
        ) or "(none)"

        propose_prompt = {
            'system': propose_info['template-system'],
            'user': (
                propose_info['template-user']
                .replace("[TASK_NAME]", task_name)
                .replace("[PLAN]", initial_plan_str)
                .replace("[FAILURE_REASON]", pred_failure_reason)
                .replace("[CURRENT_STATE]", curr_state)
                .replace("[SUCCESS_CONDITION]", success_condition)
                .replace("[ACTIONS_SO_FAR]", actions_so_far_str)
            ),
        }
        candidates_model, _ = gen_prompter.query(
            prompt=propose_prompt,
            sampling_params=propose_info['params'],
            response_model=CandidateSchema,
        )
        prompts_log.append({
            "call": "plan-propose-thinking",
            "step_index": step_index,
            "prompt": dict(propose_prompt),
            "response": candidates_model.model_dump(),
        })

        candidates = list(candidates_model.candidates[:num_candidates])
        if not candidates:
            break

        print(f"Selecting from candidates...", flush=True)
        labels = candidate_labels[:len(candidates)]
        choice_spec = {lbl: f"Option {lbl}" for lbl in labels}
        choice_spec[done_label] = "Done"
        candidates_text = _format_candidates_for_selection(candidates)
        select_prompt = {
            'system': select_info['template-system'],
            'user': (
                select_info['template-user']
                .replace("[TASK_NAME]", task_name)
                .replace("[PLAN]", initial_plan_str)
                .replace("[FAILURE_REASON]", pred_failure_reason)
                .replace("[SUCCESS_CONDITION]", success_condition)
                .replace("[ACTIONS_SO_FAR]", actions_so_far_str)
                .replace("[CANDIDATES]", candidates_text)
                .replace("[CURRENT_STATE]", curr_state)
            ),
        }
        select_text, select_score = score_prompter.query(
            prompt=select_prompt,
            sampling_params=select_info['params'],
            choice_spec=choice_spec,
        )
        prompts_log.append({
            "call": "plan-select-done",
            "step_index": step_index,
            "prompt": dict(select_prompt),
            "response": select_text,
            "score": select_score,
        })

        selected_label = (select_score or {}).get("predicted_label")
        if selected_label is None:
            from reflect.llm.prompter import extract_choice_label
            selected_label = extract_choice_label(select_text, choice_spec)

        if selected_label == done_label or selected_label not in labels:
            plan_trace.append({
                "step_index": step_index,
                "candidates": [c.model_dump() for c in candidates],
                "selected_label": selected_label,
                "selected_action": None,
                "confidence": (select_score or {}).get("confidence"),
                "entropy": (select_score or {}).get("entropy"),
                "option_probs": (select_score or {}).get("option_probs"),
                "score_status": (select_score or {}).get("score_status"),
                "done": True,
            })
            break

        selected_index = labels.index(selected_label)
        selected_candidate = candidates[selected_index]
        selected_action = ActionPrimitive(
            action=selected_candidate.action,
            obj1=selected_candidate.obj1,
            obj2=selected_candidate.obj2,
        )
        print(f"Executing: ({selected_action.action}, {selected_action.obj1}"
              + (f", {selected_action.obj2}" if selected_action.obj2 else "") + ")", flush=True)

        action_name, params = _normalize_instruction(selected_action)
        taskUtil.chosen_failure = "blocking" if taskUtil.chosen_failure == "blocking" else None
        func = _dispatch_action(action_name)
        # Clamp params to the number of positional args the function accepts
        # (excludes taskUtil, fail_execution, replan) so LLM hallucinated obj2
        # values can never collide with keyword-only arguments.
        _sig_params = [
            p for p in list(_inspect.signature(func).parameters.keys())[1:]
            if p not in ("fail_execution", "replan")
        ]
        params = params[:len(_sig_params)]
        _orig_stdout = sys.stdout
        try:
            sys.stdout = io.StringIO()
            func(taskUtil, *params, fail_execution=False, replan=True)
        finally:
            sys.stdout = _orig_stdout

        local_sg, total_points_dict, obj_held_prev, bbox3d_dict = get_scene_graph(
            last_frame + step_index, controller.last_event, object_list,
            total_points_dict, bbox3d_dict, obj_held_prev, task,
        )
        curr_state = get_scene_text_util(local_sg)

        selected_actions.append(selected_action)
        plan_trace.append({
            "step_index": step_index,
            "candidates": [c.model_dump() for c in candidates],
            "selected_label": selected_label,
            "selected_action": selected_action.model_dump(),
            "curr_state_after": curr_state,
            "confidence": (select_score or {}).get("confidence"),
            "entropy": (select_score or {}).get("entropy"),
            "option_probs": (select_score or {}).get("option_probs"),
            "score_status": (select_score or {}).get("score_status"),
            "done": False,
        })

    success = check_task_success(task['task_idx'], controller.last_event)
    controller.stop()

    selected_plan_raw = [a.model_dump() for a in selected_actions]
    return (
        {
            "task_plan": list(task['actions']),
            "selected_plan": [a.to_legacy_string() for a in selected_actions],
            "selected_plan_raw": selected_plan_raw,
            "llm_plan": selected_plan_raw,
            "llm_plan_raw": selected_plan_raw,
            "num_steps": len(selected_actions),
            "plan_trace": plan_trace,
            "success": success,
        },
        prompts_log,
    )


def reason_about_failure(
    *,
    task,
    task_name,
    llm_prompter,
    global_sg,
    prompt_info,
    L1_captions,
    L2_captions,
    selected_caption=None,
):
    """LLM-driven failure reasoning + root-cause extraction.

    If ``selected_caption`` is provided (the first L2 caption flagged by
    ``subgoal-verifier``), runs execution-grounded reasoning. Otherwise runs
    plan-level reasoning - used by the runtime's Gate B when every subgoal
    verifier passed but ``task-verifier`` reports the plan failed overall.

    Returns ``(reasoning_partial, prompts_log)`` where ``reasoning_partial``
    contains ``pred_failure_reason``, ``pred_failure_step``, ``error_type``.
    """
    reasoning_partial = {}
    prompts_log = []
    prompt = {}
    L1_orig = list(L1_captions)
    L2_orig = list(L2_captions)

    if selected_caption:
        step_name = selected_caption.split(".")[0]
        for caption in L1_orig:
            if step_name not in caption:
                continue

            action = caption.split(". ")[1].split(": ")[1].lower()
            prev_observations = get_robot_plan_mem_local(L2_orig, L1_orig, step=step_name, with_obs=True)
            prompt_name = 'reasoning-execution' if prev_observations else 'reasoning-execution-no-history'
            re_info = prompt_info[prompt_name]

            obs_start = caption.find("Action")
            prompt['system'] = re_info['template-system']
            prompt['user'] = (re_info['template-user']
                              .replace("[ACTION]", action)
                              .replace("[TASK_NAME]", task_name if task_name else task['name'])
                              .replace("[STEP]", step_name)
                              .replace("[SUMMARY]", prev_observations)
                              .replace("[OBSERVATION]", caption[obs_start:]))
            ans, _ = llm_prompter.query(prompt=prompt, sampling_params=re_info['params'])
            prompts_log.append({"call": prompt_name, "prompt": dict(prompt), "response": ans})
            reasoning_partial['pred_failure_reason'] = ans

            res_info = prompt_info['reasoning-execution-steps']
            prompt = {
                'system': res_info['template-system'],
                'user': res_info['template-user'].replace("[FAILURE_REASON]", ans),
            }
            time_steps, _ = llm_prompter.query(prompt=prompt, sampling_params=res_info['params'])
            prompts_log.append({"call": "reasoning-execution-steps", "prompt": dict(prompt), "response": time_steps})
            reasoning_partial['pred_failure_step'] = [ts.replace(",", "") for ts in time_steps.split(", ")]
            reasoning_partial['error_type'] = 'execution'
            break
    else:
        rp_info = prompt_info['reasoning-plan']
        from reflect.core.utils import get_scene_text_util
        current_state = (get_scene_text_util(global_sg) if global_sg else "") or _latest_visual_observation(L1_orig)
        prompt['system'] = rp_info['template-system']
        prompt['user'] = (rp_info['template-user']
                          .replace("[TASK_NAME]", task['name'])
                          .replace("[SUCCESS_CONDITION]", task['success_condition'])
                          .replace("[CURRENT_STATE]", current_state)
                          .replace("[OBSERVATION]", get_robot_plan_mem_local(L2_orig, L1_orig, step=None, with_obs=False)))
        ans, _ = llm_prompter.query(prompt=prompt, sampling_params=rp_info['params'])
        prompts_log.append({"call": "reasoning-plan", "prompt": dict(prompt), "response": ans})
        reasoning_partial['pred_failure_reason'] = ans

        rps_info = prompt_info['reasoning-plan-steps']
        prompt['system'] = rps_info['template-system']
        prompt['user'] = rps_info['template-user'].replace("[PREV_PROMPT]", prompt['user'] + " " + ans)
        step, _ = llm_prompter.query(prompt=prompt, sampling_params=rps_info['params'])
        prompts_log.append({"call": "reasoning-plan-steps", "prompt": dict(prompt), "response": step})
        reasoning_partial['pred_failure_step'] = [step.split(" ")[0].rstrip('.,')]
        reasoning_partial['error_type'] = 'planning'

    return reasoning_partial, prompts_log


def run_reasoning_mem(
    task,
    llm_prompter,
    global_sg,
    prompt_info,
    L2_captions,
    L1_captions,
    uncertainty_metric="entropy",
    data_path=None,
    task_name=None,
    episode_name=None,
):
    reasoning_dict = {}
    prompts_log = []   # list of {call, prompt, response} dicts

    L2_orig = list(L2_captions)
    L1_orig = list(L1_captions)

    selected_caption = ""
    prompt = {}
    sv_info = prompt_info['subgoal-verifier']   # cache sub-dict
    detector_trace = []
    oracle_failure_index = resolve_oracle_failure_index(task, L2_orig)
    sampling_params = dict(sv_info.get('params', {}))

    for idx, caption in enumerate(L2_orig):
        subgoal = caption.split(". ")[1].split(": ")[1].lower()
        obs_start = caption.find("Visual observation")
        step_name = caption.split(".")[0]
        oracle_label, evaluation_active = oracle_label_for_subgoal(task, idx, oracle_failure_index)

        prompt['system'] = sv_info['template-system']
        prompt['user'] = (sv_info['template-user']
                          .replace("[SUBGOAL]", subgoal)
                          .replace("[OBSERVATION]", caption[obs_start:]))

        ans, score_metadata = llm_prompter.query(
            prompt=prompt,
            sampling_params=sampling_params,
            choice_spec=SUBGOAL_VERIFIER_CHOICES,
        )
        predicted_label, predicted_success = parse_verifier_result(ans, score_metadata)

        unc_value = uncertainty_value(score_metadata, uncertainty_metric)

        trace_entry = {
            "step": step_name,
            "subgoal": subgoal,
            "predicted_label": predicted_label,
            "predicted_success": predicted_success,
            "oracle_success": oracle_label,
            "evaluation_active": evaluation_active,
            "score": score_metadata,
            "uncertainty_metric": uncertainty_metric,
            "uncertainty_value": unc_value,
            "response_text": ans,
        }
        detector_trace.append(trace_entry)
        prompts_log.append({
            "call": "subgoal-verifier",
            "prompt": dict(prompt),
            "response": ans,
            "score": score_metadata,
            "oracle_success": oracle_label,
            "evaluation_active": evaluation_active,
        })

    for entry, caption in zip(detector_trace, L2_orig):
        predicted_success = entry["predicted_success"]
        if predicted_success is None:
            continue
        if not predicted_success:
            selected_caption = caption
            break

    reasoning_partial, reasoning_prompts = reason_about_failure(
        task=task,
        task_name=task_name,
        llm_prompter=llm_prompter,
        global_sg=global_sg,
        prompt_info=prompt_info,
        L1_captions=L1_orig,
        L2_captions=L2_orig,
        selected_caption=selected_caption or None,
    )
    reasoning_dict.update(reasoning_partial)
    prompts_log.extend(reasoning_prompts)

    reasoning_dict['gt_failure_reason'] = task['gt_failure_reason']
    reasoning_dict['gt_failure_step'] = task['gt_failure_step']
    reasoning_dict['gt_error_type'] = 'planning' if task.get('chosen_failure') else 'execution'
    reasoning_dict['detector_trace'] = detector_trace
    reasoning_dict['uncertainty_metric'] = uncertainty_metric

    return reasoning_dict, prompts_log
