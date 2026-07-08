import os
import io
import sys
import pickle
import json
import shutil

import reflect.sim.actions as _sim_actions
from reflect.core.utils import convert_timestep_to_step
from reflect.sim.task_manager import TaskUtil, generate_video, make_controller
from reflect.core.paths import sim_output_root


_NULL_OBJ2 = {"", "null", "none", "false", "None", "False", "NULL"}

def _normalize_instruction(instr):
    """Parse any action representation to (action_name, [params])."""
    if isinstance(instr, str):
        parts = [item.strip("() ") for item in instr.split(',')]
        return parts[0], parts[1:]
    if isinstance(instr, dict):
        params = [instr["obj1"]]
        obj2 = instr.get("obj2")
        if obj2 is not None and str(obj2) not in _NULL_OBJ2:
            params.append(obj2)
        return instr["action"], params
    params = [instr.obj1]
    obj2 = getattr(instr, "obj2", None)
    if obj2 is not None and str(obj2) not in _NULL_OBJ2:
        params.append(obj2)
    return instr.action, params


def _clear_recovery_artifacts(repo_path, folder_name):
    """Remove stale recovery outputs for a single episode before replaying."""
    recovery_dir = os.path.join(repo_path, "recovery", folder_name)
    if os.path.isdir(recovery_dir):
        shutil.rmtree(recovery_dir)


def _restore_sim_state(controller, task, final_event, dropped_event=None):
    """Restore the simulator to the final state of a failed episode.

    dropped_event: event at the drop step (disk-based path); defaults to final_event.
    """
    objs = list(final_event.metadata['objects'])
    final_agent = final_event.metadata['agent']

    controller.step(
        action="Teleport",
        position=final_agent['position'],
        rotation=final_agent['rotation'],
        horizon=final_agent['cameraHorizon'],
        standing=final_agent['isStanding'],
        forceAction=True,
    )

    objectPoses = []
    dropped_obj_type = ""

    if "Dropped" == task['gt_failure_reason'].split(" ")[0]:
        dropped_obj_type = task['gt_failure_reason'].split(" ")[1]
        src_event = dropped_event if dropped_event is not None else final_event
        dropped_obj = next(o for o in src_event.metadata["objects"] if o["objectType"] == dropped_obj_type)
        objectPoses.append({
            'objectName': dropped_obj['name'],
            'position': dropped_obj['position'],
            'rotation': dropped_obj['rotation'],
        })

    for obj in objs:
        if 'Sliced' in obj['objectType']:
            org_obj_type = obj['objectType'][:obj['objectType'].find('Sliced')]
            org_obj = next(o for o in controller.last_event.metadata["objects"] if o["objectType"] == org_obj_type)
            if not org_obj['isSliced']:
                controller.step(action="SliceObject", objectId=org_obj['objectId'], forceAction=True)
        elif 'Cracked' in obj['objectType']:
            org_obj_type = obj['objectType'][:obj['objectType'].find('Cracked')]
            org_obj = next(o for o in controller.last_event.metadata["objects"] if o["objectType"] == org_obj_type)
            if not org_obj['isBroken']:
                controller.step(action="BreakObject", objectId=org_obj['objectId'], forceAction=True)

    sim_id_by_name = {o["name"]: o["objectId"] for o in controller.last_event.metadata["objects"]}
    for obj in objs:
        obj_id = sim_id_by_name[obj['name']]
        if obj['isOpen']:
            controller.step(action="OpenObject", objectId=obj_id, forceAction=True)
        else:
            controller.step(action="CloseObject", objectId=obj_id, forceAction=True)
        if obj['isToggled']:
            controller.step(action="ToggleObjectOn", objectId=obj_id, forceAction=True)
        else:
            controller.step(action="ToggleObjectOff", objectId=obj_id, forceAction=True)
        if obj['isFilledWithLiquid']:
            controller.step(
                action="FillObjectWithLiquid",
                objectId=obj_id,
                fillLiquid=obj['fillLiquid'],
                forceAction=True,
            )
        if obj['isDirty']:
            controller.step(action="DirtyObject", objectId=obj_id, forceAction=True)
        if obj['objectType'] != dropped_obj_type:
            if not obj['pickupable'] and not obj['moveable']:
                continue
            objectPoses.append({'objectName': obj['name'], 'position': obj['position'], 'rotation': obj['rotation']})

    controller.step(action='SetObjectPoses', objectPoses=objectPoses)
    controller.step(action="Done")

    sim_id_by_name_final = {o["name"]: o["objectId"] for o in controller.last_event.metadata["objects"]}
    for obj in objs:
        if obj['isPickedUp']:
            controller.step(action="PickupObject", objectId=sim_id_by_name_final[obj['name']], forceAction=True)


_ACTION_ALIASES = {
    "crack":    "crack_obj",
    "open":     "open_obj",
    "close":    "close_obj",
    "slice":    "slice_obj",
    "navigate": "navigate_to_obj",
}

def _dispatch_action(action_name: str):
    resolved = _ACTION_ALIASES.get(action_name, action_name)
    func = getattr(_sim_actions, resolved, None)
    if func is None:
        raise KeyError(f"Unknown action: {action_name!r}")
    return func


def execute_correction_plan(task_idx, taskUtil, output_dir):
    from reflect.core.utils import check_task_success
    with open(os.path.join(output_dir, 'replan.json'), 'r') as f:
        replan_json = json.load(f)
        plan = replan_json.get("llm_plan", replan_json.get("plan", []))
        for instr in plan:
            action, params = _normalize_instruction(instr)
            taskUtil.chosen_failure = "blocking" if taskUtil.chosen_failure == "blocking" else None
            print("action, params: ", action, params)
            func = _dispatch_action(action)
            func(taskUtil, *params, fail_execution=False, replan=True)

    is_success = check_task_success(task_idx, taskUtil.controller.last_event)
    print("Task success :-)" if is_success else "Task fail :-(")
    return is_success


def execute_correction_plan_mem(task_idx, taskUtil, plan):
    """In-memory equivalent of execute_correction_plan(): takes the plan list directly."""
    from reflect.core.utils import check_task_success
    for instr in plan:
        action, params = _normalize_instruction(instr)
        taskUtil.chosen_failure = "blocking" if taskUtil.chosen_failure == "blocking" else None
        try:
            print("action, params: ", action, params)
        except (ValueError, OSError):
            pass
        func = _dispatch_action(action)
        _orig_stdout = sys.stdout
        try:
            sys.stdout = io.StringIO()
            func(taskUtil, *params, fail_execution=False, replan=True)
        finally:
            sys.stdout = _orig_stdout

    is_success = check_task_success(task_idx, taskUtil.controller.last_event)
    try:
        print("Task success :-)" if is_success else "Task fail :-(")
    except (ValueError, OSError):
        pass
    return is_success


def run_correction(data_path, output_dir):
    _td = data_path
    _ss = output_dir
    with open(os.path.join(_td, 'task.json')) as f:
        task = json.load(f)
    folder_name = os.path.join(
        os.path.basename(os.path.dirname(os.path.abspath(data_path))),
        os.path.basename(os.path.abspath(data_path)),
    )
    runtime_root = str(sim_output_root())
    _clear_recovery_artifacts(runtime_root, folder_name)

    controller = make_controller(task['scene'])

    events_path = os.path.join(_td, 'events')
    lsorted = sorted(
        os.listdir(events_path),
        key=lambda x: int(os.path.splitext(x)[0].split('_')[-1])
    )
    last_frame = int(lsorted[-1].split('_')[-1].split('.')[0])
    with open(os.path.join(events_path, lsorted[-1]), 'rb') as f:
        final_event = pickle.load(f)

    # Load the event at the drop step (needed to get the pre-fall object pose)
    dropped_event = None
    if "Dropped" == task['gt_failure_reason'].split(" ")[0]:
        dropped_step = convert_timestep_to_step(task['gt_failure_step'], video_fps=1)
        with open(os.path.join(_td, 'events', f'step_{dropped_step}.pickle'), 'rb') as f:
            dropped_event = pickle.load(f)

    _restore_sim_state(controller, task, final_event, dropped_event=dropped_event)

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
    is_success = execute_correction_plan(task['task_idx'], taskUtil, output_dir)

    with open(os.path.join(_ss, 'replan.json'), 'r') as f:
        replan_json = json.load(f)
    replan_json["success"] = is_success
    with open(os.path.join(_ss, 'replan.json'), 'w') as f:
        json.dump(replan_json, f)

    generate_video(taskUtil, recovery_video=True)
    controller.stop()


def run_correction_mem(data_path, task, final_event, last_frame, replan_dict):
    """
    In-memory equivalent of run_correction().

    Parameters
    ----------
    data_path   : str   - episode data directory (used for TaskUtil folder paths)
    task        : dict  - already-loaded task.json contents
    final_event : AI2-THOR event  - last saved event of the failed execution
    last_frame  : int   - frame index corresponding to final_event
    replan_dict : dict  - replan_dict returned by generate_replan_mem
                          (keys: task_plan, llm_plan_raw, llm_plan, num_steps)

    Returns
    -------
    replan_dict : dict  - same dict with 'success' key added
    """
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

    executable_plan = replan_dict.get('llm_plan', replan_dict.get('plan', []))
    is_success = execute_correction_plan_mem(task['task_idx'], taskUtil, executable_plan)

    replan_dict = dict(replan_dict)
    replan_dict['success'] = is_success

    generate_video(taskUtil, recovery_video=True)
    controller.stop()

    return replan_dict
