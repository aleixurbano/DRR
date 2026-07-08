"""
gen_data.py
-----------
Generates synthetic training data for robotic task execution using the AI2-THOR
simulation environment. Supports controlled failure injection (missing steps,
failed actions, dropped objects, blocking/occupied objects, wrong perception)
so that downstream models can learn to detect and recover from task failures.

Main entry point: run_data_gen(data_path, task)
"""

import os
import json
import numpy as np
import random
import pickle

import reflect.sim.actions as action_primitives
from reflect.sim.task_manager import TaskUtil, generate_video, save_data, make_controller
from reflect.core.constants import TASK_DICT
from reflect.core.utils import convert_step_to_timestep


def _parse_instruction(instr: str):
    """Parse an action instruction string to (action_name, [params])."""
    parts = [item.strip("() ") for item in instr.split(',')]
    return parts[0], parts[1:]


def _resolve_action(name):
    """Look up an action function by name from action_primitives."""
    func = getattr(action_primitives, name, None)
    if func is not None:
        return func
    raise KeyError(f"Unknown action: {name!r}")


def flatten_list(lis):
    output = []
    for item in lis:
        if isinstance(item, list):
            output.extend(item)
        else:
            output.append(item)
    return output


def get_failure_injection_idx(taskUtil, actions, task, action_idxs, nav_idxs, interact_cnt=0, nav_cnt=0):
    """Choose the action index at which a synthetic failure should be injected.

    Returns the injection index, or -1 if no valid index could be found.
    """
    counter = 0
    print("[INFO] Injected failures:", taskUtil.failures_already_injected)

    already_used = flatten_list([f[1] for f in taskUtil.failures_already_injected])

    try:
        while True:
            if taskUtil.chosen_failure in ('missing_step', 'failed_action'):
                if taskUtil.chosen_failure == 'missing_step' and "specified_missing_steps" in task:
                    cnt = sum(1 for f in taskUtil.failures_already_injected if f[0] == 'missing_step')
                    if cnt < len(task['specified_missing_steps']):
                        return task['specified_missing_steps'][cnt]

                failure_injection_idx = np.random.choice(action_idxs[interact_cnt:])

                if "toggle_off" in actions[failure_injection_idx] or "close_obj" in actions[failure_injection_idx]:
                    counter += 1
                    continue

                if not already_used or failure_injection_idx not in already_used:
                    return failure_injection_idx

            elif taskUtil.chosen_failure == 'drop':
                failure_injection_idx = np.random.choice(nav_idxs[nav_cnt:])
                return failure_injection_idx

            if counter > 20:
                print(
                    f"[INFO] Unable to inject a novel failure for failure type: "
                    f"{taskUtil.chosen_failure}. Choosing a new failure type"
                )
                taskUtil.chosen_failure = np.random.choice(taskUtil.failures)

            if counter > 60:
                print("[INFO] Unable to inject a novel failure. Skipping this round. Maybe out of failures to inject.")
                return -1

            counter += 1

    except Exception as e:
        print("[INFO] Unable to inject a novel failure. Skipping this round. Maybe out of failures to inject:", e)
        return -1


def run_data_gen(data_path, task):
    """Generate and persist training episodes for a single task configuration."""
    np.random.seed(91)
    random.seed(91)

    os.makedirs(os.path.join('thor_tasks', TASK_DICT[task["task_idx"]]), exist_ok=True)
    with open(f'thor_tasks/{TASK_DICT[task["task_idx"]]}/{task["folder_name"]}.pickle', 'wb') as handle:
        pickle.dump([], handle, protocol=pickle.HIGHEST_PROTOCOL)

    for i in range(int(task['num_samples'])):
        controller = make_controller(task['scene'])
        reachable_positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]

        chosen_failure = task.get('chosen_failure', None)
        failure_injection_params = task.get('failure_injection_params', None)

        taskUtil = TaskUtil(
            folder_name=os.path.join(TASK_DICT[task["task_idx"]], task['folder_name']),
            controller=controller,
            reachable_positions=reachable_positions,
            failure_injection=task['failure_injection'],
            index=i,
            repo_path=data_path,
            chosen_failure=chosen_failure,
            failure_injection_params=failure_injection_params,
        )

        if taskUtil.chosen_failure in ('blocking', 'occupied', 'occupied_put') and 'failure_injection_params' in task:
            action_primitives.place_obj(taskUtil, task['failure_injection_params'])

        if taskUtil.chosen_failure == 'wrong perception' and 'disp_x' in taskUtil.failure_injection_params:
            action_primitives.place_obj(taskUtil, task['failure_injection_params'])

        if "preactions" in task:
            for preaction_instr in task['preactions']:
                preaction, params = _parse_instruction(preaction_instr)
                func = _resolve_action(preaction)
                func(taskUtil, *params)

        instrs = list(task['actions'])
        new_instrs = []
        action_idxs = []
        nav_idxs = []

        for idx, instr in enumerate(instrs):
            action, _ = _parse_instruction(instr)
            if action in taskUtil.INTERACT_ACTION_PRIMITIVES:
                action_idxs.append(idx)
            if action == 'navigate_to_obj':
                nav_idxs.append(idx)

        if task['failure_injection']:
            failure_injection_idx = get_failure_injection_idx(taskUtil, instrs, task, action_idxs, nav_idxs)
            if failure_injection_idx == -1:
                controller.stop()
                continue
            print("[INFO] failure_injection_idx:", failure_injection_idx)

        nav_counter = 0
        interact_counter = 0

        for i, instr in enumerate(instrs):
            action, params = _parse_instruction(instr)
            func = _resolve_action(action)

            if action in taskUtil.INTERACT_ACTION_PRIMITIVES:
                interact_counter += 1
            if action == 'navigate_to_obj':
                nav_counter += 1

            # Determine per-step failure flags
            to_drop = False
            to_drop_injection_idx = 0
            fail_execution = False

            if not taskUtil.failure_added and taskUtil.chosen_failure == 'drop' and i == failure_injection_idx:
                to_drop = True
                to_drop_injection_idx = failure_injection_idx

            if not taskUtil.failure_added and taskUtil.chosen_failure == 'missing_step' and action in taskUtil.INTERACT_ACTION_PRIMITIVES:
                if not isinstance(failure_injection_idx, list):
                    failure_injection_idx = [failure_injection_idx]
                if i in failure_injection_idx:
                    if 'gt_failure_reason' in taskUtil.gt_failure:
                        taskUtil.gt_failure['gt_failure_reason'] += ', ' + instr
                    else:
                        taskUtil.gt_failure['gt_failure_reason'] = 'Missing ' + instr
                    taskUtil.gt_failure['gt_failure_step'] = taskUtil.counter + 1
                    if i == failure_injection_idx[-1]:
                        taskUtil.failure_added = True
                        taskUtil.failures_already_injected.append([taskUtil.chosen_failure, failure_injection_idx])
                    else:
                        taskUtil.failure_added = False
                    continue

            if not taskUtil.failure_added and taskUtil.chosen_failure == 'failed_action' and action in taskUtil.INTERACT_ACTION_PRIMITIVES and i == failure_injection_idx:
                print("[INFO] Injecting failed action...")
                fail_execution = True
                taskUtil.gt_failure['gt_failure_reason'] = 'Failed to successfully execute ' + instr
                taskUtil.gt_failure['gt_failure_step'] = taskUtil.counter + 1
                taskUtil.failures_already_injected.append([taskUtil.chosen_failure, failure_injection_idx])
                taskUtil.failure_added = True

            new_instrs.append(instr)

            # Call the action with explicit keyword flags - no dynamic positional appending
            extra_kwargs = {}
            if fail_execution:
                extra_kwargs['fail_execution'] = True
            if to_drop:
                extra_kwargs['to_drop'] = True
                extra_kwargs['failure_injection_idx'] = to_drop_injection_idx
            retval = func(taskUtil, *params, **extra_kwargs)

            if retval is False:
                failure_injection_idx = get_failure_injection_idx(
                    taskUtil, instrs, task, action_idxs, nav_idxs,
                    interact_cnt=interact_counter, nav_cnt=nav_counter,
                )
                if failure_injection_idx == -1:
                    break

        for _ in range(2):
            e = controller.step(action="Done")
            save_data(taskUtil, e)

        print("[INFO] interact_actions:", taskUtil.interact_actions)
        print("[INFO] nav_actions:", taskUtil.nav_actions)

        with open(f'thor_tasks/{taskUtil.specific_folder_name}/interact_actions.pickle', 'wb') as handle:
            pickle.dump(taskUtil.interact_actions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'thor_tasks/{taskUtil.specific_folder_name}/nav_actions.pickle', 'wb') as handle:
            pickle.dump(taskUtil.nav_actions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'thor_tasks/{TASK_DICT[task["task_idx"]]}/{task["folder_name"]}.pickle', 'wb') as handle:
            pickle.dump(taskUtil.failures_already_injected, handle, protocol=pickle.HIGHEST_PROTOCOL)

        updated_task = task.copy()
        updated_task['specific_folder_name'] = taskUtil.specific_folder_name

        if 'gt_failure_reason' not in taskUtil.gt_failure:
            taskUtil.gt_failure['gt_failure_reason'] = 'No failure added'
            taskUtil.gt_failure['gt_failure_step'] = 0

        if 'gt_failure_reason' not in updated_task:
            updated_task['gt_failure_reason'] = taskUtil.gt_failure['gt_failure_reason']
            updated_task['gt_failure_step'] = convert_step_to_timestep(
                taskUtil.gt_failure['gt_failure_step'], video_fps=1
            )

        updated_task['unity_name_map'] = taskUtil.unity_name_map
        updated_task['sounds'] = taskUtil.sounds
        updated_task['actions'] = new_instrs

        with open(f'thor_tasks/{taskUtil.specific_folder_name}/task.json', 'w') as f:
            json.dump(updated_task, f)

        generate_video(taskUtil, recovery_video=False)
        controller.stop()
