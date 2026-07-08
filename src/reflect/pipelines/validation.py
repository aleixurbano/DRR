import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import pickle
from pathlib import Path
from reflect.core.constants import *
from reflect.perception.scene_graph import SceneGraph
from reflect.perception.scene_graph import Node as SceneGraphNode
from reflect.perception.local_graph import get_scene_graph
import numpy as np
from reflect.core.data import *
from reflect.core.utils import *
from reflect.perception.clip import *
from reflect.perception.audio import process_sound, audio2label
from reflect.perception.point_cloud import *
from reflect.core.episode_store import EpisodeArtifactStore

def run_sound_module(data_path, object_list):
    detected_sounds = []
    try:
        detected_sounds = process_sound(data_path, object_list)
        print("detected sounds:", detected_sounds)
    except Exception as e:
        print(e)
    return detected_sounds


def get_scene_text(scene_graph):
    output = ""
    visited = []
    for node in set(scene_graph.nodes):
        output += (node.get_name() + ", ")

    if len(output) != 0:
        output = output[:-2] + ". "
    for edge_key, edge in scene_graph.edges.items():
        start_name, end_name = edge_key
        edge_key_2 = (end_name, start_name)
        if (edge_key not in visited and edge_key_2 not in visited):
            output += edge.start.name + " is " + edge.edge_type + " " + edge.end.name
            output += ". "
        visited.append(edge_key)
    output = output[:-1]

    return output


def get_held_object(output_dir, step_idx):
    store = EpisodeArtifactStore(Path(output_dir).parent, Path(output_dir))
    found = False
    while not found:
        _sg_path = store.local_graph_dir / f'local_sg_{step_idx}.pkl'
        if _sg_path.exists():
            found = True
            with open(_sg_path, 'rb') as f:
                local_sg = pickle.load(f)
            for key in local_sg.edges:
                if "robot gripper" in key and key[0] != "nothing":
                    return key[0]
        else:
            step_idx -= 1


def generate_scene_graphs(data_path, events, object_list, nav_actions, interact_actions, WITH_AUDIO, detected_sounds, output_dir):
    store = EpisodeArtifactStore(Path(data_path), Path(output_dir))
    task = store.load_task()

    if not store.global_scene_graph_path.exists():
        # sensory-input summary
        store.ensure_local_graph_dir()
        # os.system("mkdir -p scene/{}".format(folder_name))
        key_frames = []
        prev_graph = SceneGraph(event=None, task=task)
        total_points_dict, bbox3d_dict = {}, {}
        obj_held_prev = None
        cnt, interval = 0, 2
        nav_actions_end_indices = [idx[1] for idx in nav_actions.keys()]
        for step_idx, event in enumerate(events):
            # uniformly drop intermediate navigation frames with no sound
            if (step_idx+1) not in interact_actions and ((step_idx+1) not in nav_actions_end_indices):
                cnt += 1
                if WITH_AUDIO == 1:
                    if step_idx not in detected_sounds and cnt % interval == 0:
                        continue
                elif WITH_AUDIO == 0:
                    if str(step_idx) not in task['sounds'] and cnt % interval == 0:
                        continue

            print("[Frame] " + str(step_idx+1))

            local_sg, total_points_dict, obj_held_prev, bbox3d_dict = get_scene_graph(step_idx, event, object_list, 
                                                                                      total_points_dict, bbox3d_dict, 
                                                                                      obj_held_prev, task)
            print("========================[Current Graph]=====================")
            print(local_sg)

            # 1. Select keyframe based on scene graph difference
            if local_sg != prev_graph:
                if (step_idx+1) not in key_frames:
                    key_frames.append(step_idx+1)
                    prev_graph = local_sg

            # 2. Select keyframe based on actions
            if (step_idx+1) in interact_actions or (step_idx+1) in nav_actions_end_indices:
                if (step_idx+1) not in key_frames:
                    key_frames.append(step_idx+1)

            # 3. Select keyframe based on audio
            if WITH_AUDIO == 0:
                if str(step_idx) in task['sounds']:
                    if (step_idx+1) not in key_frames:
                        key_frames.append(step_idx+1)
            elif WITH_AUDIO == 1:
                if step_idx in detected_sounds:
                    if (step_idx+1) not in key_frames:
                        key_frames.append(step_idx+1)

            store.save_pickle(store.local_graph_dir / f'local_sg_{step_idx}.pkl', local_sg)

        store.save_key_frames(key_frames)

        # save_pcd(folder_name, total_points_dict)

        # ======================Get global graph========================
        global_sg = SceneGraph(events[-1], task)
        for label in total_points_dict.keys():
            name = get_label_from_object_id(label, events, task)
            if name is not None:
                new_node = SceneGraphNode(name=name, object_id=label, pos3d=bbox3d_dict[label].get_center(), 
                        corner_pts=np.array(bbox3d_dict[label].get_box_points()),
                        pcd=total_points_dict[label], global_node=True)
                global_sg.add_node_wo_edge(new_node)

        for label in total_points_dict.keys():
            object_name = label.split("|")[0]
            if object_name in object_list:
                name = get_label_from_object_id(label, events, task)
                if name is not None:
                    for node in global_sg.total_nodes:
                        if node.name == name:
                            global_sg.add_node(node)
            
        global_sg.add_agent()
        store.save_pickle(store.global_scene_graph_path, global_sg)
        # ===============================================================


def generate_summary(data_path, events, nav_actions, interact_actions, WITH_AUDIO, detected_sounds, output_dir):
    store = EpisodeArtifactStore(Path(data_path), Path(output_dir))
    task = store.load_task()

    key_frames = store.load_key_frames()

    # event-based summary
    if not store.summary_l1_path.exists():
    # if True:
        print("[INFO] Start generating event-based summary")
        state_summary_L1 = ""
        L1_captions = []
        for step_idx, event in enumerate(events):
            if not (store.local_graph_dir / f'local_sg_{step_idx}.pkl').exists():
                continue
            if (step_idx+1) in key_frames:
                caption = ""

                # add action
                if (step_idx+1) in interact_actions:
                    caption += f"{convert_step_to_timestep(step=step_idx+1, video_fps=1)}. Action: {interact_actions[step_idx+1]}."
                    # action = interact_actions[step_idx+1]
                else:
                    for key in nav_actions:
                        min_step, max_step = key
                        if min_step <= (step_idx+1) <= max_step:
                            caption += f"{convert_step_to_timestep(step=step_idx+1, video_fps=1)}. Action: {nav_actions[key]}."
                            # action = nav_actions[key]

                if len(caption) == 0:
                    continue
                
                with open(store.local_graph_dir / f'local_sg_{step_idx}.pkl', 'rb') as f:
                    local_sg = pickle.load(f)
                    scene_text = get_scene_text(local_sg)
                    caption += f" Visual observation: {scene_text}"

                # Add audio info.
                if WITH_AUDIO == 0:
                    if str(step_idx) in task['sounds']:
                        if 'drop' in task['sounds'][str(step_idx)] and get_held_object(output_dir, step_idx-1) is not None:
                            caption += f" Auditory observation: something drops."
                        else:
                            caption += f" Auditory observation: {audio2label[task['sounds'][str(step_idx)]]}."
                elif WITH_AUDIO == 1:
                    if step_idx in detected_sounds:
                        caption += f" Auditory observation: {detected_sounds[step_idx]}."

                caption += "\n"

                state_summary_L1 += caption
                L1_captions.append(caption)
        store.save_text(store.summary_l1_path, state_summary_L1)
        print("[INFO] Write event-based summary")
    else:
        print("[INFO] Event-based summary already generated")
        L1_captions = []
        state_summary_L1 = ""
        with open(store.summary_l1_path, 'r') as f:
            L1_captions = f.readlines()
        state_summary_L1 = "".join(L1_captions)

    # subgoal-based summary
    if not store.summary_l2_path.exists():
    # if True:
        print("[INFO] Start generating subgoal-based summary")
        L2_captions = []
        for caption in L1_captions:
            step_num = convert_timestep_to_step(caption.split(".")[0], video_fps=1)
            if step_num in interact_actions:
                L2_captions.append(caption.replace("Action", "Goal"))

        state_summary_L2 = "".join(L2_captions)
        store.save_text(store.summary_l2_path, state_summary_L2)
        print("[INFO] Write subgoal-based summary")
    else:
        print("[INFO] Subgoal-based summary already generated")
        L2_captions = []
        L2_file_name = store.summary_l2_path
        if os.path.exists(L2_file_name):
            with open(L2_file_name, 'r') as f:
                L2_captions = f.readlines()
            state_summary_L2 = "".join(L2_captions)


def run_reasoning(data_path, llm_prompter, global_sg, output_dir, llm_dir):
    store = EpisodeArtifactStore(Path(data_path), Path(output_dir), Path(llm_dir))
    task = store.load_task()
    
    if store.reasoning_path.exists():
        print("[INFO] Reasoning already generated")
        with open(store.reasoning_path, 'r') as f:
            reasoning_dict = json.load(f)
        return
    else:
        reasoning_dict = {}

    save_dir = store.ensure_episode_llm_dir()

    with open(Path(llm_dir) / 'prompts.json', 'r') as f:
        prompt_info = json.load(f)

    # Load L2 captions from state_summary_L2.txt
    with open(store.summary_l2_path, 'r') as f:
        L2_captions = f.readlines()

    # Load L1 captions from state_summary_L1.txt
    with open(store.summary_l1_path, 'r') as f:
        L1_captions = f.readlines()
    
    # Loop through each subgoal and check for post-condition
    print(">>> Run step-by-step subgoal-level analysis...")
    selected_caption = ""
    prompt = {}

    for caption in L2_captions:
        print(">>> Verify subgoal...")
        subgoal = caption.split(". ")[1].split(": ")[1].lower()

        prompt['system'] = prompt_info['subgoal-verifier']['template-system']
        prompt['user'] = prompt_info['subgoal-verifier']['template-user'].replace("[SUBGOAL]", subgoal).replace("[OBSERVATION]", caption[caption.find("Visual observation"):])

        ans, _  = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['subgoal-verifier']['params'], 
                                    save=prompt_info['subgoal-verifier']['save'], save_dir=save_dir)
        is_success = int(ans.split(", ")[0] == "Yes")
        if is_success == 0:
            selected_caption = caption
            print(f"[INFO] Failure identified in subgoal [{subgoal}] at {caption.split('.')[0]}")
            break
        else:
            print(f"[INFO] Subgoal [{subgoal}] succeeded!")

    if len(selected_caption) != 0:
            print(">>> Get detailed reasoning from L1...")
            step_name = selected_caption.split(".")[0]
            for _, caption in enumerate(L1_captions):
                if step_name in caption:
                    action = caption.split(". ")[1].split(": ")[1].lower()
                    prev_observations = get_robot_plan(output_dir, step=step_name, with_obs=True)
                    if len(prev_observations) != 0:
                        prompt_name = 'reasoning-execution'
                    else:
                        prompt_name = 'reasoning-execution-no-history'
                    prompt['system'] = prompt_info[prompt_name]['template-system']
                    prompt['user'] = prompt_info[prompt_name]['template-user'].replace("[ACTION]", action)
                    prompt['user'] = prompt['user'].replace("[TASK_NAME]", task['name'])
                    prompt['user'] = prompt['user'].replace("[STEP]", step_name)
                    prompt['user'] = prompt['user'].replace("[SUMMARY]", prev_observations)
                    prompt['user'] = prompt['user'].replace("[OBSERVATION]", caption[caption.find("Action"):])
                    ans, _  = llm_prompter.query(prompt=prompt, sampling_params=prompt_info[prompt_name]['params'], 
                                                save=prompt_info[prompt_name]['save'], save_dir=save_dir)

                    print("[INFO] Predicted failure reason:", ans)
                    reasoning_dict['pred_failure_reason'] = ans

                    prompt = {}
                    prompt['system'] = prompt_info['reasoning-execution-steps']['template-system']
                    prompt['user'] = prompt_info['reasoning-execution-steps']['template-user'].replace("[FAILURE_REASON]", ans)
                    time_steps, _ = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['reasoning-execution-steps']['params'],
                                                            save=prompt_info['reasoning-execution-steps']['save'], save_dir=save_dir)
                    
                    print("[INFO] Predicted failure time steps:", time_steps, time_steps.split(", "))
                    reasoning_dict['pred_failure_step'] = [time_step.replace(",", "") for time_step in time_steps.split(", ")]
                    break
    else:
        print(">>> All actions are executed successfully, run plan-level analysis...")

        prompt['system'] = prompt_info['reasoning-plan']['template-system']
        prompt['user'] = prompt_info['reasoning-plan']['template-user'].replace("[TASK_NAME]", task['name'])
        prompt['user'] = prompt['user'].replace("[SUCCESS_CONDITION]", task['success_condition'])
        prompt['user'] = prompt['user'].replace("[CURRENT_STATE]", get_scene_text(global_sg))
        prompt['user'] = prompt['user'].replace("[OBSERVATION]", get_robot_plan(output_dir, step=None, with_obs=False))
        ans, _ = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['reasoning-plan']['params'], 
                                    save=prompt_info['reasoning-plan']['save'], save_dir=save_dir)
        
        print("[INFO] Predicted failure reason:", ans)
        reasoning_dict['pred_failure_reason'] = ans

        prompt['system'] = prompt_info['reasoning-plan-steps']['template-system']
        prompt['user'] = prompt_info['reasoning-plan-steps']['template-user'].replace("[PREV_PROMPT]", prompt['user'] + " " + ans)
        step, _ = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['reasoning-plan-steps']['params'], 
                                    save=prompt_info['reasoning-plan-steps']['save'], save_dir=save_dir)
        step_str = step.split(" ")[0]
        if step_str[-1] == '.' or step_str[-1] == ',':
            step_str = step_str[:-1]

        print("[INFO] Predicted failure time steps:", step_str)
        reasoning_dict['pred_failure_step'] = step_str

    reasoning_dict['gt_failure_reason'] = task['gt_failure_reason']
    reasoning_dict['gt_failure_step'] = task['gt_failure_step']
    
    store.save_json(store.reasoning_path, reasoning_dict, indent=2)


def generate_replan(data_path, llm_prompter, global_sg, last_event, task_object_list, output_dir, llm_dir):
    try:
        from reflect.models.plan import Plan
    except ImportError:
        from reflect.models.plan import Plan

    store = EpisodeArtifactStore(Path(data_path), Path(output_dir), Path(llm_dir))
    task = store.load_task()
    curr_state = get_scene_text(global_sg)
    print("[INFO] Current state:", curr_state)
    global_object_list = list(set([obj["objectType"] for obj in last_event.metadata["objects"]]) | set(task_object_list))

    with open(store.reasoning_path, 'r') as f:
        data = json.load(f)
        reason = data["pred_failure_reason"]

    if store.replan_path.exists():
        print("[INFO] Skipping replan generation")
        with open(store.replan_path, 'r') as f:
            replan_dict = json.load(f)
            plan_actions = replan_dict.get("llm_plan_raw", replan_dict.get("original_plan", []))
            translated_lines = replan_dict.get("llm_plan", replan_dict.get("plan", []))
    else:
        with open(Path(llm_dir) / 'prompts.json', 'r') as f:
            prompt_info = json.load(f)

        prompt = {}
        available_actions = get_admissible_actions(global_object_list, last_event)
        available_objects = sorted({token.strip("() ") for action in available_actions for token in action.split(", ")[1:]})

        prompt['system'] = prompt_info['correction']['template-system'].replace(
            "[PREFIX]",
            get_replan_prefix(available_actions=available_actions, available_objects=available_objects),
        )
        prompt['user'] = prompt_info['correction']['template-user'].replace("[TASK_NAME]", task['name']).replace("[PLAN]", get_initial_plan(task['actions']))
        prompt['user'] = prompt['user'].replace("[FAILURE_REASON]", reason)
        prompt['user'] = prompt['user'].replace("[CURRENT_STATE]", curr_state).replace("[SUCCESS_CONDITION]", task['success_condition'])
    
        # print("=====================RE-PLAN PROMPT START========================")
        # print(prompt['system'])
        # print(prompt['user'])
        # print("=====================RE-PLAN PROMPT END==========================")

        _save_dir = store.ensure_episode_llm_dir()
        plan, _ = llm_prompter.query(
            prompt=prompt,
            sampling_params=prompt_info['correction']['params'],
            save=prompt_info['correction']['save'],
            save_dir=_save_dir,
            response_model=Plan,
        )
        plan_actions = plan.model_dump()["actions"]
        translated_lines = plan_actions

        print("========================Structured plan===========================")
        print(plan_actions)

        replan_dict = {
            "task_plan": list(task['actions']),
            "llm_plan_raw": plan_actions,
            "llm_plan": translated_lines,
            "num_steps": len(translated_lines),
        }

        store.save_json(store.replan_path, replan_dict, indent=4)
