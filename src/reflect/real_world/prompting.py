import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle
import shutil
import numpy as np
import random
import torch
from PIL import Image
from reflect.real_world.detection import get_seg_model
from reflect.real_world.clip_utils import get_clip_model, get_text_feats
from reflect.real_world.scene_graph import Node as SceneGraphNode
from reflect.real_world.scene_graph import SceneGraph
from reflect.real_world.local_graph import get_scene_graph
from reflect.real_world.point_cloud_utils import *
from reflect.real_world.shared_utils import *
from argparse import ArgumentParser
import zarr
import json
from imagecodecs import imread
from reflect.real_world.logging_utils import TaskProfiler, configure_logging, get_logger
from reflect.real_world.utils import get_robot_plan
from reflect.llm.prompter import LocalLLMPrompter
from AudioCLIP.real_world_audio import get_sound_events

np.random.seed(91)
random.seed(91)

llm_prompter = LocalLLMPrompter()
device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
logger = get_logger(__name__)


def _get_video_root(folder_name):
    base = f"real_world/data/{folder_name}/videos"
    legacy = f"{base}/0/0"
    return legacy if os.path.exists(legacy) else base


def _get_summary_file_names(args):
    if args.ablation_type == 0 and args.audio_ver == 0:
        return "state_summary_L1_wo_sound.txt", "state_summary_L2_wo_sound.txt"
    if args.ablation_type == 5:
        return "state_summary_L1_BLIP2.txt", "state_summary_L2_BLIP2.txt"
    return "state_summary_L1.txt", "state_summary_L2.txt"


def _scene_graph_overview(scene_graph):
    object_names = [node.get_name() for node in scene_graph.nodes]
    return f"nodes={len(scene_graph.nodes)} edges={len(scene_graph.edges)} objects={object_names}"


def _read_text_if_exists(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r") as f:
        return f.read()


def _local_graph_artifact_stats(folder_name):
    graph_dir = f"real_world/state_summary/{folder_name}/local_graphs"
    stats = []
    if not os.path.exists(graph_dir):
        return stats
    for name in sorted(os.listdir(graph_dir), key=lambda n: int(n.split("_")[-1].split(".")[0])):
        if not name.startswith("local_sg_") or not name.endswith(".pkl"):
            continue
        path = os.path.join(graph_dir, name)
        stats.append(
            {
                "name": name,
                "path": path,
                "size": os.path.getsize(path),
                "mtime": os.path.getmtime(path),
            }
        )
    return stats


def _assert_nonempty_graph_rebuild(folder_name):
    stats = _local_graph_artifact_stats(folder_name)
    if not stats:
        raise RuntimeError(
            f"Scene-graph rebuild for '{folder_name}' did not write any local_sg_*.pkl files."
        )

    all_empty_pickles = all(item["size"] <= 96 for item in stats)
    if all_empty_pickles:
        details = ", ".join(f"{item['name']}={item['size']}B" for item in stats)
        raise RuntimeError(
            f"Scene-graph rebuild for '{folder_name}' only produced empty graph pickles: {details}. "
            "This usually means the current Python process did not actually run the patched graph-building code "
            "or the run exited before writing fresh graphs."
        )


def _attach_summaries_and_trace(args, reasoning_dict):
    l1_name, l2_name = _get_summary_file_names(args)
    summary_dir = f"real_world/state_summary/{args.folder_name}"
    l1_path = f"{summary_dir}/{l1_name}"
    l2_path = f"{summary_dir}/{l2_name}"

    reasoning_dict["l1_summary_file"] = l1_path
    reasoning_dict["l2_summary_file"] = l2_path
    reasoning_dict["l1_summary"] = _read_text_if_exists(l1_path)
    reasoning_dict["l2_summary"] = _read_text_if_exists(l2_path)

    llm_response_path = f"LLM/{args.folder_name}/response.json"
    llm_trace_path = f"{summary_dir}/llm_trace.json"
    trace_payload = {}
    if os.path.exists(llm_response_path):
        with open(llm_response_path, "r") as f:
            trace_payload = json.load(f)

    sorted_trace_payload = {
        key: trace_payload[key] for key in sorted(trace_payload.keys())
    }
    with open(llm_trace_path, "w") as f:
        json.dump(sorted_trace_payload, f, indent=2)

    reasoning_dict["llm_trace_file"] = llm_trace_path
    reasoning_dict["llm_trace_count"] = len(sorted_trace_payload)
    return reasoning_dict


def _summary_dir(folder_name):
    return f"real_world/state_summary/{folder_name}"


def _timings_path(folder_name):
    return f"{_summary_dir(folder_name)}/timings.json"


def _write_full_artifacts(args):
    return getattr(args, "artifact_level", "full") == "full"


def _make_task_profiler(args):
    return TaskProfiler(
        enabled=getattr(args, "profile", False),
        output_path=_timings_path(args.folder_name),
        task_name=args.folder_name,
    )


def _has_cached_detection_outputs(folder_name, obj_det):
    if obj_det == "mdetr":
        cache_dir = f"{_summary_dir(folder_name)}/{obj_det}_obj_det/clip_processed_det"
    else:
        cache_dir = f"{_summary_dir(folder_name)}/{obj_det}_obj_det/det"
    return os.path.isdir(cache_dir) and any(name.endswith(".pickle") for name in os.listdir(cache_dir))

def get_scene_text(scene_graph):
    output = ""
    for node in scene_graph.nodes:
        node_name = node.name
        if node.state is not None:
            node_name = f"{node_name} ({node.state})"
        output += (node_name + ", ")
    if len(scene_graph.nodes) != 0:
        output = output[:-2] + ". "
    for edge in scene_graph.edges.values():
        # filter out redundant relations
        start_node_name = str(edge.start) 
        end_node_name = str(edge.end)
        if edge.edge_type == 'on the left of':
            if (end_node_name, start_node_name) in scene_graph.edges:
                other_edge = scene_graph.edges[(end_node_name, start_node_name)]
                if other_edge.edge_type == 'on the right of':
                    continue
        if edge.edge_type == 'below':
            if (end_node_name, start_node_name) in scene_graph.edges:
                other_edge = scene_graph.edges[(end_node_name, start_node_name)]
                if other_edge.edge_type == 'above':
                    continue
        if edge.edge_type == 'near':
            if (end_node_name, start_node_name) in scene_graph.edges:
                other_edge = scene_graph.edges[(end_node_name, start_node_name)]
                if other_edge.edge_type == 'on top of' or other_edge.edge_type == 'inside' or other_edge.edge_type == 'near':
                    continue
        output += (start_node_name + " is " + edge.edge_type + " " + end_node_name)
        output += ". "
    output = output[:-1]

    return output

def save_L1_images(args, task_info):
    args.folder_name = task_info["general_folder_name"]
    os.system('mkdir -p real_world/images/{}'.format(args.folder_name))
    video_root = _get_video_root(args.folder_name)
    key_frames = []
    with open('real_world/state_summary/{}/L1_key_frames.txt'.format(args.folder_name), 'r') as f:
        frames = f.readlines()
        key_frames = [int(frame) for frame in frames]
    key_frames = key_frames[1:] # remove the first frame
    for step_idx in key_frames:
        rgb = imread(f'{video_root}/color/{step_idx}.0.0.0')
        depth = imread(f'{video_root}/depth/{step_idx}.0.0')
        # depth = np.clip(depth, 0, 1000)
        # plt.matshow(depth,cmap=plt.cm.jet,interpolation='bicubic')
        # plt.axis('off')
        im = Image.fromarray(rgb)
        im.save('real_world/images/{}/rgb_{}.png'.format(args.folder_name, step_idx))
        # plt.savefig('real_world/images/{}/depth_{}.png'.format(args.folder_name, step_idx))


def run_wo_sound(args, task_info):
    args.folder_name = task_info["general_folder_name"]

    if not os.path.exists(f'real_world/state_summary/{args.folder_name}/state_summary_L2_wo_sound.txt'):
        with open('real_world/state_summary/{}/{}'.format(args.folder_name, 'state_summary_L2.txt'), 'r') as f:
            L2_captions = f.readlines()
        
        L2_captions_wo_sound = []
        for caption in L2_captions:
            if "Auditory observation" in caption:
                L2_captions_wo_sound.append(caption[:caption.find("Auditory observation")-1])
            else:
                L2_captions_wo_sound.append(caption)
        L2_summary_wo_sound = "".join(L2_captions_wo_sound)
        with open(f'real_world/state_summary/{args.folder_name}/state_summary_L2_wo_sound.txt', 'w') as f:
            f.write(L2_summary_wo_sound)
    else:
        with open(f'real_world/state_summary/{args.folder_name}/state_summary_L2_wo_sound.txt', 'r') as f:
            L2_captions_wo_sound = f.readlines()
        L2_summary_wo_sound = "".join(L2_captions_wo_sound)

    if not os.path.exists(f'real_world/state_summary/{args.folder_name}/state_summary_L1_wo_sound.txt'):
        with open('real_world/state_summary/{}/{}'.format(args.folder_name, 'state_summary_L1.txt'), 'r') as f:
            L1_captions = f.readlines()

        L1_captions_wo_sound = []
        for caption in L1_captions:
            timestep = caption.split(".")[0]
            if "Auditory observation" in caption:
                if timestep in L2_summary_wo_sound:
                    L1_captions_wo_sound.append(caption[:caption.find("Auditory observation")-1])
            else:
                L1_captions_wo_sound.append(caption)
        L1_summary_wo_sound = "".join(L1_captions_wo_sound)
        with open(f'real_world/state_summary/{args.folder_name}/state_summary_L1_wo_sound.txt', 'w') as f:
            f.write(L1_summary_wo_sound)


    with open(f'real_world/state_summary/{args.folder_name}/global_sg.pkl', 'rb') as f:
        global_sg = pickle.load(f)

    logger.info("[Reasoning] Running failure analysis")
    run_reasoning(args=args, task=task_info, global_sg=global_sg)


def run_BLIP2(args, task):
    args.folder_name = task["general_folder_name"]
    f_name = args.folder_name
    reasoning_json_name = 'reasoning-BLIP2-direct.json'
    if not os.path.exists(f'real_world/state_summary/{args.folder_name}/{reasoning_json_name}'):
        reasoning_dict = {}

        with open('LLM/prompts-gpt4.json', 'r') as f:
            prompt_info = json.load(f)

        # get failure reason
        reason_prompt = {}
        reason_prompt['system'] = prompt_info['prompt-simple-qa-2']['template-system']
        reason_prompt['user'] = prompt_info['prompt-simple-qa-2']['template-user'].replace("[TASK_NAME]", task['name'])
        reason_prompt['user'] = reason_prompt['user'].replace("[SUCCESS_CONDITION]", task['success_condition'])

        L1_captions = []
        state_summary_L1 = ""
        with open('real_world/state_summary/{}/state_summary_L1_BLIP2.txt'.format(args.folder_name), 'r') as f:
            L1_captions = f.readlines()
        state_summary_L1 = "".join(L1_captions)
        reason_prompt['user'] = reason_prompt['user'].replace("[L1_SUMMARY]", state_summary_L1)
        reason_prompt['user'] = reason_prompt['user'].replace("[L2_SUMMARY]", get_robot_plan(args, step=None, with_obs=False))
        logger.debug("BLIP2 reason prompt: %s", reason_prompt['user'])

        reason, _ = llm_prompter.query(prompt=reason_prompt, sampling_params=prompt_info['prompt-simple-qa-2']['params'],
                            save=True, save_dir=f'LLM/{f_name}')
        
        # get failure steps
        step_prompt = {}
        step_prompt['system'] = prompt_info['prompt-simple-qa-step']['template-system']
        step_prompt['user'] = prompt_info['prompt-simple-qa-step']['template-user'].replace("[PREV_PROMPT]", reason_prompt['user'] + " " + reason)
        # print(step_prompt['user']) 
        step, _ = llm_prompter.query(prompt=step_prompt, sampling_params=prompt_info['prompt-simple-qa-step']['params'],
                        save=True, save_dir=f'LLM/{f_name}')

        step_str = step.split(" ")[0]
        if step_str[-1] == '.' or step_str[-1] == ',':
            step_str = step_str[:-1]
        reasoning_dict['pred_failure_reason'] = reason
        reasoning_dict['pred_failure_step'] = step_str
        
        reasoning_dict['gt_failure_reason'] = task['gt_failure_reason']
        reasoning_dict['gt_failure_step'] = task['gt_failure_step']
        reasoning_dict = _attach_summaries_and_trace(args, reasoning_dict)

        with open('real_world/state_summary/{}/{}'.format(args.folder_name, reasoning_json_name), 'w') as f:
            json.dump(reasoning_dict, f)


def LLM_direct_summary(args, task_info):
    # TODO: try L0 as well, need to pay attention to steps after ignore(s)
    args.folder_name = task_info["general_folder_name"]
    f_name = args.folder_name
    # Get L0 summary and convert all to text
    pickle_names = os.listdir(f'real_world/state_summary/{args.folder_name}/local_graphs')
    steps = sorted([int(p.split('.')[0].split("_")[-1]) for p in pickle_names])
    state_summary_L0 = ""
    with open('real_world/state_summary/{}/state_summary_L1.txt'.format(args.folder_name), 'r') as f:
        L0_captions = f.readlines()
    state_summary_L0 = "".join(L0_captions)

    # Prompt LLM
    with open('LLM/prompts-gpt4.json', 'r') as f:
        prompt_info = json.load(f)

    # Backward-compatible prompt key aliases for repos using newer prompt schemas.
    prompt_aliases = {
        'prompt-action-binary': 'subgoal-verifier',
        'prompt-action-reason': 'reasoning-execution',
        'prompt-action-reason-no-history': 'reasoning-execution-no-history',
        'prompt-reason-step': 'reasoning-execution-steps',
        'prompt-plan': 'reasoning-plan',
        'prompt-plan-step': 'reasoning-plan-steps',
    }
    for target_key, source_key in prompt_aliases.items():
        if target_key not in prompt_info and source_key in prompt_info:
            prompt_info[target_key] = prompt_info[source_key]

    # Backward-compatible prompt key aliases for repos using newer prompt schemas.
    prompt_aliases = {
        'prompt-action-binary': 'subgoal-verifier',
        'prompt-action-reason': 'reasoning-execution',
        'prompt-action-reason-no-history': 'reasoning-execution-no-history',
        'prompt-reason-step': 'reasoning-execution-steps',
        'prompt-plan': 'reasoning-plan',
        'prompt-plan-step': 'reasoning-plan-steps',
    }
    for target_key, source_key in prompt_aliases.items():
        if target_key not in prompt_info and source_key in prompt_info:
            prompt_info[target_key] = prompt_info[source_key]

    if not os.path.exists(f'real_world/state_summary/{args.folder_name}/summary.txt'):
        prompt = {}
        prompt['system'] = prompt_info['prompt-direct-summary']['template-system']
        prompt['user'] = prompt_info['prompt-direct-summary']['template-user'].replace("[TASK_NAME]", task['name'])
        prompt['user'] = prompt['user'].replace("[WORLD_STATE_HISTORY]", state_summary_L0)

        summary, _ = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['prompt-direct-summary']['params'], 
                        save=True, save_dir=f'LLM/{f_name}')
        logger.info("[Summary] direct summary generated")
        with open(f'real_world/state_summary/{args.folder_name}/summary.txt', 'w') as f:
            f.write(summary)
    else:
        with open(f'real_world/state_summary/{args.folder_name}/summary.txt', 'r') as f:
            summary = f.read()
    logger.debug("direct summary: %s", summary)
    
    reasoning_json_name = 'llm-direct-reasoning.json'
    
    if not os.path.exists(f'real_world/state_summary/{args.folder_name}/{reasoning_json_name}'):
        reasoning_dict = {}
        # Reasoning
        reason_prompt = {}
        reason_prompt['system'] = prompt_info['prompt-simple-qa']['template-system']
        reason_prompt['user'] = prompt_info['prompt-simple-qa']['template-user'].replace("[TASK_NAME]", task['name'])
        reason_prompt['user'] = reason_prompt['user'].replace("[SUMMARY]", summary)
        reason_prompt['user'] = reason_prompt['user'].replace("[SUCCESS_CONDITION]", task['success_condition'])
        # reason_prompt['user'] = reason_prompt['user'].replace("[WORLD_STATE_HISTORY]", state_summary_L0)

        reason, _ = llm_prompter.query(prompt=reason_prompt, sampling_params=prompt_info['prompt-simple-qa']['params'],
                        save=True, save_dir=f'LLM/{f_name}')

        logger.info("[Reasoning] direct reason=%s", reason)
        logger.debug("Direct reasoning prompt: %s", reason_prompt['user'])
        reasoning_dict['pred_failure_reason'] = reason

        # Failure steps
        step_prompt = {}
        step_prompt['system'] = prompt_info['prompt-simple-qa-step']['template-system']
        step_prompt['user'] = prompt_info['prompt-simple-qa-step']['template-user'].replace("[PREV_PROMPT]", reason_prompt['user'] + " " + reason)
        step, _ = llm_prompter.query(prompt=step_prompt, sampling_params=prompt_info['prompt-simple-qa-step']['params'],
                        save=True, save_dir=f'LLM/{f_name}')

        step_str = step.split(" ")[0]
        if step_str[-1] == '.' or step_str[-1] == ',':
            step_str = step_str[:-1]
        reasoning_dict['pred_failure_step'] = step_str
        
        reasoning_dict['gt_failure_reason'] = task['gt_failure_reason']
        reasoning_dict['gt_failure_step'] = task['gt_failure_step']
        reasoning_dict = _attach_summaries_and_trace(args, reasoning_dict)

        with open('real_world/state_summary/{}/{}'.format(args.folder_name, reasoning_json_name), 'w') as f:
            json.dump(reasoning_dict, f)


def LLM_direct_reasoning(args, task_info):
    args.folder_name = task_info["general_folder_name"]
    f_name = args.folder_name
    # get reasoning prompt
    with open('LLM/prompts-gpt4.json', 'r') as f:
        prompt_info = json.load(f)
    
    meta_data = read_zarr(f'real_world/data/{args.folder_name}/replay_buffer.zarr')
    total_frames = int(meta_data['data/stage'].shape[0])
    logger.info("[Frames] total=%s", total_frames)

    reasoning_json_name = 'reasoning-wo-framework.json'
    if not os.path.exists(f'real_world/state_summary/{args.folder_name}/{reasoning_json_name}'):
        reasoning_dict = {}

        # get failure reason
        reason_prompt = {}
        reason_prompt['system'] = prompt_info['prompt-simple-qa-2']['template-system']
        reason_prompt['user'] = prompt_info['prompt-simple-qa-2']['template-user'].replace("[TASK_NAME]", task['name'])
        reason_prompt['user'] = reason_prompt['user'].replace("[SUCCESS_CONDITION]", task['success_condition'])

        L1_captions = []
        state_summary_L1 = ""
        with open('real_world/state_summary/{}/state_summary_L1.txt'.format(args.folder_name), 'r') as f:
            L1_captions = f.readlines()
        state_summary_L1 = "".join(L1_captions)
        reason_prompt['user'] = reason_prompt['user'].replace("[L1_SUMMARY]", state_summary_L1)
        reason_prompt['user'] = reason_prompt['user'].replace("[L2_SUMMARY]", get_robot_plan(args, step=None, with_obs=False))
        logger.debug("Direct reasoning prompt: %s", reason_prompt['user'])

        reason, _ = llm_prompter.query(prompt=reason_prompt, sampling_params=prompt_info['prompt-simple-qa-2']['params'],
                            save=True, save_dir=f'LLM/{f_name}')
        
        # get failure steps
        step_prompt = {}
        step_prompt['system'] = prompt_info['prompt-simple-qa-step']['template-system']
        step_prompt['user'] = prompt_info['prompt-simple-qa-step']['template-user'].replace("[PREV_PROMPT]", reason_prompt['user'] + " " + reason)
        # print(step_prompt['user']) 
        step, _ = llm_prompter.query(prompt=step_prompt, sampling_params=prompt_info['prompt-simple-qa-step']['params'],
                        save=True, save_dir=f'LLM/{f_name}')

        step_str = step.split(" ")[0]
        if step_str[-1] == '.' or step_str[-1] == ',':
            step_str = step_str[:-1]
        reasoning_dict['pred_failure_reason'] = reason
        reasoning_dict['pred_failure_step'] = step_str
        
        reasoning_dict['gt_failure_reason'] = task['gt_failure_reason']
        reasoning_dict['gt_failure_step'] = task['gt_failure_step']
        reasoning_dict = _attach_summaries_and_trace(args, reasoning_dict)

        with open('real_world/state_summary/{}/{}'.format(args.folder_name, reasoning_json_name), 'w') as f:
            json.dump(reasoning_dict, f)


def run_reasoning(args, task, global_sg):
    # define reasoning file name based on ablation_type
    if args.ablation_type == 0:
        if args.audio_ver == 1:
            reasoning_json_name = 'reasoning.json'
        elif args.audio_ver == 0:
            reasoning_json_name = 'reasoning-wo-sound.json'
    elif args.ablation_type == 3:
        reasoning_json_name = 'reasoning-only-L2.json'
    elif args.ablation_type == 5:
        reasoning_json_name = 'reasoning-BLIP2.json'

    if os.path.exists(f'real_world/state_summary/{args.folder_name}/{reasoning_json_name}'):
    # if False:
        with open(f'real_world/state_summary/{args.folder_name}/{reasoning_json_name}', 'r') as f:
            reasoning_dict = json.load(f)
        reasoning_dict = _attach_summaries_and_trace(args, reasoning_dict)
        with open(f'real_world/state_summary/{args.folder_name}/{reasoning_json_name}', 'w') as f:
            json.dump(reasoning_dict, f)
        return
    else:
        reasoning_dict = {}

    save_dir = f'LLM/{args.folder_name}'
    os.system("mkdir -p {}".format(save_dir))

    with open('LLM/prompts-gpt4.json', 'r') as f:
        prompt_info = json.load(f)

    prompt_aliases = {
        'prompt-action-binary': 'subgoal-verifier',
        'prompt-action-reason': 'reasoning-execution',
        'prompt-action-reason-no-history': 'reasoning-execution-no-history',
        'prompt-reason-step': 'reasoning-execution-steps',
        'prompt-plan': 'reasoning-plan',
        'prompt-plan-step': 'reasoning-plan-steps',
    }
    for target_key, source_key in prompt_aliases.items():
        if target_key not in prompt_info and source_key in prompt_info:
            prompt_info[target_key] = prompt_info[source_key]

    # Load L2 captions from state_summary_L2.txt
    if args.ablation_type == 0 and args.audio_ver == 0:
        summary_file_name = 'state_summary_L2_wo_sound.txt'
    elif args.ablation_type == 5:
        summary_file_name = 'state_summary_L2_BLIP2.txt'
    else:
        summary_file_name = 'state_summary_L2.txt'
    with open('real_world/state_summary/{}/{}'.format(args.folder_name, summary_file_name), 'r') as f:
        L2_captions = f.readlines()

    # Load L1 captions from state_summary_L1.txt
    if args.ablation_type == 0 and args.audio_ver == 0:
        summary_file_name = 'state_summary_L1_wo_sound.txt'
    elif args.ablation_type == 5:
        summary_file_name = 'state_summary_L1_BLIP2.txt'
    else:
        summary_file_name = 'state_summary_L1.txt'
    with open('real_world/state_summary/{}/{}'.format(args.folder_name, summary_file_name), 'r') as f:
        L1_captions = f.readlines()
    
    # Loop through each subgoal and check for post-condition
    logger.info("[Reasoning] Checking subgoal completion")
    selected_caption = ""
    prompt = {}

    for caption in L2_captions:
        action = caption.split(". ")[1].split(": ")[1].lower()

        # prompt = prompt_action_binary_template.replace("[TASK_NAME]", task['name'])
        prompt['system'] = prompt_info['prompt-action-binary']['template-system']
        prompt['user'] = prompt_info['prompt-action-binary']['template-user'].replace("[ACTION]", action).replace("[SUBGOAL]", action)
        prompt['user'] = prompt['user'].replace("[OBSERVATION]", caption[caption.find("Visual observation"):])

        ans, _  = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['prompt-action-binary']['params'], 
                        save=True, save_dir=save_dir)
        logger.debug("Subgoal verification prompt: %s", prompt['user'])
        logger.debug("Subgoal verification answer: %s", ans)
        is_success = int(ans.split(", ")[0] == "Yes")
        if is_success == 0:
            selected_caption = caption
            logger.info("[Reasoning] Failure identified at %s", caption.split(".")[0])
            break

    # check corresponding L1 caption (plus previous observations) for reasoning
    if args.ablation_type == 0 or args.ablation_type == 5:
        explain_captions = L1_captions
    elif args.ablation_type == 3:
        explain_captions = L2_captions

    if len(selected_caption) != 0:
            logger.info("[Reasoning] Explaining failure from L1 context")
            step_name = selected_caption.split(".")[0]
            for _, caption in enumerate(explain_captions):
                if step_name in caption:
                    action = caption.split(". ")[1].split(": ")[1].lower()
                    prev_observations = get_robot_plan(args, step=step_name, with_obs=True)
                    if len(prev_observations) != 0:
                        prompt_name = 'prompt-action-reason'
                    else:
                        prompt_name = 'prompt-action-reason-no-history'

                    prompt['system'] = prompt_info[prompt_name]['template-system']
                    prompt['user'] = prompt_info[prompt_name]['template-user'].replace("[ACTION]", action)
                    prompt['user'] = prompt['user'].replace("[TASK_NAME]", task['name'])
                    prompt['user'] = prompt['user'].replace("[STEP]", step_name)
                    prompt['user'] = prompt['user'].replace("[SUMMARY]", prev_observations)
                    if args.ablation_type == 3:
                        prompt['user'] = prompt['user'].replace("[OBSERVATION]", caption[caption.find("Goal"):])
                    else:
                        prompt['user'] = prompt['user'].replace("[OBSERVATION]", caption[caption.find("Action"):])
                    ans, log_prob  = llm_prompter.query(prompt=prompt, sampling_params=prompt_info[prompt_name]['params'], 
                                                save=True, save_dir=save_dir)
                    logger.debug("Reasoning prompt: %s", prompt['user'])
                    logger.debug("Reasoning answer: %s (log_prob=%s)", ans, log_prob)

                    reasoning_dict['pred_failure_reason'] = ans
                    # Map to L1 frames
                    if args.ablation_type == 3:
                        reasoning_dict['pred_failure_step'] = [step_name]
                    else:
                        prompt = {}
                        prompt['system'] = prompt_info['prompt-reason-step']['template-system']
                        prompt['user'] = prompt_info['prompt-reason-step']['template-user'].replace("[FAILURE_REASON]", ans)
                        time_steps, log_prob = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['prompt-reason-step']['params'],
                                                                save=True, save_dir=save_dir)
                        logger.info("[Reasoning] Relevant time steps: %s", time_steps)
                        reasoning_dict['pred_failure_step'] = [time_step.replace(",", "") for time_step in time_steps.split(", ")]
                    break
    else:
        logger.info("[Reasoning] Subgoals passed, running plan-level analysis")
        prompt['system'] = prompt_info['prompt-plan']['template-system']
        prompt['user'] = prompt_info['prompt-plan']['template-user'].replace("[TASK_NAME]", task['name'])
        prompt['user'] = prompt['user'].replace("[SUCCESS_CONDITION]", task['success_condition'])
        prompt['user'] = prompt['user'].replace("[CURRENT_STATE]", get_scene_text(global_sg))
        prompt['user'] = prompt['user'].replace("[OBSERVATION]", get_robot_plan(args=args, step=None, with_obs=False))
        ans, _ = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['prompt-plan']['params'], 
                        save=True, save_dir=save_dir)
        logger.debug("Plan reasoning prompt: %s", prompt['user'])
        logger.debug("Plan reasoning answer: %s", ans)
        reasoning_dict['pred_failure_reason'] = ans
        prompt['system'] = prompt_info['prompt-plan-step']['template-system']
        prompt['user'] = prompt_info['prompt-plan-step']['template-user'].replace("[PREV_PROMPT]", prompt['user'] + " " + ans)
        step, _ = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['prompt-plan-step']['params'], 
                        save=True, save_dir=save_dir)
        logger.debug("Plan step prompt: %s", prompt['user'])
        logger.debug("Plan step answer: %s", step)
        step_str = step.split(" ")[0]
        if step_str[-1] == '.' or step_str[-1] == ',':
            step_str = step_str[:-1]
        reasoning_dict['pred_failure_step'] = step_str

    reasoning_dict['gt_failure_reason'] = task['gt_failure_reason']
    reasoning_dict['gt_failure_step'] = task['gt_failure_step']
    reasoning_dict = _attach_summaries_and_trace(args, reasoning_dict)
    logger.info(
        "[Reasoning] predicted_step=%s predicted_reason=%s",
        reasoning_dict['pred_failure_step'],
        reasoning_dict['pred_failure_reason'],
    )
    
    with open('real_world/state_summary/{}/{}'.format(args.folder_name, reasoning_json_name), 'w') as f:
        json.dump(reasoning_dict, f)


def generate_replan(f_name, global_sg, task, task_object_list, args):
    pass


def create_folders(f_name, artifact_level="full"):
    summary_dir = _summary_dir(f_name)
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(f"{summary_dir}/local_graphs", exist_ok=True)

    if artifact_level == "full":
        os.makedirs(f"{summary_dir}/mdetr_obj_det/images", exist_ok=True)
        os.makedirs(f"{summary_dir}/mdetr_obj_det/det", exist_ok=True)
        os.makedirs(f"{summary_dir}/mdetr_obj_det/clip_processed_det", exist_ok=True)
        os.makedirs(f"real_world/scene/{f_name}", exist_ok=True)


def clear_rebuild_artifacts(f_name):
    summary_dir = _summary_dir(f_name)
    scene_dir = f"real_world/scene/{f_name}"
    removable_dirs = [
        f"{summary_dir}/local_graphs",
        f"{summary_dir}/mdetr_obj_det/images",
        f"{summary_dir}/mdetr_obj_det/det",
        f"{summary_dir}/mdetr_obj_det/clip_processed_det",
        scene_dir,
    ]
    removable_files = [
        f"{summary_dir}/global_sg.pkl",
        f"{summary_dir}/L1_key_frames.txt",
        f"{summary_dir}/state_summary_L1.txt",
        f"{summary_dir}/state_summary_L2.txt",
        f"{summary_dir}/state_summary_L1_BLIP2.txt",
        f"{summary_dir}/state_summary_L2_BLIP2.txt",
        f"{summary_dir}/state_summary_L1_wo_sound.txt",
        f"{summary_dir}/state_summary_L2_wo_sound.txt",
        f"{summary_dir}/reasoning.json",
        f"{summary_dir}/reasoning-BLIP2.json",
        f"{summary_dir}/reasoning-wo-sound.json",
        f"{summary_dir}/reasoning-only-L2.json",
        f"{summary_dir}/timings.json",
        f"{summary_dir}/summary.txt",
        f"{summary_dir}/llm_trace.json",
    ]

    removed_any = False
    for path in removable_dirs:
        if os.path.isdir(path):
            shutil.rmtree(path)
            removed_any = True
    for path in removable_files:
        if os.path.exists(path):
            os.remove(path)
            removed_any = True

    if removed_any:
        logger.info("[Rebuild] Cleared cached artifacts for %s", f_name)

def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser("Robot Failure Summarization")
    parser.add_argument('--tasks','--list', nargs='*')
    parser.add_argument('--folder_name', type=str, default="", help="if pipeline should be run on only one specific folder")
    parser.add_argument('--obj_det', type=str, default="mdetr", help="which object detection model to use")
    parser.add_argument('--audio_ver', type=int, default=1, help='1 is with detected audio, 0 is without audio')
    parser.add_argument('--ablation_type', type=int, default=0, help="which experiment to run")
    parser.add_argument(
        '--mdetr_confidence_threshold',
        type=float,
        default=0.9,
        help="minimum MDETR confidence for keeping a detection",
    )
    parser.add_argument(
        '--force_rebuild_sg',
        action='store_true',
        help="rebuild scene-graph artifacts even if cached outputs already exist",
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default="INFO",
        help="logging level for pipeline output (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        '--artifact_level',
        type=str,
        default="full",
        choices=["final", "full"],
        help="which artifact set to preserve: final validation outputs only or all current debug artifacts",
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help="record per-task stage timings to real_world/state_summary/<task>/timings.json",
    )
    parser.add_argument(
        '--task_workers',
        type=str,
        default="1",
        help="batch-only worker count or 'auto' for SG/summary parallelism",
    )
    parser.add_argument(
        '--reasoning_workers',
        type=int,
        default=1,
        help="batch-only reasoning worker count; keep 1 for reproducibility",
    )
    parser.add_argument(
        '--outlier_filter_max_points',
        type=int,
        default=30000,
        help="skip expensive point-cloud outlier filtering when a cloud exceeds this size",
    )
    return parser

def read_zarr(file_path):
    meta_data = zarr.open(file_path, mode='r')
    stage = np.array(meta_data['data/stage'])
    return meta_data

def get_interact_actions(meta_data, total_frames, args, task_json):
    stages = {}
    prev_stage = None
    curr_stage = None
    for step_idx in range(0, total_frames):
        curr_stage_raw = meta_data['data/stage'][step_idx]
        curr_stage = int(np.asarray(curr_stage_raw).reshape(-1)[0])
        # for the first frame of an action
        if curr_stage not in stages:
            stages[curr_stage] = [step_idx]
        # for the last frame of an action
        if prev_stage is not None and curr_stage != prev_stage:
            # -- remove later --
            if args.folder_name == 'heatPotato1' and prev_stage == 6:
                step_idx = step_idx - 3
            if args.folder_name == 'heatPotato2' and prev_stage == 8: # 7268
                step_idx = step_idx - 2
            if args.folder_name == 'heatPotato2' and prev_stage == 9: # 10000 - 90 = 9910
                step_idx = step_idx - 90
            if args.folder_name == 'heatPotato2' and prev_stage == 10: # 11597 
                step_idx = step_idx - 4
            if args.folder_name == 'boilWater1' and prev_stage == 6: # 4685 
                step_idx = step_idx - 10
            # ------------------
            if prev_stage in stages:
                stages[prev_stage].append(step_idx-1)
        prev_stage = curr_stage
    # for the last frame for last stage
    video_root = _get_video_root(args.folder_name)
    last_idx = len(os.listdir(f'{video_root}/color'))-2
    if curr_stage is not None and curr_stage in stages:
        stages[curr_stage].append(last_idx)
    actions = task_json['actions']
    interact_actions = {}
    logger.debug("stages: %s", stages)
    for k, v in stages.items():
        if actions[k] == "Terminate":
            continue
        interact_actions[(v[0], v[1])] = actions[k]
    logger.info("[Actions] %s", interact_actions)
    return interact_actions


def run_real_world_pipeline(args, task_info, include_reasoning=True):
    configure_logging(args.log_level)
    args.folder_name = task_info["general_folder_name"]
    profiler = _make_task_profiler(args)
    video_root = _get_video_root(args.folder_name)
    global_sg_path = f"{_summary_dir(args.folder_name)}/global_sg.pkl"

    try:
        if args.force_rebuild_sg:
            clear_rebuild_artifacts(args.folder_name)
        create_folders(args.folder_name, artifact_level=getattr(args, "artifact_level", "full"))
        logger.info("[Task] %s", args.folder_name)

        with profiler.stage("metadata_load"):
            meta_data = read_zarr(f'real_world/data/{args.folder_name}/replay_buffer.zarr')
            total_frames = int(meta_data['data/stage'].shape[0])
        profiler.set_metric("total_frames", total_frames)
        logger.info("[Frames] total=%s", total_frames)

        needs_scene_graph_rebuild = args.force_rebuild_sg or not os.path.exists(global_sg_path)
        detector = None
        task_object_text_feats = None
        if needs_scene_graph_rebuild:
            detector_cache_exists = _has_cached_detection_outputs(args.folder_name, args.obj_det)
            if args.force_rebuild_sg or not detector_cache_exists:
                with profiler.stage("model_init"):
                    if args.obj_det == "mdetr":
                        detector = get_seg_model()
                    get_clip_model()
                    if task_info.get("object_list"):
                        task_object_text_feats = get_text_feats(task_info["object_list"])

        interact_actions = get_interact_actions(meta_data, total_frames, args, task_info)
        interact_actions_end_idx = [idx[1] for idx in interact_actions.keys() if "Ignore" not in interact_actions[idx]]
        logger.info("[KeyFrames] action_end_frames=%s", interact_actions_end_idx)

        ignore_dict = {}
        for key in interact_actions:
            if "Ignore" in interact_actions[key]:
                ignore_length = key[1] - key[0]
                ignore_end_idx = key[1]
                ignore_dict[ignore_end_idx] = ignore_length

        if args.audio_ver == 1:
            with profiler.stage("audio"):
                audio_path = f'{video_root}/audio.wav'
                if os.path.exists(audio_path):
                    volume_thresh = 0.03 if task_info['task_idx'] == 3 else 0.04
                    try:
                        detected_sounds = get_sound_events(audio_path=audio_path, volume_thresh=volume_thresh)
                    except Exception as exc:
                        logger.warning("[AudioWarning] Failed to process audio for %s: %s", args.folder_name, exc)
                        detected_sounds = {}
                else:
                    logger.info("[Audio] no audio file found")
                    detected_sounds = {}
            if detected_sounds:
                logger.info("[Audio] detected sounds=%s", detected_sounds)

            sound_det_idx_dict = {}
            for sound_range in detected_sounds.keys():
                step_idx = sound_range[1] * 30
                total_ignore_length = 0
                for ignore_end_idx in ignore_dict:
                    if step_idx > ignore_end_idx:
                        total_ignore_length += ignore_dict[ignore_end_idx]
                sound_det_idx_dict[sound_range[1] * 30 + total_ignore_length] = detected_sounds[sound_range]
            logger.debug("sound_det_idx_dict: %s", sound_det_idx_dict)
        else:
            sound_det_idx_dict = {}

        key_frames = []
        if needs_scene_graph_rebuild:
            total_points_dict, bbox3d_dict = {}, {}
            prev_graph = SceneGraph()
            bbox2d_dict = {}
            last_rgb = None
            detection_history = {}
            local_graph_history = {}
            object_list = task_info['object_list']
            distractor_list = task_info.get('distractor_list', [])
            pre_filtered_frames = [idx for idx in range(total_frames) if (idx == 0) or (idx in interact_actions_end_idx) or (idx in sound_det_idx_dict)]


            for step_idx in pre_filtered_frames:
                logger.info("[Frame] %s", step_idx)
                logger.debug("object list: %s", object_list)
                logger.debug("distractor list: %s", distractor_list)

                rgb = imread(f'{video_root}/color/{step_idx}.0.0.0')
                depth = imread(f'{video_root}/depth/{step_idx}.0.0')
                last_rgb = rgb
                local_sg, bbox3d_dict, total_points_dict, bbox2d_dict = get_scene_graph(
                    args,
                    rgb,
                    depth,
                    step_idx,
                    object_list,
                    distractor_list,
                    detector,
                    total_points_dict,
                    bbox3d_dict,
                    meta_data,
                    task_info,
                    object_name_feats=task_object_text_feats,
                    detection_history=detection_history,
                    local_graph_history=local_graph_history,
                    profiler=profiler,
                )
                logger.info("[SceneGraph] frame=%s %s", step_idx, _scene_graph_overview(local_sg))
                logger.debug("Current graph:\n%s", local_sg)
                with profiler.stage("local_graph_write"):
                    with open(f'{_summary_dir(args.folder_name)}/local_graphs/local_sg_{step_idx}.pkl', 'wb') as f:
                        pickle.dump(local_sg, f)
                local_graph_history[step_idx] = local_sg
                profiler.increment_metric("scene_graph_frames")

                if local_sg != prev_graph and step_idx not in key_frames:
                    key_frames.append(step_idx)
                    prev_graph = local_sg
                if step_idx in interact_actions_end_idx and step_idx not in key_frames:
                    key_frames.append(step_idx)
                if step_idx in sound_det_idx_dict and step_idx not in key_frames:
                    key_frames.append(step_idx)

            global_sg = SceneGraph()
            if last_rgb is None:
                raise RuntimeError(f"No RGB frames were processed for {args.folder_name}")
            for label in total_points_dict.keys():
                if label in bbox3d_dict.keys() and label in bbox2d_dict.keys():
                    new_node = SceneGraphNode(
                        name=label,
                        object_id=label,
                        pos3d=bbox3d_dict[label].get_center(),
                        corner_pts=np.array(bbox3d_dict[label].get_box_points()),
                        bbox2d=bbox2d_dict[label],
                        pcd=total_points_dict[label],
                        global_node=True,
                    )
                    global_sg.add_node_wo_edge(new_node)
                    global_sg.add_node(new_node, last_rgb)
            with profiler.stage("global_graph_write"):
                with open(global_sg_path, 'wb') as f:
                    pickle.dump(global_sg, f)
            graph_stats = _local_graph_artifact_stats(args.folder_name)
            logger.info("[GraphArtifacts] %s", [(item["name"], item["size"]) for item in graph_stats])
            _assert_nonempty_graph_rebuild(args.folder_name)

            with profiler.stage("keyframe_write"):
                with open(f'{_summary_dir(args.folder_name)}/L1_key_frames.txt', 'w') as f:
                    for frame in key_frames:
                        f.write(f"{frame}\n")
            profiler.set_metric("key_frame_count", len(key_frames))
        else:
            with profiler.stage("global_sg_load"):
                with open(global_sg_path, 'rb') as f:
                    global_sg = pickle.load(f)
            logger.debug("Global SG:\n%s", global_sg)
            key_frame_path = f'{_summary_dir(args.folder_name)}/L1_key_frames.txt'
            if os.path.exists(key_frame_path):
                with open(key_frame_path, 'r') as f:
                    key_frames = [int(frame.strip()) for frame in f.readlines() if frame.strip()]
                profiler.set_metric("key_frame_count", len(key_frames))

        L1_summary_file_name, L2_summary_file_name = _get_summary_file_names(args)
        l1_path = f'{_summary_dir(args.folder_name)}/{L1_summary_file_name}'
        l2_path = f'{_summary_dir(args.folder_name)}/{L2_summary_file_name}'

        if not os.path.exists(l1_path):
            with profiler.stage("L1_build"):
                logger.info("[Summary] Generating L1 summary")
                state_summary_L1 = ""
                L1_captions = []
                if not key_frames:
                    with open(f'{_summary_dir(args.folder_name)}/L1_key_frames.txt', 'r') as f:
                        key_frames = [int(frame) for frame in f.readlines()]

                for step_idx in key_frames:
                    if step_idx == 0:
                        continue
                    caption = ""
                    for key in interact_actions:
                        min_step, max_step = key
                        if min_step <= step_idx <= max_step:
                            total_ignore_length = 0
                            for ignore_end_idx in ignore_dict:
                                if step_idx > ignore_end_idx:
                                    total_ignore_length += ignore_dict[ignore_end_idx]
                            caption += f"{convert_step_to_timestep(step_idx-total_ignore_length, video_fps=30)}. Action: {interact_actions[key]}."

                    if "Ignore" in caption or "Skip" in caption:
                        continue

                    if args.ablation_type == 5:
                        with open(f'real_world/BLIP2_captions/{args.folder_name}/caption_{step_idx}.txt', 'r') as f:
                            scene_text = f.readlines()[0] + "."
                    else:
                        with open(f'{_summary_dir(args.folder_name)}/local_graphs/local_sg_{step_idx}.pkl', 'rb') as f:
                            local_sg = pickle.load(f)
                            logger.debug("L1 local graph for frame %s:\n%s", step_idx, local_sg)
                        scene_text = get_scene_text(local_sg)
                    if len(scene_text) != 0:
                        caption += f" Visual observation: {scene_text}"

                    if step_idx in sound_det_idx_dict:
                        caption += f" Auditory observation: {sound_det_idx_dict[step_idx]}."

                    caption += "\n"
                    logger.debug("L1 caption: %s", caption.strip())
                    if len(L1_captions) != 0 and caption.split(".")[0] == L1_captions[-1].split(".")[0]:
                        continue
                    state_summary_L1 += caption
                    L1_captions.append(caption)
                with open(l1_path, 'w') as f:
                    f.write(state_summary_L1)
                logger.info("[Summary] L1 captions=%s", len(L1_captions))
        else:
            logger.info("[Summary] Reusing cached L1 summary")
            with open(l1_path, 'r') as f:
                L1_captions = f.readlines()
            state_summary_L1 = "".join(L1_captions)

        if not os.path.exists(l2_path):
            with profiler.stage("L2_build"):
                logger.info("[Summary] Generating L2 summary")
                L2_captions = []
                for step_idx in interact_actions_end_idx:
                    for caption in L1_captions:
                        total_ignore_length = 0
                        for ignore_end_idx in ignore_dict:
                            if step_idx > ignore_end_idx:
                                total_ignore_length += ignore_dict[ignore_end_idx]
                        step_num = step_idx - total_ignore_length
                        if convert_step_to_timestep(step_num, video_fps=30) in caption:
                            L2_captions.append(caption.replace("Action", "Goal"))

                state_summary_L2 = "".join(L2_captions)
                with open(l2_path, 'w') as f:
                    f.write(state_summary_L2)
                logger.info("[Summary] L2 captions=%s", len(L2_captions))
        else:
            logger.info("[Summary] Reusing cached L2 summary")
            with open(l2_path, 'r') as f:
                L2_captions = f.readlines()
            state_summary_L2 = "".join(L2_captions)

        profiler.set_metric("l1_caption_count", len(L1_captions))
        profiler.set_metric("l2_caption_count", len(L2_captions))

        if include_reasoning:
            logger.info("[Reasoning] Running failure analysis")
            with profiler.stage("reasoning"):
                if args.ablation_type == 5:
                    run_BLIP2(args, task_info)
                else:
                    run_reasoning(args=args, task=task_info, global_sg=global_sg)

        profiler.save()
        return profiler.to_dict()
    finally:
        profiler.save()


def run_real_world_reasoning_only(args, task_info):
    configure_logging(args.log_level)
    args.folder_name = task_info["general_folder_name"]
    profiler = _make_task_profiler(args)
    global_sg_path = f"{_summary_dir(args.folder_name)}/global_sg.pkl"

    try:
        create_folders(args.folder_name, artifact_level=getattr(args, "artifact_level", "full"))
        if args.audio_ver == 0 and args.ablation_type in [0, 3, 5]:
            l1_name, l2_name = _get_summary_file_names(args)
            if not os.path.exists(f"{_summary_dir(args.folder_name)}/{l1_name}") and os.path.exists(f"{_summary_dir(args.folder_name)}/state_summary_L1.txt"):
                with profiler.stage("reasoning"):
                    run_wo_sound(args, task_info)
                return profiler.to_dict()

        if not os.path.exists(global_sg_path):
            raise FileNotFoundError(f"Missing global scene graph for reasoning: {global_sg_path}")

        with open(global_sg_path, 'rb') as f:
            global_sg = pickle.load(f)

        logger.info("[Reasoning] Running failure analysis")
        with profiler.stage("reasoning"):
            if args.ablation_type == 5:
                run_BLIP2(args, task_info)
            else:
                run_reasoning(args=args, task=task_info, global_sg=global_sg)
        profiler.save()
        return profiler.to_dict()
    finally:
        profiler.save()


if __name__ == '__main__':
    args = config_parser().parse_args()

    task_lis = []
    # print("Args: ", args)
    task_lis = list(map(int, args.tasks))
    if task_lis == [0]:
        task_lis = list(range(1, 31))

    # f_names = []
    with open('real_world/tasks_real_world.json', 'r') as f:
        tasks_json = json.load(f)

    for task_idx in task_lis:
        task = tasks_json['Task ' + str(task_idx)]
        # four LLM baselines
        if args.ablation_type in [0, 3, 5]:
            if args.audio_ver == 0:
                run_wo_sound(args, task)
            else:
                run_real_world_pipeline(args, task)
        elif args.ablation_type == 2:
            LLM_direct_summary(args, task)
        elif args.ablation_type == 4:
            LLM_direct_reasoning(args, task)
