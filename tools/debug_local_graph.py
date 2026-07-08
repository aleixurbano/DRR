import argparse
import json
import os
import pickle

from imagecodecs import imread

from reflect.real_world.detection import get_seg_model
from reflect.real_world.local_graph import get_scene_graph
from reflect.real_world.prompting import (
    _get_video_root,
    clear_rebuild_artifacts,
    config_parser,
    create_folders,
    read_zarr,
)


def main():
    parser = argparse.ArgumentParser(description="Build and inspect one local scene graph in a fresh process.")
    parser.add_argument("--task-id", default="Task 8")
    parser.add_argument("--step-idx", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--write-debug-copy", action="store_true")
    args_cli = parser.parse_args()

    with open("tasks_real_world.json", "r") as f:
        tasks = json.load(f)
    task = tasks[args_cli.task_id]

    run_args = config_parser().parse_args([])
    run_args.folder_name = task["general_folder_name"]
    run_args.obj_det = "mdetr"
    run_args.mdetr_confidence_threshold = args_cli.threshold
    run_args.force_rebuild_sg = True

    clear_rebuild_artifacts(run_args.folder_name)
    create_folders(run_args.folder_name)
    os.makedirs(f"real_world/state_summary/{run_args.folder_name}/local_graphs", exist_ok=True)
    os.makedirs(f"real_world/scene/{run_args.folder_name}", exist_ok=True)

    video_root = _get_video_root(run_args.folder_name)
    meta_data = read_zarr(f"real_world/data/{run_args.folder_name}/replay_buffer.zarr")
    rgb = imread(f"{video_root}/color/{args_cli.step_idx}.0.0.0")
    depth = imread(f"{video_root}/depth/{args_cli.step_idx}.0.0")

    local_sg, bbox3d_dict, total_points_dict, bbox2d_dict = get_scene_graph(
        run_args,
        rgb,
        depth,
        args_cli.step_idx,
        task["object_list"],
        task.get("distractor_list", []),
        get_seg_model(),
        {},
        {},
        meta_data,
        task,
    )

    print("task:", run_args.folder_name)
    print("frame:", args_cli.step_idx)
    print("nodes:", [node.name for node in local_sg.nodes])
    print("edges:", [str(edge) for edge in local_sg.edges.values()])
    print("bbox3d keys:", sorted(bbox3d_dict.keys()))
    print("point-cloud keys:", sorted(total_points_dict.keys()))
    print("bbox2d keys:", sorted(bbox2d_dict.keys()))

    graph_dir = f"real_world/state_summary/{run_args.folder_name}/local_graphs"
    os.makedirs(graph_dir, exist_ok=True)
    canonical_path = f"{graph_dir}/local_sg_{args_cli.step_idx}.pkl"
    with open(canonical_path, "wb") as f:
        pickle.dump(local_sg, f)
    print("wrote canonical:", canonical_path, os.path.getsize(canonical_path))

    if args_cli.write_debug_copy:
        out_path = f"{graph_dir}/local_sg_{args_cli.step_idx}_debug.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(local_sg, f)
        print("wrote:", out_path, os.path.getsize(out_path))


if __name__ == "__main__":
    main()
