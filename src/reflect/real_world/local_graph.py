import pickle
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from contextlib import nullcontext
from PIL import Image
import open3d as o3d
# from detic_object_detector import *
from reflect.real_world.detection import (
    plot_inference_segmentation,
    plot_results,
    prepare_inference_image,
)
from reflect.real_world.clip_utils import get_img_feats_batch, get_nn_text, get_text_feats
from reflect.real_world.scene_graph import Node, SceneGraph
from reflect.real_world.constants import real_world_name_map
from reflect.real_world.logging_utils import get_logger
from transforms import depth_to_point_cloud_components  # vendored MDETR; on sys.path via reflect.cli.real_world_validation
from reflect.real_world.gpu_utils import voxel_downsample_gpu_batch, outlier_removal_gpu_batch

logger = get_logger(__name__)

# RealSense intrinsic matrix
intrinsics_matrix = np.array([
        [914.27246, 0.0, 647.0733],
        [0.0, 913.2658, 356.32526],
        [0.0, 0.0, 1.0]     
    ])

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def should_write_debug_artifacts(args):
    return getattr(args, "artifact_level", "full") == "full"

# depth_intr:  [[891.35443   0.      651.71576]
#  [  0.      891.35443 363.69656]
#  [  0.        0.        1.     ]]


def visualize_box(total_points, total_colors, box, color):
    for i in np.arange(box[0][0], box[4][0], 100):
        total_points = torch.cat((total_points, torch.tensor([[i, box[0][1], box[0][2]]])), 0)
        total_colors = torch.cat((total_colors, torch.tensor([color])), 0)

        total_points = torch.cat((total_points, torch.tensor([[i, box[0][1], box[4][2]]])), 0)
        total_colors = torch.cat((total_colors, torch.tensor([color])), 0)

        total_points = torch.cat((total_points, torch.tensor([[i, box[4][1], box[0][2]]])), 0)
        total_colors = torch.cat((total_colors, torch.tensor([color])), 0)

        total_points = torch.cat((total_points, torch.tensor([[i, box[4][1], box[4][2]]])), 0)
        total_colors = torch.cat((total_colors, torch.tensor([color])), 0)

    for i in np.arange(box[0][1], box[4][1], 100):
        total_points = torch.cat((total_points, torch.tensor([[box[0][0], i, box[0][2]]])), 0)
        total_colors = torch.cat((total_colors, torch.tensor([color])), 0)

        total_points = torch.cat((total_points, torch.tensor([[box[0][0], i, box[4][2]]])), 0)
        total_colors = torch.cat((total_colors, torch.tensor([color])), 0)

        total_points = torch.cat((total_points, torch.tensor([[box[4][0], i, box[0][2]]])), 0)
        total_colors = torch.cat((total_colors, torch.tensor([color])), 0)

        total_points = torch.cat((total_points, torch.tensor([[box[4][0], i, box[4][2]]])), 0)
        total_colors = torch.cat((total_colors, torch.tensor([color])), 0)

    for i in np.arange(box[0][2], box[4][2], 100):
        total_points = torch.cat((total_points, torch.tensor([[box[0][0], box[0][1], i]])), 0)
        total_colors = torch.cat((total_colors, torch.tensor([color])), 0)

        total_points = torch.cat((total_points, torch.tensor([[box[0][0], box[4][1], i]])), 0)
        total_colors = torch.cat((total_colors, torch.tensor([color])), 0)

        total_points = torch.cat((total_points, torch.tensor([[box[4][0], box[0][1], i]])), 0)
        total_colors = torch.cat((total_colors, torch.tensor([color])), 0)

        total_points = torch.cat((total_points, torch.tensor([[box[4][0], box[4][1], i]])), 0)
        total_colors = torch.cat((total_colors, torch.tensor([color])), 0)

    return total_points, total_colors

def bb_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

# TODO: remove duplicate bbox for different labels as well
def is_duplicate_bbox(idx, outputs):
    for idx2 in range(idx+1, len(outputs['labels'])):
        temp_iou = bb_iou(outputs['bbox_2d'][idx], outputs['bbox_2d'][idx2])
        logger.debug(
            "iou between %s and %s: %s",
            outputs['labels'][idx],
            outputs['labels'][idx2],
            temp_iou,
        )
        if idx2 != idx and \
                outputs['labels'][idx2] == outputs['labels'][idx] and \
                bb_iou(outputs['bbox_2d'][idx], outputs['bbox_2d'][idx2]) > 0.25:
            return True
    return False

def confirm_obj_det(args, rgb, outputs, object_list, step_idx, object_name_feats=None):
    h, w, _ = rgb.shape
    if object_name_feats is None:
        object_name_feats = get_text_feats(object_list)
    # sorting outputs according to score as it helps remove duplicates with lower score
    outputs['scores'], outputs['labels'], outputs['pred_masks'], outputs['bbox_2d'] = zip(*sorted(zip(outputs['scores'], 
                                                                                                outputs['labels'],
                                                                                                outputs['pred_masks'],
                                                                                                outputs['bbox_2d'])))
    
    outputs['labels'] = np.array(list(outputs['labels']))
    outputs['scores'] = np.array(list(outputs['scores']))
    if isinstance(outputs['pred_masks'][0], np.ndarray):
        outputs['pred_masks'] = np.array(list(outputs['pred_masks']))
    else:
        iter = map(lambda x: x.numpy(), outputs['pred_masks'])
        outputs['pred_masks'] = np.array(list(iter))
    if isinstance(outputs['bbox_2d'][0], np.ndarray):
        outputs['bbox_2d'] = np.array(list(outputs['bbox_2d']))
    else:
        iter = map(lambda x: x.numpy(), outputs['bbox_2d'])
        outputs['bbox_2d'] = np.array(list(iter))

    logger.debug("outputs before labels: %s", outputs['labels'])
    logger.debug("outputs before scores: %s", outputs['scores'])
    logger.debug("outputs before pred_masks shape: %s", outputs['pred_masks'].shape)
    logger.debug("outputs before bbox_2d shape: %s", outputs['bbox_2d'].shape)
    logger.debug("outputs before total_detections: %s", outputs['total_detections'])


    new_outputs = {
        'total_detections': 0,
        'labels': np.array([]),
        'scores': np.array([]),
        'pred_masks': np.array([]),
        'bbox_2d': np.array([])
    }
    crop_images = []
    crop_indices = []
    for idx in range(len(outputs['bbox_2d'])):
        box = outputs['bbox_2d'][idx]
        h1 = max(0, int(box[1]-10))
        h2 = min(h, int(box[3]+10))
        w1 = max(0, int(box[0]-10))
        w2 = min(w, int(box[2]+10))
        cropped_img = rgb[h1:h2, w1:w2]
        if cropped_img.size == 0:
            continue
        crop_images.append(cropped_img)
        crop_indices.append(idx)
    crop_features = get_img_feats_batch(crop_images)

    # mask = np.full(len(outputs['labels']), False, dtype=bool)
    for crop_pos, idx in enumerate(crop_indices):
        img_feats = np.expand_dims(crop_features[crop_pos], axis=0)
        sorted_obj_names, sorted_scores = get_nn_text(object_list, object_name_feats, img_feats)
        logger.debug(
            "Confirming for %s score=%s sorted_obj_names=%s sorted_scores=%s",
            outputs['labels'][idx],
            outputs['scores'][idx],
            sorted_obj_names,
            sorted_scores,
        )
        # plt.imshow(cropped_img)
        # plt.show()
        label = outputs['labels'][idx]
        clip_conf = False
        for i in range(len(sorted_scores)):
            if i == 0:
                if sorted_obj_names[i] == label:
                    clip_conf = True
                    break
            elif sorted_scores[i] > 0.23 and label == sorted_obj_names[i]: # change from 0.25 to 0.23 for sauteeCarrot4
                clip_conf = True
                break
        if clip_conf and not is_duplicate_bbox(idx, outputs):
            if len(new_outputs['pred_masks']) == 0:
                new_outputs['pred_masks'] = np.expand_dims(outputs['pred_masks'][idx], axis=0)
                new_outputs['bbox_2d'] = np.expand_dims(outputs['bbox_2d'][idx], axis=0)
            else:
                new_outputs['pred_masks'] = np.concatenate((new_outputs['pred_masks'], np.expand_dims(outputs['pred_masks'][idx], axis=0)))
                new_outputs['bbox_2d'] = np.concatenate((new_outputs['bbox_2d'], np.expand_dims(outputs['bbox_2d'][idx], axis=0)))
            new_outputs['scores'] = np.append(new_outputs['scores'], outputs['scores'][idx])
            new_outputs['labels'] = np.append(new_outputs['labels'], outputs['labels'][idx])
        # plt.imshow(cropped_img)
        # plt.show()
    new_outputs['total_detections'] = len(new_outputs['scores'])

    logger.info(
        "[Detections] frame=%s raw=%s kept=%s labels=%s",
        step_idx,
        outputs['total_detections'],
        new_outputs['total_detections'],
        list(new_outputs['labels']),
    )
    logger.debug("outputs after scores: %s", new_outputs['scores'])
    logger.debug("outputs after pred_masks shape: %s", new_outputs['pred_masks'].shape)
    logger.debug("outputs after bbox_2d shape: %s", new_outputs['bbox_2d'].shape)
    if should_write_debug_artifacts(args):
        clip_processed_im = plot_results(rgb, new_outputs['scores'], new_outputs['bbox_2d'], new_outputs['labels'], new_outputs['pred_masks'])
        plt.imshow(clip_processed_im)
        plt.axis("off")
        plt.savefig(f"real_world/state_summary/{args.folder_name}/mdetr_obj_det/clip_processed_det/{step_idx}.png")
        plt.close()
    return new_outputs

def obj_det(rgb, object_list, detector, step_idx, args):
    if should_write_debug_artifacts(args):
        plt.imshow(rgb)
        plt.axis("off")
        plt.savefig(f"real_world/state_summary/{args.folder_name}/mdetr_obj_det/images/{step_idx}.png")
        plt.close()

    if args.obj_det == 'detic':
        # print("object detection using detic")
        retval = detector.detect_object_from_img(rgb, args=args, step_idx=step_idx)
        # print("retval.keys: ", dir(retval['instances']))
        outputs = {
            'total_detections': len(retval['instances'].pred_classes),
            'labels': [object_list[idx] for idx in retval['instances'].pred_classes],
            'scores': retval['instances'].scores,
            'pred_masks': retval['instances'].pred_masks,
            'bbox_2d': retval['instances'].pred_boxes.tensor.numpy().astype(int),
        }
    elif args.obj_det == 'mdetr':
        logger.debug("object detection using mdetr")
        outputs = {
                'total_detections': 0,
                'labels': np.array([]),
                'scores': np.array([]),
                'pred_masks': np.array([]),
                'bbox_2d': np.array([])
            }
        im = Image.fromarray(rgb)
        prepared_img = prepare_inference_image(im)
        for idx, single_obj_prompt in enumerate(object_list):
            retval = plot_inference_segmentation(
                im,
                single_obj_prompt,
                detector,
                detection_threshold=args.mdetr_confidence_threshold,
                prepared_img=prepared_img,
                include_visualization=should_write_debug_artifacts(args),
            )
            logger.debug(
                f"[MDETR] prompt='{single_obj_prompt}' "
                f"threshold={args.mdetr_confidence_threshold:.2f} "
                f"detections={len(retval['probs'])}"
            )
            if len(outputs['pred_masks']) == 0:
                outputs['pred_masks'] = retval['masks']
                outputs['bbox_2d'] = retval['bbox_2d']
            else:
                outputs['pred_masks'] = np.concatenate((outputs['pred_masks'], retval['masks']))
                outputs['bbox_2d'] = np.concatenate((outputs['bbox_2d'], retval['bbox_2d']))
            outputs['scores'] = np.concatenate((outputs['scores'], retval['probs']))
            outputs['labels'] = np.concatenate((outputs['labels'], retval['labels']))
            outputs['total_detections'] = len(outputs['scores'])

        if should_write_debug_artifacts(args) and retval['im'] is not None:
            plt.imshow(retval['im'])
            plt.axis("off")
            plt.savefig(f"real_world/state_summary/{args.folder_name}/mdetr_obj_det/det/{step_idx}.png")
            plt.close()
        logger.debug("total_detections: %s", outputs['total_detections'])
    
    return outputs

def edit_label(label_arr):
    logger.debug("mdetr labels: %s", label_arr)
    real_world_label_arr = []
    for old_label in label_arr:
        if old_label in real_world_name_map:
            real_world_label_arr.append(real_world_name_map[old_label])
        else:
            real_world_label_arr.append(old_label)
    
    ctr_dict = {}
    for i in range(len(real_world_label_arr)):
        label = real_world_label_arr[i]
        if label not in ctr_dict:
            ctr_dict[label] = [i]
        else:
            ctr_dict[label].append(i)
    for k, v in ctr_dict.items():
        if len(v) > 1:
            counter = 1
            for idx in v:
                real_world_label_arr[idx] = f'{k}-{counter}'
                counter += 1
    logger.debug("real world labels: %s", real_world_label_arr)
    return real_world_label_arr

def create_bbox_gpu(points):
    if points is None or len(points) == 0:
        return o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array([0.0, 0.0, 0.0]),
            max_bound=np.array([0.0, 0.0, 0.0]),
        )

    if isinstance(points, np.ndarray):
        tensor = torch.from_numpy(points).float()
    elif torch.is_tensor(points):
        tensor = points.float()
    else:
        tensor = torch.tensor(points, dtype=torch.float32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor = tensor.to(device)
    min_coords = tensor.min(dim=0)[0].detach().cpu().numpy()
    max_coords = tensor.max(dim=0)[0].detach().cpu().numpy()
    return o3d.geometry.AxisAlignedBoundingBox(min_bound=min_coords, max_bound=max_coords)

# TODO: add global pcd (need to think about the case of moving objects)
def get_scene_graph(
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
    object_name_feats=None,
    detection_history=None,
    local_graph_history=None,
    profiler=None,
):
    pcd_dict, bbox2d_dict = {}, {}
    local_sg = SceneGraph()
    
    # object detection and segmentation
    if detector is not None:
        if profiler is not None:
            with profiler.stage("mdetr"):
                outputs = obj_det(rgb, object_list, detector, step_idx, args)
        else:
            outputs = obj_det(rgb, object_list, detector, step_idx, args)
        if should_write_debug_artifacts(args):
            with open(f'real_world/state_summary/{args.folder_name}/{args.obj_det}_obj_det/det/{step_idx}.pickle', 'wb') as f:
                pickle.dump(outputs, f)
    else:
        if args.obj_det == 'detic':
            with open(f'real_world/state_summary/{args.folder_name}/{args.obj_det}_obj_det/det/{step_idx}.pickle', 'rb') as f:
                outputs = pickle.load(f)
        elif args.obj_det == 'mdetr':
             with open(f'real_world/state_summary/{args.folder_name}/{args.obj_det}_obj_det/clip_processed_det/{step_idx}.pickle', 'rb') as f:
                outputs = pickle.load(f)

    if outputs['total_detections'] == 0:
        # nothing is detected in this frame
        logger.info(
            f"Nothing is detected in the current frame {step_idx} "
            f"for task {args.folder_name}"
        )
        return local_sg, bbox3d_dict, pcd_dict, bbox2d_dict

    # confirming the obj detector answers using CLIP
    if args.obj_det == 'mdetr':
        if profiler is not None:
            with profiler.stage("clip_confirm"):
                outputs = confirm_obj_det(args, rgb, outputs, object_list, step_idx, object_name_feats=object_name_feats)
        else:
            outputs = confirm_obj_det(args, rgb, outputs, object_list, step_idx, object_name_feats=object_name_feats)
        outputs['labels'] = edit_label(outputs['labels'])
        outputs['old_labels'] = outputs['labels']
        if should_write_debug_artifacts(args):
            with open(f'real_world/state_summary/{args.folder_name}/{args.obj_det}_obj_det/clip_processed_det/{step_idx}.pickle', 'wb') as f:
                pickle.dump(outputs, f)

    if detection_history is not None:
        detection_history[step_idx] = outputs

    # # If mupltiple detected objects have the exact same name
    # if len(outputs['labels']) != len(set(outputs['labels'])):
    #     outputs['labels'] = edit_label(outputs['labels'])

    # RGB-D to point cloud
    depth_x_map, depth_y_map, depth_z_map, valid_depth_mask = depth_to_point_cloud_components(intrinsics_matrix, depth)
    pointcloud_stage = profiler.stage("pointcloud_cleanup") if profiler is not None else nullcontext()
    with pointcloud_stage:
        candidates = []
        candidate_sizes = []
        for idx in range(outputs['total_detections']):
            label = outputs['labels'][idx]
            if label.split("-")[0] in distractor_list:
                logger.debug("filtering out: %s", label)
                continue
            if outputs['scores'][idx] < 0:
                continue

            point_mask = np.asarray(outputs['pred_masks'][idx]).astype(bool) & valid_depth_mask
            if not np.any(point_mask):
                continue
            point_3d = np.stack(
                (
                    depth_x_map[point_mask],
                    depth_y_map[point_mask],
                    depth_z_map[point_mask],
                ),
                axis=1,
            )
            candidates.append((label, outputs['bbox_2d'][idx], point_3d))
            candidate_sizes.append(int(point_3d.shape[0]))

        if candidates:
            raw_clouds = [item[2] for item in candidates]
            downsampled_clouds = voxel_downsample_gpu_batch(raw_clouds, voxel_size=0.01)
            downsampled_sizes = [int(cloud.shape[0]) for cloud in downsampled_clouds]

            # Outlier filtering is expensive for large clouds; skip when cloud is too large.
            outlier_max_points = int(getattr(args, "outlier_filter_max_points", 30000))
            filtered_clouds = [None] * len(downsampled_clouds)
            to_filter_indices = [
                i for i, cloud in enumerate(downsampled_clouds)
                if 0 < len(cloud) <= outlier_max_points
            ]
            skipped_large_indices = [
                i for i, cloud in enumerate(downsampled_clouds)
                if len(cloud) > outlier_max_points
            ]

            if to_filter_indices:
                filtered_subset = outlier_removal_gpu_batch(
                    [downsampled_clouds[i] for i in to_filter_indices],
                    nb_neighbors=1500,
                    std_ratio=0.1,
                )
                for local_idx, cloud in zip(to_filter_indices, filtered_subset):
                    filtered_clouds[local_idx] = cloud

            for i, cloud in enumerate(downsampled_clouds):
                if filtered_clouds[i] is None:
                    filtered_clouds[i] = cloud

            if logger.isEnabledFor(logging.INFO):
                pre_min = min(candidate_sizes) if candidate_sizes else 0
                pre_max = max(candidate_sizes) if candidate_sizes else 0
                post_min = min(downsampled_sizes) if downsampled_sizes else 0
                post_max = max(downsampled_sizes) if downsampled_sizes else 0
                logger.info(
                    "[OutlierCoverage] frame=%s candidates=%s filtered=%s skipped_large=%s threshold=%s pre_pts[min,max]=(%s,%s) post_pts[min,max]=(%s,%s)",
                    step_idx,
                    len(candidates),
                    len(to_filter_indices),
                    len(skipped_large_indices),
                    outlier_max_points,
                    pre_min,
                    pre_max,
                    post_min,
                    post_max,
                )

            for (label, bbox2d, _), filtered in zip(candidates, filtered_clouds):
                pcd_dict[label] = filtered
                if label in ["fridge", "coffee machine", "table"] and label in total_points_dict:
                    total_points_dict[label] = np.concatenate((total_points_dict[label], pcd_dict[label]))
                else:
                    total_points_dict[label] = pcd_dict[label]

                box = create_bbox_gpu(total_points_dict[label])
                bbox3d_dict[label] = box
                bbox2d_dict[label] = bbox2d
    
    # for saving the pcd in this frame to file
    total_points, total_colors = None, None
    for i, label in enumerate(total_points_dict.keys()):
        if total_points is None:
            total_points = torch.tensor(total_points_dict[label])
            c = torch.tensor(COLORS[i%len(COLORS)])
            total_colors = c.repeat(len(total_points_dict[label]), 1)
        else:
            total_points = torch.cat((total_points, torch.tensor(total_points_dict[label])), 0)
            c = torch.tensor(COLORS[i%len(COLORS)])
            total_colors = torch.cat((total_colors, c.repeat(len(total_points_dict[label]), 1)), 0)
        # total_points, total_colors = visualize_box(total_points, total_colors, box.get_box_points(), [1, 0, 0])
    
    # Save pcd to file
    if total_points is not None and should_write_debug_artifacts(args):
        saved_pcd = o3d.geometry.PointCloud()
        saved_pcd.points = o3d.utility.Vector3dVector(total_points)
        saved_pcd.colors = o3d.utility.Vector3dVector(total_colors)
        o3d.io.write_point_cloud("real_world/scene/{}/scene_{}.ply".format(args.folder_name, step_idx), saved_pcd)

    # Generate local scene graph
    local_sg = SceneGraph()
    for label in pcd_dict.keys():
        # bbox = get_2d_bbox_from_3d_pcd(step_idx, event, label, total_points_dict)
        node = Node(label, 
                    object_id=label, 
                    pos3d=bbox3d_dict[label].get_center(), 
                    corner_pts=np.array(bbox3d_dict[label].get_box_points()), 
                    bbox2d=bbox2d_dict[label], 
                    pcd=total_points_dict[label],
                    depth=None)
        local_sg.add_node_wo_edge(node)
        local_sg.add_node(node, rgb=None, infer_state=False)

    local_sg.add_object_states_batch(rgb)

    obj_held = local_sg.add_agent(
        meta_data['data/gripper_pos'][step_idx],
        step_idx,
        args,
        detection_history=detection_history,
        local_graph_history=local_graph_history,
    )

    return local_sg, bbox3d_dict, total_points_dict, bbox2d_dict
