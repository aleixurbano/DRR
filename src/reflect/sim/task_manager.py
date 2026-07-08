"""
task_manager.py

Utility classes and functions for AI2-THOR task execution, failure injection,
navigation path-finding, data saving, and video generation.
"""

import os
import PIL
import json
import pickle
import imageio
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Optional
from collections import deque
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
from reflect.core.constants import *
from reflect.perception.point_cloud import *


# ---------------------------------------------------------------------------
# TaskUtil class
# ---------------------------------------------------------------------------

class TaskUtil:
    """
    Central utility object for a single task episode.

    Holds the AI2-THOR controller, navigation grid, action history,
    failure-injection state, and helper methods used throughout execution.
    """

    # Supported failure types that can be injected into a task.
    FAILURE_TYPES = ["drop", "failed_action", "missing_step"]

    # Action primitives that involve direct object interaction.
    INTERACT_ACTION_PRIMITIVES = [
        "put_on", "put_in", "pick_up", "slice_obj",
        "toggle_on", "toggle_off", "open_obj", "close_obj",
        "pour", "crack_obj",
    ]

    def __init__(
        self,
        folder_name: str,
        controller,
        reachable_positions: List[Dict[str, float]],
        failure_injection: bool,
        index: int,
        repo_path: str,
        chosen_failure: Optional[str],
        failure_injection_params: dict,
        counter: int = 0,
        replan: bool = False,
    ):
        """
        Args:
            folder_name:              Relative path used as the task identifier
                                      (e.g. "kitchen/task_0").
            controller:               AI2-THOR controller instance.
            reachable_positions:      List of {x, y, z} dicts for valid floor positions.
            failure_injection:        Whether to inject a failure into this episode.
            index:                    Episode index, used to cycle through failure types.
            repo_path:                Absolute path to the repository root.
            chosen_failure:           Explicitly specify which failure type to inject,
                                      or None to cycle automatically.
            failure_injection_params: Extra parameters for the chosen failure.
            counter:                  Step counter to resume from (default 0).
            replan:                   Whether this episode is a replanning attempt.
        """
        self.counter = counter
        self.repo_path = repo_path
        self.folder_name = folder_name

        # If injecting a failure, append a numeric suffix so each injection
        # gets its own output directory.
        if failure_injection:
            self.specific_folder_name = self._get_folder_name(folder_name, index + 1)
        else:
            self.specific_folder_name = folder_name

        self.controller = controller
        self.grid = self._create_navigation_grid()
        self.interact_actions: dict = {}
        self.nav_actions: dict = {}
        self.reachable_positions = reachable_positions
        self.reachable_points = self._get_2d_reachable_points()
        self.failure_added = False
        self.objs_w_unk_loc: list = []
        self.unity_name_map = self._get_unity_name_map()
        self.sounds: dict = {}
        self.failure_injection_params = failure_injection_params
        self.gt_failure: dict = {}

        # Determine which failure type to inject.
        if failure_injection and chosen_failure is None:
            failure_idx = index % len(self.FAILURE_TYPES)
            self.chosen_failure = self.FAILURE_TYPES[failure_idx]
            print(f"[INFO] Chosen failure: {self.chosen_failure}")
        else:
            self.chosen_failure = chosen_failure

        # Load any failures that were previously injected for this task.
        self.failures_already_injected: dict = {}
        pickle_path = (
            f"{self.repo_path}/thor_tasks"
            f"/{folder_name.split('/')[0]}/{folder_name.split('/')[1]}.pickle"
        )
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as fh:
                self.failures_already_injected = pickle.load(fh)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_unity_name_map(self) -> Dict[str, str]:
        """
        Build a mapping from Unity object names to human-readable labels for
        object types that appear more than once in the scene.

        For example, if there are two cabinets their Unity names are mapped to
        "Cabinet-1" and "Cabinet-2".

        Returns:
            dict mapping Unity object name → "ObjectType-N" string.
        """
        # Object types that can appear multiple times and need labelling.
        multi_instance_types = ["CounterTop", "StoveBurner", "Cabinet", "Faucet", "Sink"]

        # Count how many times each type appears in the scene.
        type_counts: Dict[str, int] = {}
        for obj in self.controller.last_event.metadata["objects"]:
            obj_type = obj["objectType"]
            if obj_type in multi_instance_types:
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

        # Keep only types that appear more than once (singletons don't need labels).
        types_to_label = [t for t in multi_instance_types if type_counts.get(t, 0) > 1]

        # Assign sequential numeric labels.
        unity_name_map: Dict[str, str] = {}
        for obj_type in types_to_label:
            instance_counter = 0
            for obj in self.controller.last_event.metadata["objects"]:
                if obj["objectType"] == obj_type:
                    instance_counter += 1
                    unity_name_map[obj["name"]] = f"{obj_type}-{instance_counter}"

        return unity_name_map

    def _get_folder_name(self, folder_name: str, folder_idx: int) -> str:
        """Return a folder name with a numeric suffix for failure injection episodes."""
        return f"{folder_name}-{folder_idx}"

    def _create_navigation_grid(
        self,
        grid_size: float = 0.25,
        grid_min: float = -5,
        grid_max: float = 5.1,
    ) -> np.ndarray:
        """
        Create a 2-D spatial grid used for navigation path-finding.

        Args:
            grid_size: Cell size in metres.
            grid_min:  Minimum coordinate value (metres).
            grid_max:  Maximum coordinate value (metres), exclusive.

        Returns:
            NumPy array of shape (N, N, 2) containing (x, z) coordinates.
        """
        return np.mgrid[grid_min:grid_max:grid_size, grid_min:grid_max:grid_size].transpose(1, 2, 0)

    def _get_2d_reachable_points(self) -> np.ndarray:
        """
        Convert the reachable-positions list to a flat (N, 2) array of (x, z) pairs.

        The y-component (vertical) is ignored because navigation is planar.

        Returns:
            NumPy array of shape (N, 2).
        """
        return np.array([[p["x"], p["z"]] for p in self.reachable_positions])

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_reasoning_json(self) -> dict:
        """Load and return the reasoning JSON produced for this episode."""
        with open(os.path.join(self.repo_path, 'state_summary', self.specific_folder_name, 'reasoning.json')) as fh:
            return json.load(fh)


# ---------------------------------------------------------------------------
# Standalone navigation utilities
# ---------------------------------------------------------------------------

def closest_position(
    object_position: Dict[str, float],
    reachable_positions: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Return the reachable position closest to *object_position* on the ground plane.

    Only the x/z axes are considered; y (vertical) is intentionally ignored.

    Args:
        object_position:     Target position dict with at least "x" and "z" keys.
        reachable_positions: List of candidate positions, each with "x" and "z" keys.

    Returns:
        The entry from *reachable_positions* with the smallest x/z distance.
    """
    best_pos = reachable_positions[0]
    min_dist_sq = float("inf")

    for pos in reachable_positions:
        dist_sq = (pos["x"] - object_position["x"]) ** 2 + (pos["z"] - object_position["z"]) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_pos = pos

    return best_pos

# ---------------------------------------------------------------------------
# BFS path-finding
# ---------------------------------------------------------------------------

class Node:
    """
    A single cell in the BFS grid used by :func:`find_path`.

    Attributes:
        x:      Row index in the grid.
        y:      Column index in the grid.
        parent: The node from which this node was reached (used to reconstruct paths).
    """

    def __init__(self, x: int, y: int, parent: Optional["Node"] = None):
        self.x = x
        self.y = y
        self.parent = parent

    def __repr__(self) -> str:
        return str((self.x, self.y))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.x == other.x and self.y == other.y


# Row/column offsets for the four cardinal directions (up, left, right, down).
DELTA_ROW = [-1, 0, 0, 1]
DELTA_COL = [0, -1, 1, 0]


def _is_valid_cell(
    row: int,
    col: int,
    grid_size: int,
    reachable_set: frozenset,
    grid: list,
) -> bool:
    """
    Check whether grid cell (row, col) is within bounds and is a reachable position.

    Args:
        row:           Row index to test.
        col:           Column index to test.
        grid_size:     Side length of the square grid.
        reachable_set: Frozenset of (x, z) tuples for valid positions.
        grid:          The navigation grid as a nested Python list.

    Returns:
        True if the cell is valid and reachable, False otherwise.
    """
    if row < 0 or col < 0 or row >= grid_size or col >= grid_size:
        return False
    return tuple(grid[row][col]) in reachable_set


def _reconstruct_path(node: Optional[Node], path: List[Node]) -> None:
    """
    Recursively walk parent pointers to build the path list in-place.

    Args:
        node: Terminal node whose ancestry encodes the full path.
        path: List to which nodes are appended in order from source to destination.
    """
    if node is not None:
        _reconstruct_path(node.parent, path)
        path.append(node)


def find_path(
    matrix: np.ndarray,
    start_row: int = 0,
    start_col: int = 0,
    target_pos: Optional[List[int]] = None,
    reachable_points: Optional[np.ndarray] = None,
) -> Optional[List[Node]]:
    """
    Find the shortest path between two grid cells using Breadth-First Search.

    Args:
        matrix:           N×N navigation grid as a NumPy array.
        start_row:        Row index of the source cell.
        start_col:        Column index of the source cell.
        target_pos:       [row, col] of the destination cell.
        reachable_points: (M, 2) array of valid world-space [x, z] positions.

    Returns:
        Ordered list of :class:`Node` objects from source to destination,
        or None if no path exists.
    """
    grid = matrix.tolist()
    grid_size = len(grid)
    reachable_set = frozenset(map(tuple, reachable_points.tolist()))

    queue: deque = deque()
    source = Node(start_row, start_col)
    queue.append(source)
    visited: set = {(source.x, source.y)}

    while queue:
        current = queue.popleft()
        curr_row, curr_col = current.x, current.y

        # Destination reached - reconstruct and return the path.
        if curr_row == target_pos[0] and curr_col == target_pos[1]:
            path: List[Node] = []
            _reconstruct_path(current, path)
            return path

        # Explore all four cardinal neighbours.
        for d_row, d_col in zip(DELTA_ROW, DELTA_COL):
            next_row = curr_row + d_row
            next_col = curr_col + d_col

            if _is_valid_cell(next_row, next_col, grid_size, reachable_set, grid):
                next_node = Node(next_row, next_col, current)
                key = (next_node.x, next_node.y)
                if key not in visited:
                    visited.add(key)
                    queue.append(next_node)

    # No path found.
    return None


# ---------------------------------------------------------------------------
# Failure detection
# ---------------------------------------------------------------------------

def obj_is_blocked(task_util: "TaskUtil", src_obj_type: str) -> bool:
    """
    Determine whether *src_obj_type* is occluded by a target object in camera space.

    Two conditions must both be true for the source to be considered blocked:
      1. The Euclidean distance between the two objects is less than 0.3 m.
      2. The normalised depth component of the vector from target → source
         in camera space exceeds 0.75 (i.e. source is behind the target).

    Args:
        task_util:    Active :class:`TaskUtil` instance.
        src_obj_type: Object type string of the potentially blocked object.

    Returns:
        True if the source object is blocked, False otherwise.
    """
    target_obj_type = task_util.failure_injection_params["target_obj_type"]
    objects = task_util.controller.last_event.metadata["objects"]

    # Retrieve 3-D world positions of both objects.
    src_obj = next(obj for obj in objects if obj["objectType"] == src_obj_type)
    src_pos = np.array([src_obj["position"]["x"], src_obj["position"]["y"], src_obj["position"]["z"]])

    target_obj = next(obj for obj in objects if obj["objectType"] == target_obj_type)
    target_pos = np.array([target_obj["position"]["x"], target_obj["position"]["y"], target_obj["position"]["z"]])

    # Only carry out the full occlusion test when objects are close together.
    distance = np.linalg.norm(src_pos - target_pos)
    if distance >= 0.3:
        print(f"[INFO] {src_obj_type} is not blocked (distance {distance:.3f} m >= 0.3 m).")
        return False

    # Project both object positions into camera space.
    event = task_util.controller.last_event
    agent = event.metadata["agent"]
    camera_world_xyz = torch.as_tensor([agent["position"]["x"], agent["position"]["y"], agent["position"]["z"]])
    rotation = agent["rotation"]["y"]
    horizon = agent["cameraHorizon"]

    cam_target = world_space_xyz_to_camera_space_xyz(
        torch.tensor(np.expand_dims(target_pos, 0)).reshape(3, 1),
        camera_world_xyz, rotation, horizon,
    ).flatten()

    cam_src = world_space_xyz_to_camera_space_xyz(
        torch.tensor(np.expand_dims(src_pos, 0)).reshape(3, 1),
        camera_world_xyz, rotation, horizon,
    ).flatten()

    # Normalised direction vector from target to source in camera space.
    direction = cam_src - cam_target
    norm_direction = direction / np.linalg.norm(direction)

    # A positive z-component above the threshold means the source is behind the target.
    is_blocked = bool(norm_direction[2] > 0.75)
    print(f"[INFO] {src_obj_type} is {'blocked' if is_blocked else 'not blocked'}.")
    return is_blocked

# ---------------------------------------------------------------------------
# Data persistence helpers
# ---------------------------------------------------------------------------

def save_data(task: "TaskUtil", event, replan: bool = False) -> None:
    """
    Persist a single simulation step to disk.

    Saves the raw event object as a pickle file and the ego-centric RGB frame
    as a PNG image, both numbered by the current step counter.

    Args:
        task:   Active :class:`TaskUtil` instance.
        event:  AI2-THOR event returned by the last controller action.
        replan: If True, data is saved under the ``recovery/`` directory
                instead of ``thor_tasks/``.
    """
    task.counter += 1
    folder = "recovery" if replan else "thor_tasks"
    base_dir = f"{task.repo_path}/{folder}/{task.specific_folder_name}"

    # Ensure output directories exist.
    os.makedirs(f"{base_dir}/events", exist_ok=True)
    os.makedirs(f"{base_dir}/ego_img", exist_ok=True)

    # Save the full event object.
    with open(f"{base_dir}/events/step_{task.counter}.pickle", "wb") as fh:
        pickle.dump(event, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the ego-centric frame as a PNG.
    plt.imsave(f"{base_dir}/ego_img/img_step_{task.counter}.png", np.asarray(event.frame, order="C"))


def _write_video_with_audio(
    frame_filenames: List[str],
    img_dir: str,
    video_path: str,
    audio_path: str,
    sounds: dict,
    sound_dir: str,
) -> None:
    """
    Write an MP4 video from a sorted list of image filenames, optionally mixing
    in audio clips at specified start times.

    Args:
        frame_filenames: Sorted list of image file names (not full paths).
        img_dir:         Directory containing the image files.
        video_path:      Output path for the MP4 file.
        audio_path:      Output path for the lossless sidecar WAV.
        sounds:          Dict mapping start_time (seconds) → audio filename.
        sound_dir:       Directory containing the audio files.
    """
    # Write silent video from individual frames.
    with imageio.get_writer(video_path, mode="I", fps=1) as writer:
        for filename in frame_filenames:
            img = PIL.Image.open(os.path.join(img_dir, filename)).convert("RGB")
            writer.append_data(np.array(img))

    if not sounds:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return

    # Build a list of audio clips positioned at their respective start times.
    audio_clips = [
        AudioFileClip(os.path.join(sound_dir, filename)).with_start(start_time)
        for start_time, filename in sounds.items()
    ]

    # Mix audio onto the video and overwrite the silent file.
    clip = VideoFileClip(video_path)
    mixed_audio = CompositeAudioClip(audio_clips).with_duration(clip.duration)
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        mixed_audio.write_audiofile(audio_path, fps=48000, nbytes=2, logger=None)

        clip.audio = mixed_audio
        os.remove(video_path)
        clip.write_videofile(video_path, audio_codec="aac", logger=None)
    finally:
        mixed_audio.close()
        clip.close()
        for audio_clip in audio_clips:
            audio_clip.close()


def generate_video(task_util: "TaskUtil", recovery_video: bool) -> None:
    """
    Compile all saved ego-centric frames into an annotated MP4 video.

    The output is written to either ``thor_tasks/`` or ``recovery/`` depending
    on the *recovery_video* flag, and is named ``original-video.mp4`` or
    ``recovery-video.mp4`` respectively.

    Args:
        task_util:      Active :class:`TaskUtil` instance.
        recovery_video: If True, read frames from the ``recovery/`` directory
                        and produce a ``recovery-video.mp4``.
    """
    folder = "recovery" if recovery_video else "thor_tasks"
    img_dir = f"{task_util.repo_path}/{folder}/{task_util.specific_folder_name}/ego_img/"
    save_dir = f"{task_util.repo_path}/{folder}/{task_util.specific_folder_name}/"
    # Sounds live in main/assets/sounds/ (sibling of this file), not inside the episode dir.
    sound_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "sounds")

    # Sort frame filenames numerically by the step number embedded in the name.
    if not os.path.isdir(img_dir):
        return
    all_frames = os.listdir(img_dir)
    sorted_frames = sorted(all_frames, key=lambda f: int(os.path.splitext(f)[0].split("_")[-1]))

    if not sorted_frames:
        return

    video_name = "recovery-video.mp4" if recovery_video else "original-video.mp4"
    audio_name = "recovery-audio.wav" if recovery_video else "original-audio.wav"
    video_path = os.path.join(save_dir, video_name)
    audio_path = os.path.join(save_dir, audio_name)

    _write_video_with_audio(
        frame_filenames=sorted_frames,
        img_dir=img_dir,
        video_path=video_path,
        audio_path=audio_path,
        sounds=task_util.sounds,
        sound_dir=sound_dir,
    )


def make_controller(scene: str):
    """Create a standard AI2-THOR controller for the given scene."""
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    kwargs = dict(
        agentMode="default",
        massThreshold=None,
        scene=scene,
        visibilityDistance=1.5,
        gridSize=0.25,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        width=960,
        height=960,
        fieldOfView=60,
    )
    if not os.environ.get("REFLECT_VISIBLE_RENDERING"):
        kwargs["platform"] = CloudRendering
    return Controller(**kwargs)
