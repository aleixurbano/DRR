"""
audio.py
--------
Identifies acoustic events in a recorded episode by combining audio embeddings
(wav2clip) with egocentric image embeddings (CLIP).

The main public API returns a canonical per-segment label map so downstream
captioning and reasoning do not inherit per-frame spillover from long clips.
"""

import os
import itertools
import pickle
import warnings

import numpy as np
from reflect.perception.clip import get_text_feats, get_nn_text_w_audio, get_img_feats
from reflect.core.constants import NAME_MAP

# ---------------------------------------------------------------------------
# Mapping from raw audio file names to human-readable event labels.
# Multiple files can share the same label (e.g. both faucet and pour → "water runs in sink").
# ---------------------------------------------------------------------------
audio2label = {
    "toggle-on-faucet.wav":        "water runs in sink",
    "toggle-on-toaster.wav":       "toaster turns on",
    "toggle-on-stoveburner.wav":   "stove burner turns on",
    "drop-pot.wav":                "object drops or cracks on hard surface",
    "drop-plastic-bowl.wav":       "object drops or cracks on hard surface",
    "drop-egg.wav":                "object drops or cracks on hard surface",
    "slice-bread.wav":             "slice bread",
    "toggle-on-microwave.wav":     "microwave turns on",
    "toggle-on-coffeemachine.wav": "coffee machine turns on",
    "pour-water-in-sink.wav":      "water runs in sink",
    "open-fridge.wav":             "fridge opens",
    "close-fridge.wav":            "fridge closes",
    "open-microwave.wav":          "microwave opens",
    "close-microwave.wav":         "microwave closes",
    "crack-egg.wav":               "egg cracks",
}

# Deduplicated list of all possible text event labels used for CLIP retrieval.
TEXT_LIST = list(set(audio2label.values()))

_WAV2CLIP_IMPORT_ERROR = None
_MOVIEPY_IMPORT_ERROR = None
_AUDIO_IMPORT_WARNING_EMITTED = False

try:
    import wav2clip
except ImportError as exc:
    wav2clip = None
    _WAV2CLIP_IMPORT_ERROR = exc

try:
    from PIL import Image
    from moviepy import AudioFileClip, VideoFileClip
except ImportError as exc:
    Image = None
    AudioFileClip = None
    VideoFileClip = None
    _MOVIEPY_IMPORT_ERROR = exc

model = None

AUDIO_VOLUME_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_ranges(frames):
    """Convert an unsorted collection of frame indices into contiguous (start, end) ranges.

    Parameters
    ----------
    frames : iterable of int
        Frame indices that contain sound.

    Yields
    ------
    (int, int)
        Inclusive (start_frame, end_frame) pairs for each contiguous run.
    """
    for _, group in itertools.groupby(
        enumerate(sorted(set(frames))), lambda t: t[1] - t[0]
    ):
        group = list(group)
        yield group[0][1], group[-1][1]


def _get_total_frames(data_path):
    """Return the number of 1 FPS ego frames recorded for the episode."""
    ego_img_dir = os.path.join(data_path, "ego_img")
    if os.path.isdir(ego_img_dir):
        return len(
            [
                name
                for name in os.listdir(ego_img_dir)
                if name.startswith("img_step_") and name.endswith(".png")
            ]
        )

    clip = VideoFileClip(os.path.join(data_path, "original-video.mp4"))
    try:
        return int(clip.fps * clip.duration)
    finally:
        clip.close()


def _load_audio_source(data_path):
    """Prefer the lossless sidecar WAV, falling back to the video's audio track."""
    sidecar_path = os.path.join(data_path, "original-audio.wav")
    if os.path.exists(sidecar_path):
        return None, AudioFileClip(sidecar_path)

    video_clip = VideoFileClip(os.path.join(data_path, "original-video.mp4"))
    if video_clip.audio is None:
        video_clip.close()
        return None, None
    return video_clip, video_clip.audio


def _safe_max_volume(audio_clip, start_sec, end_sec):
    """Return max volume for a segment, treating missing/invalid audio as silence."""
    try:
        subclip = audio_clip.subclip(start_sec, end_sec)
        return float(subclip.max_volume())
    except Exception:
        return 0.0


def _build_text_list(object_list):
    """Build the scene-aware label candidates used for retrieval."""
    scene_obj_names = {NAME_MAP.get(obj, obj.lower()) for obj in object_list}
    return list(
        {
            label
            for label in TEXT_LIST
            if "drops" in label or any(name in label for name in scene_obj_names)
        }
    )


def _audio_stack_available():
    return wav2clip is not None and Image is not None and AudioFileClip is not None and VideoFileClip is not None


def _audio_stack_error():
    if _WAV2CLIP_IMPORT_ERROR is not None:
        return _WAV2CLIP_IMPORT_ERROR
    return _MOVIEPY_IMPORT_ERROR


def _ensure_audio_stack():
    global model
    if not _audio_stack_available():
        raise ImportError(
            "Audio detection requires the optional audio stack (`wav2clip`, `moviepy`, and PIL)."
        ) from _audio_stack_error()

    if model is None:
        model = wav2clip.get_model()
        model.eval()


def warn_audio_stack_unavailable_once():
    global _AUDIO_IMPORT_WARNING_EMITTED
    if _AUDIO_IMPORT_WARNING_EMITTED:
        return
    _AUDIO_IMPORT_WARNING_EMITTED = True
    warnings.warn(
        "Audio detection is unavailable because the optional audio stack is missing. "
        "Install `wav2clip` and related media dependencies, or run with `--with-audio 0`.",
        RuntimeWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def process_sound_segments(data_path, object_list=None):
    """Detect and classify audible segments in a recorded episode.

    For every contiguous segment of audible frames the function:
      1. Embeds the audio with wav2clip.
      2. Embeds the ego-centric frame at the start of the segment with CLIP.
      3. Ranks candidate text labels using a combined audio-visual similarity
         score.

    Parameters
    ----------
    data_path : str
        Directory that contains:
          - ``original-video.mp4``        - full episode recording.
          - ``original-audio.wav``        - optional lossless sidecar audio.
          - ``interact_actions.pickle``   - dict keyed by frame indices at which
            the agent performed an interaction.
          - ``ego_img/img_step_<n>.png``  - egocentric images per step.
    object_list : list of str, optional
        Object class names present in the scene. Used to filter the candidate
        text labels to those relevant to the current scene.

    Returns
    -------
    list[dict]
        Each entry contains ``start_frame``, ``end_frame``, ``canonical_frame``,
        and ``label`` for one contiguous audible segment.
    """
    _ensure_audio_stack()
    object_list = object_list or []

    # -----------------------------------------------------------------
    # Step 1: find all frames that contain significant audio energy.
    # -----------------------------------------------------------------
    carrier_clip, audio_clip = _load_audio_source(data_path)
    if audio_clip is None:
        return []

    try:
        total_frames = _get_total_frames(data_path)
        frames_w_sound = [
            frame
            for frame in range(total_frames)
            if _safe_max_volume(audio_clip, frame, frame + 1) > AUDIO_VOLUME_THRESHOLD
        ]

        # Group consecutive audible frames into contiguous (start, end) ranges.
        frame_ranges = list(_to_ranges(frames_w_sound))

        # -----------------------------------------------------------------
        # Step 2: build a scene-aware candidate label list.
        # Include generic "drops" labels and labels that mention any object
        # present in the current scene.
        # -----------------------------------------------------------------
        text_list = _build_text_list(object_list)
        if not text_list:
            return []

        text_feats = get_text_feats(text_list)

        # Load the set of frames at which the agent performed an interaction.
        with open(os.path.join(data_path, "interact_actions.pickle"), "rb") as f:
            interact_actions = pickle.load(f)
        interact_steps = set(interact_actions.keys())  # keep current semantics unchanged

        # -----------------------------------------------------------------
        # Step 3: classify each audible segment.
        # -----------------------------------------------------------------
        pred_segments = []

        for start_frame, end_frame in frame_ranges:
            sub_audio = audio_clip.subclip(start_frame, end_frame + 1)
            signal = sub_audio.to_soundarray().astype(np.float32)

            if signal.size == 0:
                continue

            # Convert stereo → mono by averaging channels.
            if signal.ndim == 2 and signal.shape[1] == 2:
                signal = signal.mean(axis=1)

            max_abs = np.max(np.abs(signal))
            if max_abs == 0:
                continue
            norm_signal = signal / max_abs

            # Compute and L2-normalise the audio embedding.
            audio_feats = wav2clip.embed_audio(norm_signal, model)[0]
            audio_feats /= np.linalg.norm(audio_feats)

            # Compute the visual embedding for the first frame of the segment.
            img_path = os.path.join(data_path, "ego_img", f"img_step_{start_frame + 1}.png")
            img = np.array(Image.open(img_path).convert("RGB"))
            img_feats = get_img_feats(img)[0]

            # Keep the current weighting semantics unchanged.
            segment_frames = set(range(start_frame, end_frame + 1))
            weight = 2 if segment_frames & interact_steps else 0

            sorted_texts, _ = get_nn_text_w_audio(
                text_list, text_feats, img_feats, audio_feats, weight=weight
            )

            pred_segments.append(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "canonical_frame": start_frame,
                    "label": sorted_texts[0],
                }
            )

        return pred_segments
    finally:
        if carrier_clip is not None:
            carrier_clip.close()
        else:
            audio_clip.close()


def process_sound_framewise(data_path, object_list=None):
    """Return the dense per-frame label map for debugging visualisations."""
    pred_sounds = {}
    for segment in process_sound_segments(data_path, object_list):
        for frame in range(segment["start_frame"], segment["end_frame"] + 1):
            pred_sounds[frame] = segment["label"]
    return pred_sounds


def process_sound(data_path, object_list=None):
    """Return one canonical detected sound label per audible segment."""
    return {
        segment["canonical_frame"]: segment["label"]
        for segment in process_sound_segments(data_path, object_list)
    }
