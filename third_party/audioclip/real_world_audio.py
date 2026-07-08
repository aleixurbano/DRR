import os
import sys
sys.path.append(os.path.abspath(f'{os.getcwd()}/AudioCLIP'))
import numpy as np
import torch
from AudioCLIP.model import AudioCLIP
from AudioCLIP.utils.transforms import ToTensor1D
try:
    from moviepy.editor import AudioFileClip
except ImportError:
    # moviepy>=2 exposes AudioFileClip at top level.
    from moviepy import AudioFileClip
import itertools
from constants import real_world_sound_map
from logging_utils import get_logger

torch.set_grad_enabled(False)

logger = get_logger(__name__)

MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
# derived from ESResNeXt
SAMPLE_RATE = 44100
# derived from CLIP
IMAGE_SIZE = 224
IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
IMAGE_STD = 0.26862954, 0.26130258, 0.27577711

LABELS = ['gas stove burner turns on', "water runs in sink", "cracking sound"]

aclp = AudioCLIP(pretrained=f'AudioCLIP/assets/{MODEL_FILENAME}')
audio_transforms = ToTensor1D()


def _subclip(clip, start, end):
    if hasattr(clip, "subclip"):
        return clip.subclip(start, end)
    return clip.subclipped(start, end)

def extract_audio_from_video(audio_path, volume_thresh):
    def to_ranges(iterable):
        iterable = sorted(set(iterable))
        for _, group in itertools.groupby(enumerate(iterable),
                                            lambda t: t[1] - t[0]):
            group = list(group)
            yield group[0][1], group[-1][1]

    pred_sounds = {}
    input_audio = AudioFileClip(audio_path)
    if hasattr(input_audio, "set_fps"):
        input_audio = input_audio.set_fps(SAMPLE_RATE)
    else:
        input_audio = input_audio.with_fps(SAMPLE_RATE)
    duration = int(input_audio.duration)
    frames_w_sound = []
    for cur_time in range(0, duration, 4):
        if cur_time+4 > duration:
            break
        subaudio = _subclip(input_audio, cur_time, cur_time+4)
        max_volume = subaudio.max_volume()
        if max_volume > volume_thresh:
            frames_w_sound += range(cur_time, cur_time+4)

    sound_ranges = list(to_ranges(frames_w_sound))
    logger.debug("sound ranges: %s", sound_ranges)
    
    filtered_sound_ranges = []
    for sound_range in sound_ranges:
        if sound_range[1] - sound_range[0] < 4:
            subaudio = _subclip(input_audio, sound_range[0], sound_range[1])
            max_volume = subaudio.max_volume()
            logger.debug("max volume: %s", max_volume)
            if max_volume > 0.5:
                filtered_sound_ranges.append(sound_range)
        else:
            filtered_sound_ranges.append(sound_range)
    logger.debug("filtered sound ranges: %s", filtered_sound_ranges)

    # plt.figure()
    # f, axarr = plt.subplots(len(sound_ranges), 1, figsize=(20, 16))
    tracks, max_volumes = [], []
    for idx, frame_range in enumerate(filtered_sound_ranges):
        logger.debug("FRAME %s", frame_range)
        if frame_range[0] + 5 > duration:
            break
        sub_audio = _subclip(input_audio, frame_range[0], frame_range[0]+5)
        max_volumes.append(sub_audio.max_volume())
        signal = sub_audio.to_soundarray().astype(np.float32)
        tracks.append(signal)

    return tracks, filtered_sound_ranges, max_volumes

def format_time(total_seconds):
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f'{minutes:.0f}:{seconds:02.0f}'

def get_sound_events(audio_path, volume_thresh=0.05):
    tracks, sound_ranges, max_volumes = extract_audio_from_video(audio_path, volume_thresh)
    if len(tracks) == 0:
        return {}
    elif len(tracks) == 1:
        tracks = [tracks[0], tracks[0]]
    audios = torch.stack([audio_transforms(track.reshape(1, -1)) for track in tracks])
    if "makeCoffee" in audio_path:
        labels = LABELS + ["coffee machine turns on"]
    else:
        labels = LABELS
    texts = [[label] for label in labels]

    ((audio_features, _, _), _), _ = aclp(audio=audios)
    ((_, _, text_features), _), _ = aclp(text=texts)

    audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
    text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)

    scale_audio_text = torch.clamp(aclp.logit_scale_at.exp(), min=1.0, max=100.0)
    logits_audio_text = scale_audio_text * audio_features @ text_features.T

    sound_events = {}
    confidence = logits_audio_text.softmax(dim=1)
    for idx in range(len(sound_ranges)):
        max_volume = max_volumes[idx]
        if sound_ranges[idx][1] - sound_ranges[idx][0] < 8 and max_volume > 0.8:
            sound_events[tuple(sound_ranges[idx])] = "something drops on the ground"
            continue
        conf_values, ids = confidence[idx].topk(1)
        if labels[ids] in real_world_sound_map:
            sound_events[tuple(sound_ranges[idx])] = real_world_sound_map[labels[ids]]
        else:
            sound_events[tuple(sound_ranges[idx])] = labels[ids]
        # print(conf_values, (format_time(sound_ranges[idx][0]), format_time(sound_ranges[idx][1])), LABELS[ids])

    logger.debug("sound events: %s", sound_events)
    return sound_events

if __name__ == '__main__':
    audio_path = 'real_world/data/makeCoffee3/videos/0/0/audio.wav'
    get_sound_events(audio_path, volume_thresh=0.03)
