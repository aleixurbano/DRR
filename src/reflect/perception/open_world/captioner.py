"""Per-object VLM captioning for the ConceptGraphs-style scene-graph upgrade.

Replaces YOLOe class labels with free-form VLM descriptions. Backends:
  - HF transformers: Qwen2.5-VL-7B-Instruct, InternVL3-8B
  - Ollama (already in stack): qwen3.5:9b, gemma4:26b

Backend selection is per-call (`backend=...`) or via env var `REFLECT_CAPTIONER`.

Inputs are an RGB image plus a single-object mask + bbox; the function returns
a short noun-phrase description (3-7 words) suitable as an `object_tag`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import List, Literal, Optional, Sequence

import numpy as np
from PIL import Image

# Ollama vision tags that we know are pulled locally.
_OLLAMA_VLMS = ("qwen3.5:9b", "qwen3.5:27b", "gemma4:26b", "gemma4:31b")
# HF VLM backends.
_HF_VLM_IDS = {
    "qwen2_5_vl_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2_5_vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "internvl3_8b":  "OpenGVLab/InternVL3-8B-hf",
}

DEFAULT_BACKEND = os.environ.get("REFLECT_CAPTIONER", "qwen2_5_vl_7b")

# Prompt (mirrors ConceptGraphs' query but constrained to short noun phrases so
# the downstream LLM relation-labelling step doesn't get long-form prose).
CAPTION_PROMPT = (
    "This is a crop centered on one object, with extra context around it. "
    "Some versions of the crop may have non-object pixels darkened or outlined. "
    "Ignore the background treatment and describe only the central object in 3 to 7 words. "
    "Output only the description, no full sentences or extra punctuation."
)

# How to render the mask onto the crop before passing to the VLM (mirrors
# ConceptGraphs' `masking_option`).
MaskMode = Literal["none", "blackout", "red_outline"]


@dataclass
class CaptionResult:
    text: str
    backend: str
    latency_s: float


def _crop_with_mask(rgb: np.ndarray, mask: np.ndarray, bbox_xyxy: np.ndarray,
                    *, padding: int = 12, mask_mode: MaskMode = "blackout") -> Image.Image:
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy.astype(int)
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(W, x2 + padding), min(H, y2 + padding)
    crop = rgb[y1:y2, x1:x2].copy()
    sub_mask = mask[y1:y2, x1:x2]

    if mask_mode == "blackout":
        # Black out non-mask area; keeps the VLM focused on the object.
        crop[~sub_mask] = (crop[~sub_mask] * 0.15).astype(crop.dtype)
    elif mask_mode == "red_outline":
        import cv2
        contours, _ = cv2.findContours(sub_mask.astype(np.uint8) * 255,
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, contours, -1, (255, 0, 0), 3)
    return Image.fromarray(crop)


def _pil_to_b64(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- HF transformers backends ---------------------------------------------------

@lru_cache(maxsize=2)
def _load_hf_qwen_vl(model_id: str, device: str):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    return proc, model


def _caption_qwen_hf(crop: Image.Image, model_id: str) -> str:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc, model = _load_hf_qwen_vl(model_id, device)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop},
        {"type": "text",  "text":  CAPTION_PROMPT},
    ]}]
    text_in = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(
        text=[text_in], images=[crop], padding=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    trimmed = out_ids[:, inputs.input_ids.shape[1]:]
    text = proc.batch_decode(trimmed, skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)[0].strip()
    return _normalize_caption(text)


@lru_cache(maxsize=1)
def _load_hf_internvl(model_id: str, device: str):
    import torch
    from transformers import AutoProcessor, InternVLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(model_id)
    model = InternVLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    return proc, model


def _caption_internvl_hf(crop: Image.Image, model_id: str) -> str:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc, model = _load_hf_internvl(model_id, device)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop},
        {"type": "text",  "text":  CAPTION_PROMPT},
    ]}]
    inputs = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    trimmed = out_ids[:, inputs.input_ids.shape[1]:]
    text = proc.batch_decode(trimmed, skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)[0].strip()
    return _normalize_caption(text)


# --- Ollama backends ------------------------------------------------------------

@lru_cache(maxsize=1)
def _ollama_client():
    from ollama import Client
    return Client()


def _looks_like_think_only(text: str) -> bool:
    import re

    t = (text or "").strip().lower()
    if not t:
        return True
    t = re.sub(r"</?think>", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" :\n\t")
    return t in {"", "think", "thinking"}


def _looks_like_caption_control_text(text: str) -> bool:
    t = (text or "").strip().lower()
    if _looks_like_think_only(t):
        return True
    return t.startswith((
        "got it",
        "let's look",
        "lets look",
        "i'll look",
        "ill look",
        "i will look",
        "i'll analyze",
        "ill analyze",
        "i will analyze",
        "sure,",
        "sure.",
        "okay,",
        "okay.",
        "here's",
        "here is",
    ))


def _extract_caption_sentence(text: str) -> str:
    import re

    t = re.sub(r"(?is)^\s*<think>.*?</think>\s*", "", (text or "")).strip()
    t = re.sub(r"(?is)^\s*</?think>\s*", "", t).strip()
    if not t:
        return ""

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    cues = (
        "the central object is",
        "the object is",
        "the image shows",
        "this image shows",
        "the central object in the image is",
        "the object in the image is",
        "the subject is",
        "it is",
    )

    for sentence in sentences:
        lower = sentence.lower()
        if any(cue in lower for cue in cues):
            return sentence

    return sentences[0] if sentences else t


def _caption_ollama(crop: Image.Image, model_tag: str) -> str:
    """Reasoning models like qwen3.5 / gemma4 spend their token budget in
    `thinking` mode by default; we explicitly disable thinking so the whole
    output budget goes to the visible answer.
    """
    client = _ollama_client()
    img_bytes = _pil_to_b64(crop)
    prompts = [CAPTION_PROMPT]
    if model_tag.startswith("qwen3-vl"):
        prompts.append(
            "Return exactly one short noun phrase naming the central object. "
            "Do not acknowledge the request. Do not mention the image. "
            "Do not output <think> tags or reasoning.\n\n"
            + CAPTION_PROMPT
        )

    text = ""
    for prompt in prompts:
        resp = client.chat(
            model=model_tag,
            messages=[{"role": "user", "content": prompt, "images": [img_bytes]}],
            think=False,
            options={"temperature": 0.0, "num_predict": 32},
        )
        msg = resp["message"] if isinstance(resp, dict) else resp.message
        text = (msg.get("content") if isinstance(msg, dict) else msg.content) or ""
        if not text.strip():
            # Some Ollama builds ignore think=False; fall back to thinking content.
            thinking = msg.get("thinking") if isinstance(msg, dict) else getattr(msg, "thinking", None)
            text = thinking or ""
        if not _looks_like_caption_control_text(text):
            break
    return _normalize_caption(text)


# --- Normalization --------------------------------------------------------------

def _normalize_caption(text: str) -> str:
    """Strip quoting / common prefixes / trailing punctuation so the caption is
    a clean noun phrase usable as an `object_tag`."""
    import re
    t = _extract_caption_sentence(text)
    t = re.sub(r"(?is)^\s*<think>.*?</think>\s*", "", t)
    t = re.sub(r"(?is)^\s*</?think>\s*", "", t)
    if _looks_like_caption_control_text(t):
        return "unknown"
    # Strip common preambles produced by various VLMs.
    for pre in [
        "the object is ", "the central object is ", "the image shows ",
        "this image shows ", "the central object in the image is ",
        "in the image, ", "the object in the image is ",
    ]:
        if t.lower().startswith(pre):
            t = t[len(pre):]
            break
    # Drop quotes, articles, trailing punctuation.
    t = t.strip(' "\'`.')
    t = re.sub(r"^(a|an|the)\s+", "", t, flags=re.I)
    # First sentence only.
    t = t.split(".")[0].split("\n")[0].strip()
    if _looks_like_caption_control_text(t):
        return "unknown"
    # Cap to 8 words for downstream-prompt token discipline.
    parts = t.split()
    if len(parts) > 8:
        t = " ".join(parts[:8])
    return t.lower() if t else "unknown"


# --- Public API -----------------------------------------------------------------

def caption(
    rgb: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: np.ndarray,
    *,
    backend: Optional[str] = None,
    mask_mode: MaskMode = "blackout",
) -> CaptionResult:
    """Caption a single masked object. `backend` defaults to REFLECT_CAPTIONER."""
    import time
    backend = (backend or DEFAULT_BACKEND).strip()
    crop = _crop_with_mask(rgb, mask, bbox_xyxy, mask_mode=mask_mode)

    t0 = time.time()
    if backend in _HF_VLM_IDS:
        model_id = _HF_VLM_IDS[backend]
        if backend.startswith("internvl"):
            text = _caption_internvl_hf(crop, model_id)
        else:
            text = _caption_qwen_hf(crop, model_id)
    elif backend in _OLLAMA_VLMS or backend.startswith(("qwen3", "gemma", "llava", "mistral", "phi")):
        text = _caption_ollama(crop, backend)
    else:
        raise ValueError(f"Unknown captioner backend: {backend!r}. "
                          f"Valid: {list(_HF_VLM_IDS)} or {list(_OLLAMA_VLMS)}.")
    return CaptionResult(text=text, backend=backend, latency_s=time.time() - t0)


def caption_batch(
    rgb: np.ndarray,
    detections: Sequence,            # list of detector.Detection
    *,
    backend: Optional[str] = None,
    mask_mode: MaskMode = "blackout",
) -> List[CaptionResult]:
    """Caption every detection in a frame. Sequential (no batched VLM inference yet)."""
    return [caption(rgb, d.mask, d.bbox_xyxy, backend=backend, mask_mode=mask_mode)
             for d in detections]


def list_backends() -> dict[str, list[str]]:
    return {"hf": list(_HF_VLM_IDS), "ollama": list(_OLLAMA_VLMS)}
