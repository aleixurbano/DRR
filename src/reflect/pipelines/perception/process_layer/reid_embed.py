"""Masked-crop CLIP embeddings for cross-occlusion track re-identification.

We only re-encode every ~30 frames per track (controlled by `TrackStore`),
so this can be a "real" forward pass without dominating the per-frame budget.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

import open_clip


# ImageNet mean (in 0-255 space) - used to blank out the background of the
# masked crop so the CLIP encoder sees a clean foreground.
_IMAGENET_MEAN_U8 = np.array([124, 117, 104], dtype=np.uint8)


class MaskedCLIPEncoder:
    """Encodes the tight bbox of a mask with the background set to grey."""

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess

    @torch.inference_mode()
    def encode(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        """Return a (D,) L2-normalized embedding, or None if the mask is empty."""
        if not mask.any():
            return None
        crop = _masked_crop(rgb, mask)
        if crop is None:
            return None
        tensor = self._preprocess(Image.fromarray(crop)).unsqueeze(0).to(self.device)
        feat = self._model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)


def _masked_crop(rgb: np.ndarray, mask: np.ndarray, pad: int = 4) -> np.ndarray | None:
    """Tight bbox crop with background blanked to ImageNet mean grey."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0 = max(0, ys.min() - pad)
    y1 = min(rgb.shape[0], ys.max() + 1 + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(rgb.shape[1], xs.max() + 1 + pad)
    crop = rgb[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    # Background pixels go to ImageNet mean grey so the encoder has no obvious
    # context leak. Foreground pixels remain untouched.
    crop[~crop_mask] = _IMAGENET_MEAN_U8
    return crop
