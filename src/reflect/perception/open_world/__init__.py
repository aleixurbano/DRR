"""Open-world perception module for REFLECT.

Pipeline: SAM2/SAM3 → OpenCLIP open-vocabulary naming → Depth Anything v2
(monocular fallback) → spatial relations via reused SceneGraph heuristics.

Backend selection (env var):
    REFLECT_SAM_BACKEND=sam2  (default; publicly available)
    REFLECT_SAM_BACKEND=sam3  (requires HF access to facebook/sam3)

Auth: run `huggingface-cli login` before first use if you want SAM3 weights.
"""

from .builder import perceive_image, perceive_rgbd_sequence, Frame

__all__ = ["perceive_image", "perceive_rgbd_sequence", "Frame"]
