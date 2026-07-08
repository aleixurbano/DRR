"""
    Tier 0: Startup
    - First agent lookup, understand the environment
"""
from pydantic import BaseModel
from typing import Sequence
import numpy as np
from ollama import Client

SYSTEM_PROMPT = (
    "You are a visual perception module for a robot arm. "
    "Your output feeds directly into an open-set object detector (YOLOE), "
    "so every entry must be a self-contained, detectable object, not a sub-component. "
    "For example: output 'oven', never 'oven door' or 'oven handle'. "
    "Output 'pot', never 'pot handle'. Output 'faucet', never 'faucet knob'. "
    "One unique name per object. No explanations, no duplicates."
)

USER_PROMPT = (
    "List every distinct, whole object visible in this robotic workspace image/s. "
    "Include objects a robot could grasp, place, or interact with as a unit, "
    "as well as large functional surfaces and appliances that serve as landmarks or placement targets. "
    "You may include color or material in the name when it helps distinguish similar items (e.g. 'blue bowl', 'metal pot'). "
    "Do NOT include: parts of objects (handles, knobs, doors, burners, lids), "
    "structural background elements (walls, backsplash, ceiling), "
    "or abstract regions (countertop edge, table surface). "
    "Each entry must refer to exactly one whole, independently existing object."
)

class StringList(BaseModel):
    objects: list[str]

def describe_environment(
    rgb: np.ndarray | list[np.ndarray],
    extras: Sequence[str] = (),
    client: Client | None = None
) -> list[str]:
    """Use a vision-language model to generate a high-level description of the scene.

    Returns a *stable-ordered* list of unique, cleaned object names. The
    Dirichlet index axis depends on this order, so the function must NOT
    return a set - Python set ordering is hash-randomised across processes
    and a reload would silently misalign every `alpha` vector.
    """
    
    # Reuse the client if provided, otherwise create a new one
    ollama_client = client or Client()

    # Convert image arrays to a format suitable for the API (e.g. JPEG bytes)
    rgb_list = rgb if isinstance(rgb, list) else [rgb]
    images_converted = [convert_rgb_array_to_image(img) for img in rgb_list]

    # Api call to the vision-language model with a structured output format
    response = ollama_client.chat(
        model="qwen3.5:9b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT, "images": images_converted}
        ],
        format=StringList.model_json_schema(),
        options={
            'temperature': 0.1, 
            'num_predict': -1
        },
        think=True
    )

    # Parse the structured response into a list of objects
    object_list = StringList.model_validate_json(response.message.content)

    # Clean and normalize, deduplicating while preserving first-seen order.
    seen: set[str] = set()
    cleaned: list[str] = []
    for obj in list(object_list.objects) + list(extras):
        name = obj.strip().lower()
        if name and name not in seen:
            seen.add(name)
            cleaned.append(name)
    return cleaned

def convert_rgb_array_to_image(rgb_array):
    """Convert a NumPy RGB array to a PIL Image."""
    from PIL import Image
    import io

    pil_image = Image.fromarray(rgb_array)
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()