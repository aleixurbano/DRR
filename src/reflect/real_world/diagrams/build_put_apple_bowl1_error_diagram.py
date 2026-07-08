from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


FRAME_ID = 1276
TASK_NAME = "putAppleBowl1"

CANVAS_W = 2400
CANVAS_H = 1700
MARGIN = 60

BG = (248, 246, 241)
PANEL = (255, 255, 255)
PANEL_BORDER = (218, 214, 206)
TITLE = (34, 34, 34)
MUTED = (92, 92, 92)
RED = (191, 55, 55)
BLUE = (48, 107, 177)
GOLD = (187, 126, 34)
GREEN = (64, 132, 90)
GRAY = (110, 110, 110)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            ]
        )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = load_font(46, bold=True)
FONT_SUBTITLE = load_font(24)
FONT_PANEL = load_font(28, bold=True)
FONT_BODY = load_font(22)
FONT_BODY_BOLD = load_font(22, bold=True)
FONT_SMALL = load_font(18)
FONT_TINY = load_font(16)


def crop_nonwhite(img: Image.Image, tolerance: int = 250) -> Image.Image:
    rgb = img.convert("RGB")
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg)
    bbox = diff.point(lambda p: 255 if p > (255 - tolerance) else 0).getbbox()
    return rgb.crop(bbox) if bbox else rgb


def fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    resample = getattr(Image, "Resampling", Image).LANCZOS
    out = img.copy()
    out.thumbnail((max_w, max_h), resample=resample)
    return out


def draw_round_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=24, fill=PANEL, outline=PANEL_BORDER, width=3)


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_spacing: int = 6,
) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + line_spacing
    return y


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
    draw.line([start, end], fill=color, width=6)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_len = 18
    arrow_angle = math.pi / 7
    p1 = (
        end[0] - arrow_len * math.cos(angle - arrow_angle),
        end[1] - arrow_len * math.sin(angle - arrow_angle),
    )
    p2 = (
        end[0] - arrow_len * math.cos(angle + arrow_angle),
        end[1] - arrow_len * math.sin(angle + arrow_angle),
    )
    draw.polygon([end, p1, p2], fill=color)


def add_label(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
    bbox = draw.textbbox((0, 0), text, font=FONT_SMALL)
    pad_x, pad_y = 10, 6
    box = (
        xy[0],
        xy[1],
        xy[0] + (bbox[2] - bbox[0]) + pad_x * 2,
        xy[1] + (bbox[3] - bbox[1]) + pad_y * 2,
    )
    draw.rounded_rectangle(box, radius=12, fill=(255, 255, 255), outline=color, width=3)
    draw.text((xy[0] + pad_x, xy[1] + pad_y - 1), text, font=FONT_SMALL, fill=color)


def build_stats(det_payload: dict) -> tuple[list[float], list[float]]:
    h, w = det_payload["pred_masks"].shape[1:]
    img_area = float(w * h)
    apple_areas: list[float] = []
    bowl_areas: list[float] = []
    for label, box in zip(det_payload["labels"], det_payload["bbox_2d"]):
        x1, y1, x2, y2 = map(float, box)
        area_pct = max(0.0, (x2 - x1) * (y2 - y1) / img_area * 100.0)
        if label == "red apple":
            apple_areas.append(area_pct)
        elif "bowl" in label:
            bowl_areas.append(area_pct)
    return apple_areas, bowl_areas


def extract_llm_info(llm_trace: dict) -> tuple[str, str]:
    verifier_text = ""
    explanation_text = ""
    for item in llm_trace.values():
        system = item["prompt"].get("system", "")
        response = item.get("response", "")
        user = item["prompt"].get("user", "")
        if "success verifier" in system:
            verifier_text = (
                "Goal: pick up apple\n"
                "Observation: dark blue bowl. dark blue bowl is inside robot gripper.\n"
                f"LLM answer: {response}"
            )
        elif "provide explanation for a robot failure" in system and "Visual observation:" in user:
            explanation_text = response
    return verifier_text, explanation_text


def shorten(values: list[float]) -> str:
    return ", ".join(f"{v:.1f}%" for v in values[:4])


def main() -> None:
    code_dir = Path(__file__).resolve().parent.parent
    diagram_dir = Path(__file__).resolve().parent
    base = code_dir / "real_world" / "state_summary" / TASK_NAME
    mdetr_dir = base / "mdetr_obj_det"

    original = crop_nonwhite(Image.open(mdetr_dir / "images" / f"{FRAME_ID}.png"))
    raw_det_img = crop_nonwhite(Image.open(mdetr_dir / "det" / f"{FRAME_ID}.png"))
    filtered_img = crop_nonwhite(Image.open(mdetr_dir / "clip_processed_det" / f"{FRAME_ID}.png"))

    with open(mdetr_dir / "det" / f"{FRAME_ID}.pickle", "rb") as f:
        raw_det = pickle.load(f)
    with open(mdetr_dir / "clip_processed_det" / f"{FRAME_ID}.pickle", "rb") as f:
        filtered_det = pickle.load(f)
    with open(base / "llm_trace.json", "r") as f:
        llm_trace = json.load(f)

    with open(base / "state_summary_L1.txt", "r") as f:
        l1_lines = [line.strip() for line in f.readlines() if line.strip()]
    with open(base / "state_summary_L2.txt", "r") as f:
        l2_lines = [line.strip() for line in f.readlines() if line.strip()]

    apple_areas, bowl_areas = build_stats(raw_det)
    verifier_text, explanation_text = extract_llm_info(llm_trace)

    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(canvas)

    draw.text((MARGIN, 40), "How The Real Pipeline Fails On putAppleBowl1 At 00:42", font=FONT_TITLE, fill=TITLE)
    subtitle = (
        "One diagram, two goals: show the pipeline stages and show the exact place where the "
        "apple-vs-bowl error enters the system."
    )
    draw_wrapped_text(draw, subtitle, (MARGIN, 100), FONT_SUBTITLE, MUTED, CANVAS_W - MARGIN * 2)

    top_y = 160
    top_h = 610
    top_w = 700
    top_gap = 40
    top_xs = [60, 60 + top_w + top_gap, 60 + (top_w + top_gap) * 2]

    bottom_y = 840
    bottom_w = 520
    bottom_h = 760
    bottom_gap = 30
    bottom_xs = [60, 60 + bottom_w + bottom_gap, 60 + (bottom_w + bottom_gap) * 2, 60 + (bottom_w + bottom_gap) * 3]

    top_boxes = [(x, top_y, x + top_w, top_y + top_h) for x in top_xs]
    bottom_boxes = [(x, bottom_y, x + bottom_w, bottom_y + bottom_h) for x in bottom_xs]

    for box in top_boxes + bottom_boxes:
        draw_round_panel(draw, box)

    # Panel 1: original frame
    box = top_boxes[0]
    draw.text((box[0] + 26, box[1] + 20), "1. What A Human Sees", font=FONT_PANEL, fill=TITLE)
    fitted = fit_image(original, top_w - 52, 360)
    img_x = box[0] + (top_w - fitted.width) // 2
    img_y = box[1] + 75
    canvas.paste(fitted, (img_x, img_y))
    img_draw = ImageDraw.Draw(canvas)
    # Hand-picked callouts over the cropped frame.
    apple_rect = (img_x + 220, img_y + 190, img_x + 294, img_y + 263)
    bowl_rect = (img_x + 43, img_y + 190, img_x + 130, img_y + 265)
    img_draw.ellipse(apple_rect, outline=RED, width=6)
    img_draw.ellipse(bowl_rect, outline=BLUE, width=6)
    add_label(img_draw, "Visible apple", (apple_rect[0] + 38, apple_rect[1] - 36), RED)
    add_label(img_draw, "Bowl on counter", (bowl_rect[0] - 6, bowl_rect[1] - 36), BLUE)
    human_text = (
        "At this moment the apple is visibly near the gripper and the bowl is still on the counter. "
        "This is the ground truth a person would describe from the image."
    )
    draw_wrapped_text(draw, human_text, (box[0] + 26, box[1] + 470), FONT_BODY, MUTED, top_w - 52)

    # Panel 2: raw detector output
    box = top_boxes[1]
    draw.text((box[0] + 26, box[1] + 20), "2. Raw MDETR Output", font=FONT_PANEL, fill=TITLE)
    fitted = fit_image(raw_det_img, top_w - 52, 360)
    img_x = box[0] + (top_w - fitted.width) // 2
    img_y = box[1] + 75
    canvas.paste(fitted, (img_x, img_y))
    raw_text = (
        f"11 raw detections total: {sum(label == 'red apple' for label in raw_det['labels'])} apple, "
        f"{sum('bowl' in label for label in raw_det['labels'])} bowl.\n"
        f"Apple box areas: {shorten(apple_areas)} of the full image.\n"
        f"Bowl box areas: {shorten(bowl_areas)} of the full image."
    )
    draw_wrapped_text(draw, raw_text, (box[0] + 26, box[1] + 460), FONT_BODY, MUTED, top_w - 52)
    warn_text = "Problem: several apple boxes are huge scene-level regions, not tight boxes around the apple."
    draw_wrapped_text(draw, warn_text, (box[0] + 26, box[1] + 555), FONT_SMALL, RED, top_w - 52)

    # Panel 3: filtered detector output
    box = top_boxes[2]
    draw.text((box[0] + 26, box[1] + 20), "3. After CLIP Filtering", font=FONT_PANEL, fill=TITLE)
    fitted = fit_image(filtered_img, top_w - 52, 360)
    img_x = box[0] + (top_w - fitted.width) // 2
    img_y = box[1] + 75
    canvas.paste(fitted, (img_x, img_y))
    filtered_text = (
        f"Kept detections: {filtered_det['total_detections']}.\n"
        f"Surviving label: {', '.join(filtered_det['labels'])}.\n"
        "All apple candidates were dropped during CLIP confirmation and duplicate cleanup."
    )
    draw_wrapped_text(draw, filtered_text, (box[0] + 26, box[1] + 460), FONT_BODY, MUTED, top_w - 52)
    draw_wrapped_text(
        draw,
        "This is the first irreversible mistake. The apple is already gone from the pipeline state.",
        (box[0] + 26, box[1] + 555),
        FONT_SMALL,
        RED,
        top_w - 52,
    )

    # Arrows top row
    draw_arrow(draw, (top_boxes[0][2] + 10, top_y + 250), (top_boxes[1][0] - 10, top_y + 250), GRAY)
    draw_arrow(draw, (top_boxes[1][2] + 10, top_y + 250), (top_boxes[2][0] - 10, top_y + 250), GRAY)
    add_label(draw, "Object detection", (top_boxes[0][2] - 120, top_y + 200), GOLD)
    add_label(draw, "CLIP confirmation", (top_boxes[1][2] - 150, top_y + 200), GOLD)

    # Bottom panel 1: scene graph
    box = bottom_boxes[0]
    draw.text((box[0] + 26, box[1] + 20), "4. Scene Graph Output", font=FONT_PANEL, fill=TITLE)
    graph_note = (
        "The scene graph is built only from the surviving detections. Since the apple was removed, "
        "the graph contains only the bowl."
    )
    draw_wrapped_text(draw, graph_note, (box[0] + 26, box[1] + 72), FONT_BODY, MUTED, bottom_w - 52)
    bowl_box = (box[0] + 95, box[1] + 250, box[0] + 425, box[1] + 325)
    grip_box = (box[0] + 135, box[1] + 485, box[0] + 390, box[1] + 560)
    draw.rounded_rectangle(bowl_box, radius=18, fill=(235, 243, 252), outline=BLUE, width=4)
    draw.rounded_rectangle(grip_box, radius=18, fill=(252, 239, 239), outline=RED, width=4)
    bowl_text = "Node: dark blue bowl"
    grip_text = "Node: robot gripper"
    bowl_bbox = draw.textbbox((0, 0), bowl_text, font=FONT_BODY_BOLD)
    grip_bbox = draw.textbbox((0, 0), grip_text, font=FONT_BODY_BOLD)
    draw.text(
        (bowl_box[0] + (bowl_box[2] - bowl_box[0] - (bowl_bbox[2] - bowl_bbox[0])) // 2, bowl_box[1] + 20),
        bowl_text,
        font=FONT_BODY_BOLD,
        fill=BLUE,
    )
    draw.text(
        (grip_box[0] + (grip_box[2] - grip_box[0] - (grip_bbox[2] - grip_bbox[0])) // 2, grip_box[1] + 20),
        grip_text,
        font=FONT_BODY_BOLD,
        fill=RED,
    )
    draw_arrow(draw, ((bowl_box[0] + bowl_box[2]) // 2, bowl_box[3] + 5), ((grip_box[0] + grip_box[2]) // 2, grip_box[1] - 12), GRAY)
    add_label(draw, "inside", (box[0] + 205, box[1] + 386), GRAY)
    scene_text = (
        "Why the wrong edge appears: add_agent() sees the gripper as engaged and the bowl center is "
        "close enough to the camera, so it tags the bowl as held."
    )
    draw_wrapped_text(draw, scene_text, (box[0] + 26, box[1] + 610), FONT_SMALL, MUTED, bottom_w - 52)

    # Bottom panel 2: summaries
    box = bottom_boxes[1]
    draw.text((box[0] + 26, box[1] + 20), "5. English Summary Stage", font=FONT_PANEL, fill=TITLE)
    draw_wrapped_text(
        draw,
        "The pipeline turns the scene graph into plain English before the LLM sees anything.",
        (box[0] + 26, box[1] + 72),
        FONT_BODY,
        MUTED,
        bottom_w - 52,
    )
    l1_box = (box[0] + 26, box[1] + 180, box[2] - 26, box[1] + 360)
    l2_box = (box[0] + 26, box[1] + 410, box[2] - 26, box[1] + 590)
    draw.rounded_rectangle(l1_box, radius=18, fill=(249, 246, 238), outline=(227, 219, 195), width=3)
    draw.rounded_rectangle(l2_box, radius=18, fill=(240, 247, 241), outline=(199, 220, 200), width=3)
    draw.text((l1_box[0] + 16, l1_box[1] + 14), "L1 summary line", font=FONT_BODY_BOLD, fill=GOLD)
    draw_wrapped_text(draw, l1_lines[0], (l1_box[0] + 16, l1_box[1] + 55), FONT_BODY, TITLE, l1_box[2] - l1_box[0] - 32)
    draw.text((l2_box[0] + 16, l2_box[1] + 14), "L2 summary line", font=FONT_BODY_BOLD, fill=GREEN)
    draw_wrapped_text(draw, l2_lines[0], (l2_box[0] + 16, l2_box[1] + 55), FONT_BODY, TITLE, l2_box[2] - l2_box[0] - 32)
    draw_wrapped_text(
        draw,
        "Notice the wording: the text already says the bowl is inside the gripper. That incorrect sentence becomes the LLM's evidence.",
        (box[0] + 26, box[1] + 640),
        FONT_SMALL,
        MUTED,
        bottom_w - 52,
    )

    # Bottom panel 3: LLM view
    box = bottom_boxes[2]
    draw.text((box[0] + 26, box[1] + 20), "6. What The LLM Receives", font=FONT_PANEL, fill=TITLE)
    llm_box_1 = (box[0] + 26, box[1] + 140, box[2] - 26, box[1] + 360)
    llm_box_2 = (box[0] + 26, box[1] + 410, box[2] - 26, box[1] + 660)
    draw.rounded_rectangle(llm_box_1, radius=18, fill=(240, 244, 250), outline=(197, 210, 228), width=3)
    draw.rounded_rectangle(llm_box_2, radius=18, fill=(250, 242, 242), outline=(228, 199, 199), width=3)
    draw.text((llm_box_1[0] + 16, llm_box_1[1] + 14), "Subgoal verifier prompt", font=FONT_BODY_BOLD, fill=BLUE)
    verifier_lines = verifier_text.splitlines()
    prompt_y = llm_box_1[1] + 55
    for line in verifier_lines:
        prompt_y = draw_wrapped_text(
            draw,
            line,
            (llm_box_1[0] + 16, prompt_y),
            FONT_SMALL,
            TITLE,
            llm_box_1[2] - llm_box_1[0] - 32,
            line_spacing=5,
        )
        prompt_y += 6
    draw.text((llm_box_2[0] + 16, llm_box_2[1] + 14), "Failure explanation produced by the LLM", font=FONT_BODY_BOLD, fill=RED)
    draw_wrapped_text(
        draw,
        explanation_text,
        (llm_box_2[0] + 16, llm_box_2[1] + 55),
        FONT_SMALL,
        TITLE,
        llm_box_2[2] - llm_box_2[0] - 32,
    )
    draw_wrapped_text(
        draw,
        "The LLM is not looking back at the image. It only reasons over the wrong text summary it was given.",
        (box[0] + 26, box[1] + 700),
        FONT_SMALL,
        MUTED,
        bottom_w - 52,
    )

    # Bottom panel 4: root cause
    box = bottom_boxes[3]
    draw.text((box[0] + 26, box[1] + 20), "7. Root Cause Chain", font=FONT_PANEL, fill=TITLE)
    chain_items = [
        "1. Phrase grounding creates oversized apple boxes, some covering almost the whole image.",
        "2. CLIP evaluates those huge crops, which contain much more than the apple.",
        "3. Apple candidates are rejected; the bowl is the only surviving object.",
        "4. The gripper heuristic assigns the remaining nearby bowl to the robot gripper.",
        "5. Every later stage stays logically consistent, but consistent with the wrong state.",
    ]
    y = box[1] + 88
    for item in chain_items:
        bullet = (box[0] + 32, y + 8, box[0] + 42, y + 18)
        draw.ellipse(bullet, fill=RED)
        y = draw_wrapped_text(draw, item, (box[0] + 58, y), FONT_BODY, TITLE, bottom_w - 84)
        y += 12

    takeaway_box = (box[0] + 26, box[1] + 560, box[2] - 26, box[1] + 710)
    draw.rounded_rectangle(takeaway_box, radius=20, fill=(252, 246, 232), outline=(223, 188, 108), width=4)
    draw.text((takeaway_box[0] + 16, takeaway_box[1] + 14), "Takeaway", font=FONT_BODY_BOLD, fill=GOLD)
    takeaway = (
        "This is a good teaching example because the pipeline itself is working as designed. "
        "The error enters early, at object localization and post-filtering, then propagates cleanly."
    )
    draw_wrapped_text(draw, takeaway, (takeaway_box[0] + 16, takeaway_box[1] + 52), FONT_BODY, TITLE, takeaway_box[2] - takeaway_box[0] - 32)

    # Arrows between bottom panels
    draw_arrow(draw, (bottom_boxes[0][2] + 8, bottom_y + 360), (bottom_boxes[1][0] - 8, bottom_y + 360), GRAY)
    draw_arrow(draw, (bottom_boxes[1][2] + 8, bottom_y + 360), (bottom_boxes[2][0] - 8, bottom_y + 360), GRAY)
    draw_arrow(draw, (bottom_boxes[2][2] + 8, bottom_y + 360), (bottom_boxes[3][0] - 8, bottom_y + 360), GRAY)
    draw_arrow(draw, ((top_boxes[2][0] + top_boxes[2][2]) // 2, top_boxes[2][3] + 12), ((bottom_boxes[0][0] + bottom_boxes[0][2]) // 2, bottom_boxes[0][1] - 14), GRAY)

    footer = (
        f"Generated from saved artifacts in real_world/state_summary/{TASK_NAME} for frame {FRAME_ID}. "
        "The diagram is reproducible via build_put_apple_bowl1_error_diagram.py."
    )
    draw.text((MARGIN, CANVAS_H - 36), footer, font=FONT_TINY, fill=MUTED)

    output_path = diagram_dir / "putAppleBowl1_error_pipeline.png"
    canvas.save(output_path, quality=95)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
