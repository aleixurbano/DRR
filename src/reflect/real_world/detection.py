import os
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
from skimage.measure import find_contours
from argparse import ArgumentParser

from matplotlib.patches import Polygon
from hubconf import *  # vendored MDETR; on sys.path via reflect.cli.real_world_validation
import cv2
from reflect.real_world.logging_utils import get_logger


logger = get_logger(__name__)

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False)
_SEG_MODEL = None
_TOKENIZED_CAPTIONS = {}

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def get_seg_model():
    global _SEG_MODEL
    if _SEG_MODEL is None:
        _SEG_MODEL = mdetr_efficientnetB3_phrasecut(pretrained=True).to(device)
        _SEG_MODEL.eval()
    return _SEG_MODEL


def prepare_inference_image(im):
    return transform(im).unsqueeze(0).to(device)


def get_tokenized_caption(seg_model, caption):
    cache_key = (caption, str(device))
    cached = _TOKENIZED_CAPTIONS.get(cache_key)
    if cached is not None:
        return cached
    tokenized = seg_model.detr.transformer.tokenizer(
        [caption],
        padding="longest",
        return_tensors="pt",
    )
    _TOKENIZED_CAPTIONS[cache_key] = tokenized
    return tokenized

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def plot_results(pil_img, scores, boxes, labels, masks):
    np_image = np.array(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if masks is None:
      masks = [None for _ in range(len(scores))]
    # print("---", len(scores), len(boxes), len(labels), len(masks))
    assert len(scores) == len(boxes) == len(labels) == len(masks)
    for s, (xmin, ymin, xmax, ymax), l, mask, c in zip(scores, boxes.tolist(), labels, masks, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{l}: {s:0.2f}'
        ax.text(xmin, ymin, text, fontsize=8)

        if mask is None:
          continue
        np_image = apply_mask(np_image, mask, c)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
          # Subtract the padding and flip (y, x) to (x, y)
          verts = np.fliplr(verts) - 1
          p = Polygon(verts, facecolor="none", edgecolor=c)
          ax.add_patch(p)

    # plt.imshow(np_image)
    # plt.axis('off')
    # plt.show()
    im = Image.fromarray(np_image)
    return im

def add_res(results, ax, color='green'):
    #for tt in results.values():
    if True:
        bboxes = results['boxes']
        labels = results['labels']
        scores = results['scores']
        #keep = scores >= 0.0
        #bboxes = bboxes[keep].tolist()
        #labels = labels[keep].tolist()
        #scores = scores[keep].tolist()
    #print(torchvision.ops.box_iou(tt['boxes'].cpu().detach(), torch.as_tensor([[xmin, ymin, xmax, ymax]])))
    
    colors = ['purple', 'yellow', 'red', 'green', 'orange', 'pink']
    
    for i, (b, ll, ss) in enumerate(zip(bboxes, labels, scores)):
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color=colors[i], linewidth=3))
        cls_name = ll if isinstance(ll,str) else CLASSES[ll]
        text = f'{cls_name}: {ss:.2f}'
        logger.debug(text)
        ax.text(b[0], b[1], text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))


def plot_inference(im, caption, idx, model):
  # mean-std normalize the input image (batch-size: 1)
  img = prepare_inference_image(im)

  # propagate through the model
  memory_cache = model(img, [caption], encode_and_save=True)
  outputs = model(img, [caption], encode_and_save=False, memory_cache=memory_cache)

  # keep only predictions with 0.7+ confidence
  probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
  keep = (probas > 0.7).cpu()

  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)

  # Extract the text spans predicted by each box
  positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
  predicted_spans = defaultdict(str)
  for tok in positive_tokens:
    item, pos = tok
    if pos < 255:
        span = memory_cache["tokenized"].token_to_chars(0, pos)
        predicted_spans [item] += " " + caption[span.start:span.end]

  labels = [predicted_spans [k] for k in sorted(list(predicted_spans .keys()))]
  # plot_results(idx, im, probas[keep], bboxes_scaled, labels)
  
  return outputs, labels

def plot_inference_segmentation(
    im,
    caption,
    seg_model=None,
    detection_threshold=0.9,
    prepared_img=None,
    include_visualization=True,
):
  # mean-std normalize the input image (batch-size: 1)
  if seg_model is None:
    seg_model = get_seg_model()
  img = prepared_img if prepared_img is not None else prepare_inference_image(im)

  # print("caption: ",caption)
  # plt.imshow(img[0].permute(1,2,0).cpu())
  # plt.show()

  # propagate through the model
  outputs = seg_model(img, [caption])

  nan_keys = [key for key in ("pred_logits", "pred_boxes", "pred_masks") if torch.isnan(outputs[key]).any()]
  if nan_keys:
    raise RuntimeError(
        f"MDETR produced NaNs for caption '{caption}' on device '{device}'. "
        f"Affected outputs: {', '.join(nan_keys)}. "
        "This usually indicates a model/checkpoint or dependency compatibility issue."
    )

  # print("outputs: ", outputs.keys())

  # keep only predictions above the configured confidence threshold
  probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
  keep = (probas > detection_threshold).cpu()
  # print("probas: ", np.array(probas).shape)

  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)

  # Interpolate masks to the correct size
  w, h = im.size
  masks = F.interpolate(outputs["pred_masks"], size=(h, w), mode="bilinear", align_corners=False)
  masks = masks.cpu()[0, keep].sigmoid() > 0.5

  shrinked_masks = []
  if len(masks) != 0:
    for mask in masks:
      kernel = np.ones((3, 3), np.uint8)
      eroded_mask = cv2.erode(np.array(mask, dtype=np.float32), kernel, iterations=2)
      shrinked_masks.append(eroded_mask)
    shrinked_masks = np.array(shrinked_masks)
  else:
     shrinked_masks = masks
  
  tokenized = get_tokenized_caption(seg_model, caption)

  # Extract the text spans predicted by each box
  positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
  predicted_spans = defaultdict(str)
  for tok in positive_tokens:
    item, pos = tok
    if pos < 255:
        span = tokenized.token_to_chars(0, pos)
        predicted_spans [item] += " " + caption[span.start:span.end]

  labels = []
  for item_idx in range(len(bboxes_scaled)):
    label = predicted_spans.get(item_idx, "").strip()
    labels.append(label if label else caption)
  debug_im = None
  if include_visualization:
    debug_im = plot_results(im, probas[keep], bboxes_scaled, labels, masks)
  retval = {
    "probs": probas[keep],
    "labels": [caption]*len(masks),
    "bbox_2d": bboxes_scaled,
    "masks": shrinked_masks,
    "im": debug_im
  }
  # print("prob, label, bboxes_scaled, mask: ", probas[keep], labels, bboxes_scaled.shape, masks.shape)
  return retval

# def detect_object(im, prompt, idx):
#   masks_dict, bboxes_dict = {}, {}
#   _, bbox_labels = plot_inference(im, prompt, idx)
#   for single_obj_prompt in bbox_labels:
#     masks, bboxes = plot_inference_segmentation(im, single_obj_prompt, idx)
#     if len(masks) == 0:
#       continue
#     masks_dict[single_obj_prompt.strip()] = masks[0]
#     bboxes_dict[single_obj_prompt.strip()] = bboxes[0]
  
#   return masks_dict, bboxes_dict

def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser("Robot Failure Summarization")
    parser.add_argument('--folder_name', type=str, default="", help="if pipeline should be run on only one specific folder")
    return parser

if __name__ == "__main__":
  args = config_parser().parse_args()
  os.system(f"mkdir -p object_detection/mdetr/{args.folder_name}")

  task_objects = ["faucet", "mug", "sink"]

  # prompt = ','.join(task_objects)
  # total_frames = len(os.listdir(f"real_world/data/{args.folder_name}/rgb/"))
  total_frames = 1
  for idx in range(1, total_frames+1):
      plt.figure(figsize=(16,10))
      # im = Image.open(f"real_world/data/{args.folder_name}/rgb/{idx}.png").convert('RGB')
      im = Image.open(f"real_world/data/test/rgb/3.png").convert('RGB') 
      # _, labels = plot_inference(im, prompt, idx)
      for single_obj_prompt in task_objects:
        retval = plot_inference_segmentation(im, single_obj_prompt, get_seg_model())
      
      # cv2.imwrite(f"object_detection/mdetr/{args.folder_name}/img_step_{idx}.png", np.array(im))
      logger.debug("type(retval['im']): %s", type(retval['im']))
      plt.imshow(retval['im'])
      # plt.savefig(f"object_detection/mdetr/{args.folder_name}/img_step_{idx}.png")
      plt.savefig(f"real_world/state_summary/temp/mdetr/{idx}.png")
      plt.close()
