import clip
import numpy as np
from PIL import Image
import torch

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False)

clip_version = "ViT-B/16"
clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768, 'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
model = None
preprocess = None
_TEXT_FEATS_CACHE = {}


def get_clip_model_and_preprocess():
  global model, preprocess
  if model is None or preprocess is None:
    model, preprocess = clip.load(clip_version, device=device)  # clip.available_models()
    model.eval()
  return model, preprocess


def get_clip_model():
  return get_clip_model_and_preprocess()[0]

def get_text_feats(in_text, batch_size=64):
  cache_key = tuple(in_text)
  cached = _TEXT_FEATS_CACHE.get(cache_key)
  if cached is not None:
    return cached
  clip_model, _ = get_clip_model_and_preprocess()
  text_tokens = clip.tokenize(in_text).to(device)
  text_id = 0
  text_feats = np.zeros((len(in_text), clip_feat_dim), dtype=np.float32)
  while text_id < len(text_tokens):  # Batched inference.
    batch_size = min(len(in_text) - text_id, batch_size)
    text_batch = text_tokens[text_id:text_id+batch_size]
    with torch.no_grad():
      batch_feats = clip_model.encode_text(text_batch).float()
    batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
    batch_feats = np.float32(batch_feats.cpu())
    text_feats[text_id:text_id+batch_size, :] = batch_feats
    text_id += batch_size
  _TEXT_FEATS_CACHE[cache_key] = text_feats
  return text_feats

def get_img_feats(img):
  return get_img_feats_batch([img])[0:1]


def get_img_feats_batch(images):
  if len(images) == 0:
    return np.zeros((0, clip_feat_dim), dtype=np.float32)

  clip_model, clip_preprocess = get_clip_model_and_preprocess()
  processed_images = []
  for img in images:
    img_pil = Image.fromarray(np.uint8(img))
    processed_images.append(clip_preprocess(img_pil))
  img_in = torch.stack(processed_images, dim=0).to(device)
  with torch.no_grad():
    img_feats = clip_model.encode_image(img_in).float()
  img_feats /= img_feats.norm(dim=-1, keepdim=True)
  img_feats = np.float32(img_feats.cpu())
  return img_feats

def get_nn_text(raw_texts, text_feats, img_feats):
  scores = text_feats @ img_feats.T
  scores = scores.squeeze()
  high_to_low_ids = np.argsort(scores).squeeze()[::-1]
  high_to_low_texts = [raw_texts[i] for i in high_to_low_ids]
  high_to_low_scores = np.sort(scores).squeeze()[::-1]
  return high_to_low_texts, high_to_low_scores

def get_nn_text_w_audio(raw_texts, text_feats, img_feats, audio_feats, weight):
  scores = text_feats @ audio_feats.T + weight * text_feats @ img_feats.T
  scores = scores.squeeze()
  high_to_low_ids = np.argsort(scores).squeeze()[::-1]
  high_to_low_texts = [raw_texts[i] for i in high_to_low_ids]
  high_to_low_scores = np.sort(scores).squeeze()[::-1]
  return high_to_low_texts, high_to_low_scores
  
