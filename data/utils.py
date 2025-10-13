import numpy as np
import torch
from torch.nn import functional as F

import base64
import cv2


def crop_image(img: np.ndarray, center: tuple[int, int], window_size: int) -> np.ndarray:
    img = torch.from_numpy(img).permute(2, 0, 1)

    _, H, W = img.shape
    cx, cy = center
    half = window_size // 2

    left   = max(0, half - cx)
    right  = max(0, cx + half + 1 - W) if window_size % 2 else max(0, cx + half - W + 1)
    top    = max(0, half - cy)
    bottom = max(0, cy + half + 1 - H) if window_size % 2 else max(0, cy + half - H + 1)

    img_padded = F.pad(img, (left, right, top, bottom), value=0)

    cx_p, cy_p = cx + left, cy + top

    x1, x2 = cx_p - half, cx_p + half + (window_size % 2)
    y1, y2 = cy_p - half, cy_p + half + (window_size % 2)

    return img_padded[:, y1:y2, x1:x2].permute(1, 2, 0).numpy()


def base64_image_to_numpy(base64_string):
    img_bytes = base64.b64decode(base64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.cvtColor(cv2.imdecode(img_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    return img

def base64_mask_to_numpy(base64_string):
    img_bytes = base64.b64decode(base64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.cvtColor(cv2.imdecode(img_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    
    return img


