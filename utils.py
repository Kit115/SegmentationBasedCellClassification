import torch
import numpy as np
import cv2
import base64

from matplotlib import pyplot as plt

from scipy import ndimage as ndi

from torch.nn import functional as F
from typing import Iterable, List, Tuple, Union


def plot_image(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    H, W = img.shape[:2]

    plt.imshow(img)
    
    plt.axis('off')
    plt.show()
    
def plot_image_with_labeled_keypoints(img_tensor, keypoints, keypoint_labels):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    H, W = img.shape[:2]

    plt.imshow(img)
    
    for label, (x, y) in zip(keypoint_labels, keypoints):
        px, py = x * W, y * H  # scale to pixel coords
        color = {
            -1: "red",
            0: "green",
            1: "blue"
        }[label.item()]
        plt.scatter(px, py, c=color, s=30, marker='o', edgecolors='white')
    
    plt.axis('off')
    plt.show()

def plot_image_with_keypoints(img_tensor, keypoints):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    H, W = img.shape[:2]

    plt.imshow(img)
    
    for (x, y) in keypoints:
        px, py = x * W, y * H  # scale to pixel coords
        plt.scatter(px, py, c='red', s=30, marker='o', edgecolors='white')
    
    plt.axis('off')
    plt.show()


def find_heatmap_peaks(
    H,
    *,
    sigma=1.0,
    nms_radius=8,
    threshold_rel=0.2,
    threshold_abs=None,
    min_confidence = 0.6
):
    H = np.asarray(H, dtype=float)
    if H.ndim != 2:
        raise ValueError("H must be a 2D array")

    Hs = ndi.gaussian_filter(H, sigma=sigma) if sigma and sigma > 0 else H

    if threshold_abs is None:
        t = float(Hs.max()) * float(threshold_rel)
    else:
        t = float(threshold_abs)

    if not np.isfinite(t):
        return []

    size = 2 * int(nms_radius) + 1
    Hmax = ndi.maximum_filter(Hs, size=size, mode="nearest")
    candidates = (Hs == Hmax) & (Hs >= t)

    structure = np.ones((3, 3), dtype=bool)
    labeled, n_labels = ndi.label(candidates, structure=structure)
    if n_labels == 0:
        return []

    peak_indices = []
    for label_id in range(1, n_labels + 1):
        ys, xs = np.nonzero(labeled == label_id)
        vals = Hs[ys, xs]
        k = np.argmax(vals)
        peak_indices.append((ys[k], xs[k]))

    peaks = []
    for rc in peak_indices:
        entry = {"rc": rc, "score": float(Hs[rc])}
        peaks.append(entry)

    # Sort final output by score descending
    peaks.sort(key=lambda p: p["score"], reverse=True)
    return torch.from_numpy(np.stack([p["rc"]for p in peaks if p["score"] > (min_confidence)], 0)[:, [1, 0]]).float() / 224.
        
def make_class_predictions_for_keypoints(keypoints, class_map, window_size=1):
    keypoints = (keypoints * 224).long()
    class_predictions = []
    confidences = []
    
    for keypoint in keypoints:
        x, y = keypoint
        x, y = int(x), int(y)

        y_min, y_max = y - window_size, y + window_size + 1
        x_min, x_max = y - window_size, y + window_size + 1

        summed_logits     = class_map[y_min:y_max, x_min:x_max, :].sum((0, 1))
        probs = summed_logits / summed_logits.sum()
        
        pred       = probs.argmax().item()
        confidence = probs[pred].item()
        
        class_predictions.append(pred)
        confidences.append(confidence)
        
    return class_predictions, confidences



def heatmap_to_bboxes(
    heatmap: Union[np.ndarray, "torch.Tensor"],
    *,
    threshold: float = 0.5,
    connectivity: int = 1,
    min_area: int = 1
) -> List[Tuple[int, int, int, int]]:
    hw = heatmap.numpy()
    mask = hw > threshold  # keep islands strictly above 0.5

    # Connected components with SciPy (very common dependency).
    # Fallback to scikit-image if you prefer; both are fine.
    from scipy import ndimage as ndi

    structure = ndi.generate_binary_structure(2, connectivity)  # 2D, 4 or 8
    labeled, n = ndi.label(mask, structure=structure)

    boxes: List[Tuple[int, int, int, int]] = []
    centers = []
    if n == 0:
        return boxes

    # ndi.find_objects returns a list of slices for each label
    obj_slices = ndi.find_objects(labeled)
    for sl in obj_slices:
        if sl is None:
            continue
        (ys, xs) = sl  # rows, cols
        h = ys.stop - ys.start
        w = xs.stop - xs.start
        area = h * w if min_area > 1 else int((labeled[ys, xs] > 0).sum())
        if area < min_area:
            continue
        # Convert to (x1,y1,x2,y2) with exclusive max coords
        boxes.append((int(xs.start), int(ys.start), int(xs.stop), int(ys.stop)))
        centers.append(((xs.start + xs.stop) / 2 / 224., (ys.start + ys.stop) / 2 / 224.))
        

    return boxes, centers

def show_heatmap_with_boxes(
    heatmap_tensor: "torch.Tensor",
    boxes: Iterable[Tuple[int, int, int, int]],
    *,
    figsize: Tuple[int, int] = (5, 5),
    linewidth: float = 1.5
) -> None:
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not isinstance(heatmap_tensor, torch.Tensor):
        raise TypeError("heatmap_tensor must be a torch.Tensor")
    if heatmap_tensor.ndim != 3 or heatmap_tensor.shape[0] != 1:
        raise ValueError(f"Expected shape (1, 224, 224), got {tuple(heatmap_tensor.shape)}")

    img = heatmap_tensor[0].detach().cpu().float().numpy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, origin="upper")  # default colormap; change if you like
    for (x1, y1, x2, y2) in boxes:
        w, h = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), w, h, fill=False, linewidth=linewidth)
        ax.add_patch(rect)
    ax.set_axis_off()
    plt.show()

def show_heatmap_with_boxes_and_keypoints(
    heatmap_tensor: "torch.Tensor",
    boxes: Iterable[Tuple[int, int, int, int]],
    keypoints: Iterable[Tuple[int, int]],
    *,
    figsize: Tuple[int, int] = (5, 5),
    linewidth: float = 1.5
) -> None:
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not isinstance(heatmap_tensor, torch.Tensor):
        raise TypeError("heatmap_tensor must be a torch.Tensor")
    if heatmap_tensor.ndim != 3 or heatmap_tensor.shape[0] != 1:
        raise ValueError(f"Expected shape (1, 224, 224), got {tuple(heatmap_tensor.shape)}")

    img = heatmap_tensor[0].detach().cpu().float().numpy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, origin="upper")  # default colormap; change if you like
    for (x1, y1, x2, y2), (cx, cy) in zip(boxes, keypoints):
        w, h = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), w, h, fill=False, linewidth=linewidth, color="r")
        ax.add_patch(rect)
        plt.scatter(cx * 224, cy * 224, c='red', s=30, marker='o', edgecolors='white')
    ax.set_axis_off()
    plt.show()
