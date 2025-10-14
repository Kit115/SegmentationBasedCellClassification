import os
import json
import numpy as np

from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import torch
from torch.nn import functional as F

from utils import (
    base64_image_to_numpy,
    base64_mask_to_numpy,
    crop_image
)

import random

label_to_idx = {
    "Myofibroblast":                0,
    "UndifferentiatedMSC":          1,
    "PartiallydifferentiatedMSC":   -1
}


def load_json_as_obj(path):
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return obj


def create_segmentation_mask(label_object):
    segmentation_mask = np.zeros((1040, 1388), dtype=np.float32)

    for mask_object in label_object["shapes"]:
        x_min = int(mask_object['points'][0][0])
        y_min = int(mask_object['points'][0][1])
        x_max = int(mask_object['points'][1][0] + 1)
        y_max = int(mask_object['points'][1][1] + 1)
        
        mask = base64_mask_to_numpy(mask_object['mask'])[:, :, 0]
        segmentation_mask[y_min:y_max, x_min:x_max] = mask
    return segmentation_mask

def create_classification_mask(label_object):
    classification_mask = -np.ones((1040, 1388), dtype=np.int8) # -1 as the default label

    for mask_object in label_object["shapes"]:
        x_min = int(mask_object['points'][0][0])
        y_min = int(mask_object['points'][0][1])
        x_max = int(mask_object['points'][1][0] + 1)
        y_max = int(mask_object['points'][1][1] + 1)

        cell_label = label_to_idx[mask_object['label']]
        mask = base64_mask_to_numpy(mask_object['mask'])[:, :, 0]
        mask[mask == 0] = -1
        mask[mask == 1] = cell_label
        classification_mask[y_min:y_max, x_min:x_max] = mask.astype(np.int8)
    return classification_mask 

def create_positive_sampling_mask(segmentation_mask):
    positive_sampling_mask = segmentation_mask.copy()
    positive_sampling_mask[segmentation_mask == 0] = float("-inf")
    return positive_sampling_mask 

def create_negative_sampling_mask(segmentation_mask, sigma=64):
    negative_sampling_mask                          = gaussian_filter(segmentation_mask, sigma=sigma)
    negative_sampling_mask[segmentation_mask == 1]  = float("-inf")
    return negative_sampling_mask


def create_dataset(label_paths, save_path):
    os.makedirs(f"{save_path}/images", exist_ok=True)
    os.makedirs(f"{save_path}/segmentation/positive_sampling_masks", exist_ok=True)
    os.makedirs(f"{save_path}/segmentation/negative_sampling_masks", exist_ok=True)
    os.makedirs(f"{save_path}/classification/masks", exist_ok=True)
    os.makedirs(f"{save_path}/classification/crops", exist_ok=True)
    os.makedirs(f"{save_path}/classification/labels", exist_ok=True)
    os.makedirs(f"{save_path}/evaluation", exist_ok=True)

    for i, label_path in enumerate(tqdm(label_paths)):
        label_object = load_json_as_obj(label_path)

        numpy_image = base64_image_to_numpy(label_object["imageData"])

        segmentation_mask   = create_segmentation_mask(label_object)
        positive_sampling_mask = create_positive_sampling_mask(segmentation_mask)
        negative_sampling_mask = create_negative_sampling_mask(segmentation_mask)

        classification_mask = create_classification_mask(label_object)

        piecewise_masks = []
        classification_crops  = []
        classification_labels = []
        
        for mask_object in label_object["shapes"]:
            cell_label = label_to_idx[mask_object['label']]

            min_x = mask_object['points'][0][0]
            min_y = mask_object['points'][0][1]
            max_x = mask_object['points'][1][0] + 1
            max_y = mask_object['points'][1][1] + 1
            
            piecewise_masks.append({
                "label":            cell_label,
                "min_x":            min_x,
                "min_y":            min_y,
                "max_x":            max_x,
                "max_y":            max_y,
                "base64_mask":      mask_object['mask'],
            })
            
            if cell_label == -1:
                continue
                
            center_x = int((min_x + max_x) / 2)
            center_y = int((min_y + max_y) / 2)

            classification_crops.append(crop_image(numpy_image, center=(center_x, center_y), window_size=224))
            classification_labels.append(cell_label)
            

        np.save(f"{save_path}/images/{i:04d}.npy", numpy_image)
        np.save(f"{save_path}/segmentation/negative_sampling_masks/{i:04d}.npy", negative_sampling_mask)
        np.save(f"{save_path}/segmentation/positive_sampling_masks/{i:04d}.npy", positive_sampling_mask)
        
        np.save(f"{save_path}/classification/masks/{i:04d}.npy", classification_mask)
        np.save(f"{save_path}/classification/crops/{i:04d}.npy", np.stack(classification_crops, 0))
        np.save(f"{save_path}/classification/labels/{i:04d}.npy", np.array(classification_labels))
        

        with open(f"{save_path}/evaluation/{i:04d}.json", "w+") as f:
            f.write(json.dumps(piecewise_masks, indent=4))



def main(args):
    source_path = "data/raw/labels"
    label_paths = [f"{source_path}/{ln}" for ln in os.listdir(f"{source_path}/")if ln.endswith('.json')]

    rng = random.Random(args.shuffle_seed)
    rng.shuffle(label_paths)

    train_frac, val_frac, test_frac = args.split_sizes

    test_end_idx    = int(test_frac * len(label_paths))
    val_end_idx     = test_end_idx + int(val_frac * len(label_paths))

    test_label_paths    = label_paths[:test_end_idx]
    val_label_paths     = label_paths[test_end_idx:val_end_idx]
    train_label_paths   = label_paths[val_end_idx:]

    create_dataset(train_label_paths,   "data/prepared/train")
    create_dataset(val_label_paths,     "data/prepared/validation")
    create_dataset(test_label_paths,    "data/prepared/test")
    


if __name__ == "__main__":
    import argparse

    parser  = argparse.ArgumentParser()
    parser.add_argument('--split-sizes',    nargs=3, type=float,  default=[0.7, 0.1, 0.2])
    parser.add_argument('--shuffle-seed',   type=int,   default=0)
    
    args    = parser.parse_args()
    main(args)


