import torch
import numpy as np

from torchvision import tv_tensors
from torchvision.transforms import v2
from torch.utils.data import Dataset

import os
import random
import json


class SegmentationDataset(Dataset):
    def __init__(self, split="train", augmentations="train", device="cpu", num_positives=32, num_negatives=32):
        super().__init__()
        assert split in         ["train", "validation", "test"] 
        assert augmentations in [None, "train", "evaluation"]

        self.num_positives = num_positives
        self.num_negatives = num_negatives
        
        self.device        = device
        self.samples       = self._load_samples(split=split)
        self.augmentations = self._set_up_augmentations(augmentations=augmentations)

    def _load_samples(self, split):
        images_path                  = f"data/prepared/{split}/images"
        positive_sampling_masks_path = f"data/prepared/{split}/segmentation/positive_sampling_masks"
        negative_sampling_masks_path = f"data/prepared/{split}/segmentation/negative_sampling_masks"
         
        num_images = len([fn for fn in os.listdir(images_path) if fn.endswith(".npy")])

        samples = []
        for i in range(num_images):
            image = torch.from_numpy(np.load(f"{images_path}/{i:04d}.npy")).permute(2, 0, 1).to(self.device)
            positive_sampling_mask  = tv_tensors.Mask(
                torch.from_numpy(
                    np.load(f"{positive_sampling_masks_path}/{i:04d}.npy")
                )
            ).to(self.device)
            
            negative_sampling_mask = tv_tensors.Mask(
                torch.from_numpy(
                    np.load(f"{negative_sampling_masks_path}/{i:04d}.npy")
                )
            ).to(self.device)
            
            samples.append((image, positive_sampling_mask, negative_sampling_mask))

        return samples 
            
    def _set_up_augmentations(self, augmentations):
        if augmentations == "train":
            transforms = v2.Compose([
                v2.RandomRotation(180.),
                v2.RandomResizedCrop(size=(224, 224), antialias=True),
                v2.RandomHorizontalFlip(p=0.5)
            ])
        else:
            transforms = v2.Compose([
                v2.Resize((224, 224))
            ])
        return transforms

    def _apply_augmentations(self, image, ps_mask, ns_mask):
        ps_valid = tv_tensors.Mask(torch.isfinite(ps_mask).to(torch.uint8))
    
        batch = {
            "image": image,
            "ps": ps_mask,
            "ps_valid": ps_valid,
            "ns": ns_mask
        }
    
        aug = self.augmentations(batch)
    
        aug_ps_mask   = aug["ps"]
        aug_ps_valid  = aug["ps_valid"].to(torch.bool)
    
        aug_ps_mask = aug_ps_mask.clone()
        aug_ps_mask[~aug_ps_valid] = float('-inf')
    
        return aug["image"], aug_ps_mask, aug["ns"]

    def __len__(self):
        return len(self.samples)

    def _sample_positive_queries(self, aug_ps_mask, temp=1.0):
        probs = torch.softmax(aug_ps_mask.view(-1) / temp, dim=0)
        flat_idx = torch.multinomial(probs, self.num_positives, replacement=False)
            
        rows = flat_idx // 224
        cols = flat_idx % 224
        positive_queries = torch.stack([cols, rows], dim=1)
        return positive_queries

    def _sample_negative_queries(self, aug_ns_mask, temp=0.1):
        probs = torch.softmax(aug_ns_mask.view(-1) / temp, dim=0)
        flat_idx = torch.multinomial(probs, self.num_negatives, replacement=False)
            
        rows = flat_idx // 224
        cols = flat_idx % 224
        negative_queries = torch.stack([cols, rows], dim=1)
        return negative_queries

    def __getitem__(self, idx):
        image, ps_mask, ns_mask = self.samples[idx]
        
        (
            aug_image, 
            aug_ps_mask, 
            aug_ns_mask
        ) = self._apply_augmentations(image.clone(), ps_mask.clone(), ns_mask.clone())
        
        while (aug_ps_mask != float('-inf')).sum().int() < self.num_positives:
            (
                aug_image, 
                aug_ps_mask, 
                aug_ns_mask,
            ) = self._apply_augmentations(image.clone(), ps_mask.clone(), ns_mask.clone())
        
        positive_queries = self._sample_positive_queries(aug_ps_mask)
        negative_queries = self._sample_negative_queries(aug_ns_mask)
        queries = torch.cat((positive_queries, negative_queries), 0).float() / 224.
        
        segmentation_labels = torch.zeros(self.num_positives+self.num_negatives, device=self.device)
        segmentation_labels[:self.num_positives] = 1.
        
        return aug_image, queries, segmentation_labels



class ClassificationDataset(Dataset):
    def __init__(self, split="train", augmentations="train", device="cpu"):
        super().__init__()
        
        assert split in ["train", "validation", "test"]
        assert augmentations in ["train", "evaluation"]
        
        self.device        = device
        self.crops, self.labels = self._load_samples(split=split)
        self.transforms = self._prepare_transforms(augmentations=augmentations)

    def _load_samples(self, split):
        images_path = f"data/prepared/{split}/images"
        crops_path  = f"data/prepared/{split}/classification/crops"
        labels_path = f"data/prepared/{split}/classification/labels"
        
        num_images = len([fn for fn in os.listdir(images_path) if fn.endswith(".npy")])

        samples = []
        for i in range(num_images):
            crops = np.load(f"{crops_path}/{i:04d}.npy")
            labels = np.load(f"{labels_path}/{i:04d}.npy")

            samples.append((crops, labels))
            
        crops  = np.concat([s[0] for s in samples], 0)
        labels = np.concat([s[1] for s in samples], 0)
        return crops, labels

    def _prepare_transforms(self, augmentations):
        if augmentations == "train":
            transforms = v2.Compose([
                v2.RandomRotation(180.),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                
                v2.RandomPhotometricDistort(),
                v2.RandomGrayscale(p=0.1),
                v2.RandomAdjustSharpness(2, p=0.3)
            ])
        else:
            transforms = None
        return transforms

    def _apply_transforms(self, image):
        if self.transforms is None:
            return image
        return self.transforms(image)

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop, label = self.crops[idx], self.labels[idx]
        crop = torch.from_numpy(crop).permute(2, 0, 1).float()
        crop = self._apply_transforms(crop)
        return crop, label


class EvaluationDataset(Dataset):
    def __init__(self, split="train", device="cpu"):
        super().__init__()
        
        assert split in ["train", "validation", "test"]
        
        self.device        = device
        self.samples = self._load_samples(split=split)

    def _load_samples(self, split):
        images_path      = f"data/prepared/{split}/images"
        class_maps_path  = f"data/prepared/{split}/classification/masks"
        mask_lists_path  = f"data/prepared/{split}/evaluation/"
        
        
        num_images = len([fn for fn in os.listdir(images_path) if fn.endswith(".npy")])

        samples = []
        for i in range(num_images):
            image = torch.from_numpy(np.load(f"{images_path}/{i:04d}.npy")).permute(2, 0, 1).to(self.device)
            class_map = torch.from_numpy(np.load(f"{class_maps_path}/{i:04d}.npy")).to(self.device)

            with open(f"{mask_lists_path}/{i:04d}.json", "r") as f:
                mask_list = json.load(f)

            samples.append((image, class_map, mask_list))
        
        return samples            

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, class_map, mask_list = self.samples[idx]
        return (
            image.to(self.device), 
            class_map.to(self.device), 
            mask_list
        )
