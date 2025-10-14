import torch
from torch import nn
from torch.nn import functional as F

from functools import partial


def embed_transform(points):
    batch_size, num_points, _ = points.shape
    
    transformed_coords = []
    for i in range(12):
        factor = 2 ** i
        transformed_coords.append((factor*points).sin())
        transformed_coords.append((factor*points).cos())
    return torch.cat(transformed_coords, -1).reshape(batch_size, num_points, -1)

class NucleusSegmenter(nn.Module):
    def __init__(self):
        super().__init__()

        activation_cls = partial(nn.GELU, approximate="tanh")
        k = 3
        p = k // 2

        self.plane_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=k, stride=2, padding=p),
            activation_cls(),
            nn.Conv2d(16, 32, kernel_size=k, stride=2, padding=p),
            activation_cls(),
            nn.Conv2d(32, 4, kernel_size=k, padding=p),
            activation_cls(),
        )

        self.global_encoder = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1), # 112
            activation_cls(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1), # 56
            activation_cls(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # 28
            activation_cls(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 14
            activation_cls(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 7
            activation_cls(),
            nn.AvgPool2d(7),
            nn.Flatten(1)
        )

        self.point_transform_fn = embed_transform

        self.point_encoder = nn.Sequential(
            nn.Linear(48, 256),
            activation_cls(),
            nn.Linear(256, 256),
        )

        patch_dim = 4 * 8 * 8
        self.head = nn.Sequential(
            nn.Linear(256 + patch_dim + 64, 512),
            activation_cls(),
            nn.Linear(512, 1),
        )

        self._patch_size = 8
        self._fmap_hw = 56

    @staticmethod
    def _build_sampling_grid(points_xy01, H, W, patch_size, device, dtype):
        B, N, _ = points_xy01.shape
    
        centers = points_xy01 * 2.0 - 1.0
        cx = centers[..., 0]
        cy = centers[..., 1]
    
        step_x = 2.0 / (W - 1)
        step_y = 2.0 / (H - 1)
    
        ps = patch_size
        half = (ps - 1) / 2.0
    
        offs_x = torch.linspace(-half, half, steps=ps, device=device, dtype=dtype) * step_x
        offs_y = torch.linspace(-half, half, steps=ps, device=device, dtype=dtype) * step_y
    
        offx2d, offy2d = torch.meshgrid(offs_x, offs_y, indexing='ij')
    
        grid_x = cx[..., None, None] + offx2d
        grid_y = cy[..., None, None] + offy2d
    
        grid = torch.stack((grid_x, grid_y), dim=-1)
        grid = grid.view(B * N, ps, ps, 2)
        return grid


    def _extract_local_patches(self, fmap, points_xy01):
        B, C, H, W = fmap.shape
        assert H == self._fmap_hw and W == self._fmap_hw, f"Expected {self._fmap_hw}x{self._fmap_hw} fmap"

        Bp, N, _ = points_xy01.shape
        assert Bp == B

        grid = self._build_sampling_grid(points_xy01, H, W, self._patch_size, fmap.device, fmap.dtype)

        fmap_rep = fmap.repeat_interleave(N, dim=0)

        patches = F.grid_sample(
            fmap_rep, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        patches = patches.view(B, N, C, self._patch_size, self._patch_size)
        return patches

    def forward(self, points, images, to_prob=False):
        B, N, _ = points.shape
        raw_points_xy01 = points

        points_transformed = self.point_transform_fn(points)           # (B, N, d_in)
        point_feats = self.point_encoder(points_transformed)           # (B, N, 256)

        fmap = self.plane_encoder(images)                              # (B, 4, 56, 56)
        global_context = self.global_encoder(images)                   # (B, 64)

        gc = global_context.unsqueeze(1).repeat(1, N, 1)

        patches = self._extract_local_patches(fmap, raw_points_xy01)   # (B, N, 4, 8, 8)
        patches = patches.flatten(start_dim=2)                         # (B, N, 256)
        
        fused = torch.cat([point_feats, patches, gc], dim=-1)           # (B, N, 512)
        logit = self.head(fused)                                        # (B, N, 1)

        if to_prob:
            return torch.sigmoid(logit)

        return logit

