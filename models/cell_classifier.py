import torch
from torch import nn

class CellClassifier(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3),  # 8 x 112 x 112
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2), # 16 x 56 x 56
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32 x 28 x 28
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 128 x 7 x 7
            nn.AvgPool2d(7),
            nn.Flatten(1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 2)
        )
        
    def forward(self, images):
        return self.layers(images)




