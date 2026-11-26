"""本作业内复用的人脸坐标网络与掩码损失。"""
import torch
from torch import nn


class FaceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((3, 3)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 3 * 3, 128), nn.ReLU(), nn.Linear(128, 30))

    def forward(self, x):
        return self.head(self.features(x))


def masked_mse(predictions, targets, mask):
    safe = torch.where(mask, targets, torch.zeros_like(targets))
    return ((predictions - safe).square() * mask).sum() / mask.sum().clamp_min(1)
