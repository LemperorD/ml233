"""本作业内复用的小型 U-Net；输入输出为归一化单通道灰度。"""
import torch
from torch import nn
from torch.nn import functional as F


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU())


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2 = block(1, 16), block(16, 32)
        self.center = block(32, 64)
        self.dec2, self.dec1 = block(96, 32), block(48, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        height, width = x.shape[-2:]
        x = F.pad(x, (0, (-width) % 4, 0, (-height) % 4), mode='replicate')
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.center(F.max_pool2d(b, 2))
        d = self.dec2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], dim=1))
        e = self.dec1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a], dim=1))
        return self.output(e).sigmoid()[..., :height, :width]
