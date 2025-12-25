import torch
import torch.nn as nn
from torchvision import models

class RiceQualityModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)

        # IMPORTANT: overwrite classifier EXACTLY
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.backbone.last_channel, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def get_model(num_classes=2):
    return RiceQualityModel(num_classes)
