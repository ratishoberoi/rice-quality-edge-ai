import torch
import torch.nn as nn
from torchvision import models

class RiceQualityNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = models.mobilenet_v2(pretrained=True)

        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
