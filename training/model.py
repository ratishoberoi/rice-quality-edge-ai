import torch.nn as nn
from torchvision import models

def get_model():
    model = models.mobilenet_v2(weights="DEFAULT")
    
    for param in model.features.parameters():
        param.requires_grad = False  # transfer learning

    model.classifier = nn.Sequential(
        nn.Linear(model.last_channel, 2)
    )
    return model
