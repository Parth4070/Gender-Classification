import torch.nn as nn
from torchvision import models


def create_resnet18(num_classes: int = 2,freeze_backbone: bool = True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    num_features = model.fc.in_features

    model.fc = nn.Linear(num_features,num_classes)

    return model