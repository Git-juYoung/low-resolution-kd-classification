import torch.nn as nn
from torchvision import models


def build_resnet50(freeze_until, num_classes=25):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for p in model.parameters():
        p.requires_grad = False

    unfreeze = False
    for name, module in model.named_children():
        if name == freeze_until:
            unfreeze = True
            continue

        if unfreeze:
            for p in module.parameters():
                p.requires_grad = True

    for p in model.fc.parameters():
        p.requires_grad = True

    return model

