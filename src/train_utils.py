import torch
import torch.nn as nn
import torch.optim as optim

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_criterion(label_smoothing):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def build_optimizer(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay)

def build_scheduler(optimizer, mode, factor, patience, threshold):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=threshold,
    )

