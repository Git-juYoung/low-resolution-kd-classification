#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
sys.path.insert(0, "src")

import torch
import pandas as pd

from seed import set_seed
from config import teacher_config
from data import split_and_encode, build_test_dataloader
from transforms import build_teacher_transforms
from datasets import TeacherTestDataset
from models import build_resnet50
from train_utils import get_device, build_criterion
from engine import evaluate


def main():
    set_seed()
    
    df = pd.read_csv("data/train.csv")
    _, _, test_df = split_and_encode(df)
    
    _, teacher_test_transform = build_teacher_transforms()
    
    test_dataset = TeacherTestDataset(test_df, transform=teacher_test_transform, base_dir="data")
    
    test_loader = build_test_dataloader(
        test_dataset,
        batch_size=teacher_config["batch_size"],
        num_workers=teacher_config["num_workers"],
        pin_memory=True,
    )
    
    device = get_device()
    
    teacher = build_resnet50(
        freeze_until=teacher_config["freeze_until"]
    ).to(device)
    
    best_teacher_path = "models/best_teacher_model.pth"
    teacher.load_state_dict(torch.load(best_teacher_path, map_location=device))
    
    criterion = build_criterion(label_smoothing=0.0)
    
    test_loss, test_acc, test_time = evaluate(
        teacher,
        test_loader,
        criterion,
        device
    )
    
    print(f"[Teacher Test] Loss: {test_loss:.4f}, Acc: {test_acc:.4f} ({test_time:.1f}s)")

if __name__ == "__main__":
    main()

