#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.insert(0, "src")

from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

from seed import set_seed
from config import student_config
from data import split_and_encode, build_test_dataloader
from transforms import build_student_transforms
from datasets import StudentTestDataset
from models import build_resnet50
from train_utils import get_device, build_criterion
from engine import evaluate


def main():
    set_seed()
    
    df = pd.read_csv("data/train.csv")
    _, _, test_df = split_and_encode(df)
    
    _, test_tf = build_student_transforms()
    
    test_dataset = StudentTestDataset(test_df, transform=test_tf, base_dir="data")
    test_loader = build_test_dataloader(
        test_dataset,
        batch_size=student_config["batch_size"],
        num_workers=student_config["num_workers"],
        pin_memory=True,
    )
    
    device = get_device()
    
    student = build_resnet50(
        freeze_until=student_config["freeze_until"]
    ).to(device)
    
    best_path = "models/best_student_hard_model.pth"
    student.load_state_dict(torch.load(best_path, map_location=device))
    
    criterion = build_criterion(label_smoothing=0.0)
    
    test_loss, test_acc, test_time = evaluate(
        student,
        test_loader,
        criterion,
        device,
        epoch=None
    )
    
    print(f"{test_loss:.4f}, Acc: {test_acc:.4f} ({test_time:.1f}s)")

if __name__ == "__main__":
    main()

