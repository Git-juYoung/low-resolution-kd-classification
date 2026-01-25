import sys
sys.path.insert(0, "src")

import torch
import pandas as pd

from seed import set_seed
from config import student_config
from data import split_and_encode, build_test_dataloader
from transforms import build_student_transforms
from datasets import KDTestDataset
from models import build_resnet50
from train_utils import get_device, build_criterion
from engine import evaluate


def main():
    set_seed()
    
    df = pd.read_csv("data/train.csv")
    
    _, _, test_df = split_and_encode(df)
    
    _, test_transform = build_student_transforms()
    
    test_dataset = KDTestDataset(test_df, transform=test_transform)
    
    test_loader = build_test_dataloader(
        test_dataset,
        batch_size=student_config["batch_size"],
        num_workers=student_config["num_workers"],
        pin_memory=True,
    )
    
    device = get_device()
    
    model = build_resnet50(
        freeze_until=student_config["freeze_until"]
    )
    model.to(device)
    
    student_path = "models/best_student_model.pth"
    
    model.load_state_dict(
        torch.load(student_path, map_location=device)
    )
    
    criterion = build_criterion(label_smoothing=0.0)
    
    test_loss, test_acc, test_time = evaluate(
        model,
        test_loader,
        criterion,
        device
    )
    
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} ({test_time:.1f}s)")

if __name__ == "__main__":
    main()

