import sys
sys.path.insert(0, "src")

import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from seed import set_seed
from config import student_config
from data import split_and_encode, build_train_val_dataloaders
from transforms import build_student_transforms
from datasets import StudentDataset
from models import build_resnet50
from train_utils import get_device, build_criterion, build_optimizer, build_scheduler
from early_stopping import EarlyStopping
from engine import train_one_epoch, evaluate

import wandb


def main():
    set_seed()
    
    WANDB_PROJECT = "birdCNN"
    WANDB_RUN_NAME = "student_hard"
    
    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=student_config,
    )
    
    df = pd.read_csv("data/train.csv")
    train_df, val_df, _ = split_and_encode(df)
    
    train_tf, val_tf = build_student_transforms()
    
    train_dataset = StudentDataset(train_df, transform=train_tf, base_dir="data")
    val_dataset = StudentDataset(val_df, transform=val_tf, base_dir="data")
    
    train_loader, val_loader = build_train_val_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=student_config["batch_size"],
        num_workers=student_config["num_workers"],
        pin_memory=True,
    )
    
    device = get_device()
    print(device)
    
    student = build_resnet50(freeze_until=student_config["freeze_until"]).to(device)
    
    criterion = build_criterion(label_smoothing=student_config["label_smoothing"])
    
    optimizer = build_optimizer(
        student,
        lr=student_config["lr"],
        weight_decay=student_config["weight_decay"],
    )
    
    scheduler = build_scheduler(
        optimizer,
        mode="min",
        factor=student_config["scheduler_factor"],
        patience=student_config["scheduler_patience"],
        threshold=student_config["scheduler_threshold"],
    )
    
    epochs = student_config["epochs"]
    
    os.makedirs("models", exist_ok=True)
    
    early_stopper = EarlyStopping(
        patience=student_config["early_stopping_patience"],
        save_path="models/best_student_hard_model.pth",
    )
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_time = train_one_epoch(
            student,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch=epoch,
        )
    
        val_loss, val_acc, val_time = evaluate(
            student,
            val_loader,
            criterion,
            device,
            epoch=epoch,
        )
    
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
    
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} ({train_time:.1f}s) | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.4f} ({val_time:.1f}s)"
        )
    
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "lr": current_lr,
                "time/train_sec": train_time,
                "time/val_sec": val_time,
            },
            step=epoch,
        )
    
        stop = early_stopper.step(val_loss, student)
        if stop:
            break
    
    wandb.finish()

if __name__ == "__main__":
    main()

