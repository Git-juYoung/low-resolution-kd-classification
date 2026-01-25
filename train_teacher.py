import pandas as pd
import os
import sys
sys.path.append("src")

from seed import set_seed
from config import teacher_config
from data import split_and_encode, build_train_val_dataloaders
from transforms import build_teacher_transforms
from datasets import TeacherDataset
from models import build_resnet50
from train_utils import get_device, build_criterion, build_optimizer, build_scheduler
from early_stopping import EarlyStopping
from engine import train_one_epoch, evaluate

import wandb


def main():
    set_seed()
    
    WANDB_PROJECT = "birdCNN"
    WANDB_RUN_NAME = "teacher"
    
    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=teacher_config,
    )
    
    df = pd.read_csv("data/train.csv")
    
    train_df, val_df, _ = split_and_encode(df)
    
    train_transform, val_transform = build_teacher_transforms()
    
    train_dataset = TeacherDataset(train_df, transform=train_transform)
    val_dataset = TeacherDataset(val_df, transform=val_transform)
    
    train_loader, val_loader = build_train_val_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=teacher_config["batch_size"],
        num_workers=teacher_config["num_workers"],
        pin_memory=True,
    )
    
    model = build_resnet50(
        freeze_until=teacher_config["freeze_until"],
    )
    
    device = get_device()
    print(device)
    
    model = model.to(device)
    
    criterion = build_criterion(label_smoothing=teacher_config["label_smoothing"])
    
    optimizer = build_optimizer(
        model,
        lr=teacher_config["lr"],
        weight_decay=teacher_config["weight_decay"],
    )
    
    scheduler = build_scheduler(
        optimizer,
        mode="min",
        factor=teacher_config["scheduler_factor"],
        patience=teacher_config["scheduler_patience"],
        threshold=teacher_config["scheduler_threshold"],
    )
    
    epochs = teacher_config["epochs"]
    
    os.makedirs("models", exist_ok=True)
    
    early_stopper = EarlyStopping(
        patience=teacher_config["early_stopping_patience"],
        save_path="models/best_teacher_model.pth",
    )
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_time = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch=epoch
        )
    
        val_loss, val_acc, val_time = evaluate(
            model,
            val_loader,
            criterion,
            device,
            epoch=epoch
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
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "lr": current_lr,
        },
        step=epoch
        )
    
        stop = early_stopper.step(val_loss, model)
        if stop:
            break
    
    wandb.finish()

if __name__ == "__main__":
    main()

