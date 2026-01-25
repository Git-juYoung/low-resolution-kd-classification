import sys
sys.path.insert(0, "src")

import pandas as pd
import os
import torch 

from seed import set_seed
from config import student_config, teacher_config
from data import split_and_encode, build_train_val_dataloaders
from transforms import build_student_transforms, build_teacher_transforms
from datasets import KDDataset
from models import build_resnet50
from train_utils import get_device, build_criterion, build_optimizer, build_scheduler
from early_stopping import EarlyStopping
from engine import train_one_epoch_kd, evaluate_kd_student

import wandb


def main():
    set_seed()
    
    WANDB_PROJECT = "birdCNN"
    WANDB_RUN_NAME = "student_kd"
    
    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=student_config
    )
    
    df = pd.read_csv("data/train.csv")
    
    train_df, val_df, _ = split_and_encode(df)
    
    student_train_tf, student_val_tf = build_student_transforms()
    _, teacher_eval_tf = build_teacher_transforms()
    
    train_dataset = KDDataset(
        train_df,
        student_transform=student_train_tf,
        teacher_transform=teacher_eval_tf
    )
    val_dataset = KDDataset(
        val_df,
        student_transform=student_val_tf,
        teacher_transform=teacher_eval_tf
    )
    
    train_loader, val_loader = build_train_val_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=student_config["batch_size"],
        num_workers=student_config["num_workers"],
        pin_memory=True
    )
    
    student = build_resnet50(
        freeze_until=student_config["freeze_until"]
    )
    
    teacher = build_resnet50(
        freeze_until=None
    )
    
    device = get_device()
    print(device)
    
    student = student.to(device)
    teacher = teacher.to(device)
    
    teacher_path = "models/best_teacher_model.pth"
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
        
    criterion = build_criterion(label_smoothing=student_config["label_smoothing"])
    
    optimizer = build_optimizer(
        student,
        lr=student_config["lr"],
        weight_decay=student_config["weight_decay"]
    )
    
    scheduler = build_scheduler(
        optimizer,
        mode="min",
        factor=student_config["scheduler_factor"],
        patience=student_config["scheduler_patience"],
        threshold=student_config["scheduler_threshold"]
    )
    
    epochs = student_config["epochs"]
    
    os.makedirs("models", exist_ok=True)
    
    early_stopper = EarlyStopping(
        patience=student_config["early_stopping_patience"],
        save_path="models/best_student_model.pth"
    )
    
    T = student_config["kd_temperature"]
    alpha = student_config["kd_alpha"]
    beta = student_config["kd_beta"]
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_time = train_one_epoch_kd(
            student=student,
            teacher=teacher,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            T=T,
            alpha=alpha,
            beta=beta,
            epoch=epoch
        )
    
        val_loss, val_acc, val_time = evaluate_kd_student(
            student=student,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch
        )
    
        scheduler.step(val_loss)
    
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} ({train_time:.1f}s) | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.4f} ({val_time:.1f}s)"
        )
    
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "lr": current_lr,
            "time/train_sec": train_time,
            "time/val_sec": val_time,
            "kd/T": T,
            "kd/alpha": alpha,
            "kd/beta": beta
        }, step=epoch)
    
        stop = early_stopper.step(val_loss, student)
        if stop:
            break
    
    wandb.finish()

if __name__ == "__main__":
    main()

