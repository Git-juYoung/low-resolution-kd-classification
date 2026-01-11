#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import time
from tqdm.auto import tqdm

import torch.nn.functional as F



def train_one_epoch(model, loader, optimizer, criterion, device, epoch=None):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0

    start = time.time()

    pbar = tqdm(loader, desc=f"Train{'' if epoch is None else f' [{epoch}]'}", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_correct += (outputs.argmax(1) == labels).sum().item()
        total += bs

        pbar.set_postfix(
            loss=f"{running_loss/total:.4f}",
            acc=f"{running_correct/total:.4f}",
        )

    epoch_time = time.time() - start
    return running_loss / total, running_correct / total, epoch_time


def evaluate(model, loader, criterion, device, epoch=None):
    model.eval()
    running_loss, running_correct, total = 0.0, 0, 0

    start = time.time()

    pbar = tqdm(loader, desc=f"Val{'' if epoch is None else f' [{epoch}]'}", leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            bs = images.size(0)
            running_loss += loss.item() * bs
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total += bs

            pbar.set_postfix(
                loss=f"{running_loss/total:.4f}",
                acc=f"{running_correct/total:.4f}",
            )

    epoch_time = time.time() - start
    return running_loss / total, running_correct / total, epoch_time



def train_one_epoch_kd(student, teacher, loader, optimizer, criterion, device, T, alpha, beta, epoch=None):
    student.train()
    teacher.eval()

    running_loss, running_correct, total = 0.0, 0, 0
    start = time.time()

    pbar = tqdm(loader, desc=f"Train{'' if epoch is None else f' [{epoch}]'}", leave=False)

    for lr_images, hr_images, labels in pbar:
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        student_logits = student(lr_images)

        with torch.no_grad():
            teacher_logits = teacher(hr_images)

        hard_loss = criterion(student_logits, labels)

        student_log_prob = F.log_softmax(student_logits / T, dim=1)
        teacher_prob = F.softmax(teacher_logits / T, dim=1)

        soft_loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (T * T)

        loss = alpha * hard_loss + beta * soft_loss
        loss.backward()
        optimizer.step()

        bs = lr_images.size(0)
        running_loss += loss.item() * bs
        running_correct += (student_logits.argmax(1) == labels).sum().item()
        total += bs

        pbar.set_postfix(
            loss=f"{running_loss/total:.4f}",
            acc=f"{running_correct/total:.4f}",
            hard=f"{hard_loss.item():.4f}",
            soft=f"{soft_loss.item():.4f}",
        )

    epoch_time = time.time() - start
    return running_loss / total, running_correct / total, epoch_time



def evaluate_kd_student(student, loader, criterion, device, epoch=None):
    student.eval()
    running_loss, running_correct, total = 0.0, 0, 0

    start = time.time()
    pbar = tqdm(loader, desc=f"Val{'' if epoch is None else f' [{epoch}]'}", leave=False)

    with torch.no_grad():
        for lr_images, _, labels in pbar:
            lr_images = lr_images.to(device)
            labels = labels.to(device)

            outputs = student(lr_images)
            loss = criterion(outputs, labels)

            bs = lr_images.size(0)
            running_loss += loss.item() * bs
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total += bs

            pbar.set_postfix(
                loss=f"{running_loss/total:.4f}",
                acc=f"{running_correct/total:.4f}",
            )

    epoch_time = time.time() - start
    return running_loss / total, running_correct / total, epoch_time

