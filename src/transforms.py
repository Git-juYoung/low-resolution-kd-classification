#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from torchvision import models, transforms



def build_student_transforms():
    weights = models.ResNet50_Weights.DEFAULT
    tf = weights.transforms(antialias=True)

    mean = getattr(tf, "mean", (0.485, 0.456, 0.406))
    std  = getattr(tf, "std",  (0.229, 0.224, 0.225))

    train_transform = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03,
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1, value="random"),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_test_transform



def build_teacher_transforms():
    weights = models.ResNet50_Weights.DEFAULT
    tf = weights.transforms(antialias=True)

    mean = getattr(tf, "mean", (0.485, 0.456, 0.406))
    std  = getattr(tf, "std",  (0.229, 0.224, 0.225))

    teacher_train_transform = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.08,
            contrast=0.08,
            saturation=0.08,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    teacher_val_test_transform = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return teacher_train_transform, teacher_val_test_transform