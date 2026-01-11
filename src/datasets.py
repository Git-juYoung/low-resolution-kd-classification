#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


class TeacherDataset(Dataset):
    def __init__(self, df, transform, base_dir="data"):
        self.data = df.reset_index(drop=True)
        self.transform = transform
        self.base_dir = Path(base_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rel_path = self.data.iloc[index]["upscale_img_path"]
        img_path = self.base_dir / rel_path.lstrip("./")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = int(self.data.iloc[index]["label"])
        return image, label


class StudentDataset(Dataset):
    def __init__(self, df, transform, base_dir="data"):
        self.data = df.reset_index(drop=True)
        self.transform = transform
        self.base_dir = Path(base_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rel_path = self.data.iloc[index]["img_path"]
        img_path = self.base_dir / rel_path.lstrip("./")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = int(self.data.iloc[index]["label"])
        return image, label

        
class KDDataset(Dataset):
    def __init__(self, df, student_transform, teacher_transform, base_dir="data"):
        self.data = df.reset_index(drop=True)
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform
        self.base_dir = Path(base_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        lr_rel = self.data.iloc[index]["img_path"]
        hr_rel = self.data.iloc[index]["upscale_img_path"]

        lr_path = self.base_dir / lr_rel.lstrip("./")
        hr_path = self.base_dir / hr_rel.lstrip("./")

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        lr_img = self.student_transform(lr_img)
        hr_img = self.teacher_transform(hr_img)

        label = int(self.data.iloc[index]["label"])
        return lr_img, hr_img, label


class TeacherTestDataset(Dataset):
    def __init__(self, df, transform, base_dir="data"):
        self.data = df.reset_index(drop=True)
        self.transform = transform
        self.base_dir = __import__("pathlib").Path(base_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rel = self.data.iloc[idx]["upscale_img_path"]
        path = self.base_dir / rel.lstrip("./")
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = int(self.data.iloc[idx]["label"])
        return img, label

        
class KDTestDataset(Dataset):
    def __init__(self, df, transform, base_dir="data"):
        self.data = df.reset_index(drop=True)
        self.transform = transform
        self.base_dir = Path(base_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rel_path = self.data.iloc[index]["img_path"]
        img_path = self.base_dir / rel_path.lstrip("./")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = int(self.data.iloc[index]["label"])
        return image, label


class StudentTestDataset(Dataset):
    """
    LR-only test dataset for hard-loss baseline student.
    Uses `img_path` column.
    """
    def __init__(self, df, transform, base_dir="data"):
        self.data = df.reset_index(drop=True)
        self.transform = transform
        self.base_dir = Path(base_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rel_path = self.data.iloc[index]["img_path"]
        img_path = self.base_dir / rel_path.lstrip("./")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = int(self.data.iloc[index]["label"])
        return image, label