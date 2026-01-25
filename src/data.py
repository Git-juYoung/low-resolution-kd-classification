from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

def split_and_encode(df, test_size=0.2, seed=42):
    train_df, val_test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        val_test_df, test_size=0.5, stratify=val_test_df["label"], random_state=seed
    )

    le = LabelEncoder()
    le.fit(train_df["label"])

    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    train_df["label"] = le.transform(train_df["label"])
    val_df["label"]   = le.transform(val_df["label"])
    test_df["label"]  = le.transform(test_df["label"])

    return train_df, val_df, test_df

def build_train_val_dataloaders(
    train_dataset,
    val_dataset,
    batch_size,
    num_workers,
    pin_memory,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def build_test_dataloader(
    test_dataset,
    batch_size,
    num_workers,
    pin_memory,
):
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return test_loader

