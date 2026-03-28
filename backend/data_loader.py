# ==========================================
# data_loader.py
# Federated Multimodal Dataset Preparation
# ==========================================

import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config


# ==========================================
# Global Encoders
# ==========================================

loc_encoder = LabelEncoder()
label_encoder = LabelEncoder()


# ==========================================
# Metadata Preprocessing
# ==========================================

def preprocess_metadata(df, fit=False):
    df = df.copy()

    # Age
    df['age'] = df['age'].fillna(df['age'].median())
    df['age'] = df['age'] / 100.0

    # Sex
    df['sex'] = df['sex'].fillna("unknown")
    df['sex_encoded'] = df['sex'].map({
        'male': 0,
        'female': 1,
        'unknown': 2
    })

    # Localization
    if fit:
        df['loc_encoded'] = loc_encoder.fit_transform(df['localization'])
    else:
        df['loc_encoded'] = loc_encoder.transform(df['localization'])

    # Label
    if fit:
        df['label'] = label_encoder.fit_transform(df['dx'])
    else:
        df['label'] = label_encoder.transform(df['dx'])

    return df


# ==========================================
# Dataset Class
# ==========================================

class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, img_dirs, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dirs = img_dirs
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def find_image_path(self, image_id):
        for folder in self.img_dirs:
            path = os.path.join(folder, image_id + ".jpg")
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"{image_id}.jpg not found in provided folders")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.find_image_path(row['image_id'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        metadata = torch.tensor([
            row['age'],
            row['sex_encoded'],
            row['loc_encoded']
        ], dtype=torch.float32)

        label = torch.tensor(row['label'], dtype=torch.long)

        return image, metadata, label


# ==========================================
# Create Federated DataLoaders
# ==========================================

def create_dataloaders():

    df = pd.read_csv(config.METADATA_PATH)

    # Global split
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SPLIT,
        stratify=df["dx"],
        random_state=config.RANDOM_SEED
    )

    # Fit encoders ONLY on training data
    train_df = preprocess_metadata(train_df, fit=True)
    test_df = preprocess_metadata(test_df)

    # Split training into 2 equal Non-IID clients
    train_df = train_df.sample(frac=1, random_state=config.RANDOM_SEED)

    mid = len(train_df) // 2
    client1_df = train_df.iloc[:mid].sort_values(by="label")
    client2_df = train_df.iloc[mid:].sort_values(by="label", ascending=False)

    # Image directories (LOCAL)
    IMG_DIRS = [
        "HAM10000_images",
        os.path.join("HAM10000_balanced", "images")
    ]

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            config.IMAGE_SIZE,
            scale=(0.75, 1.0),
            interpolation=InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
        transforms.RandomAffine(degrees=0, shear=10, scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE),
            interpolation=InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Datasets
    client1_dataset = HAM10000Dataset(client1_df, IMG_DIRS, train_transform)
    client2_dataset = HAM10000Dataset(client2_df, IMG_DIRS, train_transform)
    global_test_dataset = HAM10000Dataset(test_df, IMG_DIRS, val_transform)

    # Loaders
    client1_loader = DataLoader(
        client1_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    client2_loader = DataLoader(
        client2_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    global_test_loader = DataLoader(
        global_test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    return client1_loader, client2_loader, global_test_loader