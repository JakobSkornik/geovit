import random
import cv2
import glob
import io
import json
import msgpack
import numpy as np
import os
import torch

# import reverse_geocoder as rg
from PIL import Image
from albumentations import (
    HorizontalFlip,
    ShiftScaleRotate,
    CLAHE,
    RandomBrightnessContrast,
    HueSaturationValue,
    Normalize,
    Resize,
    Compose,
)
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, random_split
from utils import denormalize_coordinates, normalize_coordinates


DIR = "shards/"
TRAIN_RATIO = 0.9


def load_data_from_shards(shard_fnames):
    records = []
    for shard_fname in shard_fnames:
        with open(shard_fname, "rb") as infile:
            for record in msgpack.Unpacker(infile, raw=False):
                records.append(record)
    return records


def get_image(record):
    return Image.open(io.BytesIO(record["image"]))


class StreetViewDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_fnames = os.listdir(img_dir)
        self.transform = transform
        self.country_indices = self.__load_country_indices__()

    def __load_country_indices__(self):
        with open("countries.json", "r") as fp:
            return json.load(fp)

    def __len__(self):
        return len(self.img_fnames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_fnames[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            transformed = self.transform(image=np.array(image))
            image = transformed["image"]

        # Parse coordinates from filename
        coords = self.img_fnames[idx][:-4]
        split_coords = coords.split(",")
        if len(split_coords) != 2:
            raise ValueError(
                f"Filename {self.img_fnames[idx]} is not in the expected format latitude,longitude.png"
            )

        latitude, longitude = map(float, split_coords)
        coordinates_tensor = torch.tensor(
            [latitude, longitude],
            dtype=torch.float32,
        )

        return {
            "image": image,
            "coordinates": coordinates_tensor,
            "country": self.country_indices[idx],
        }


class StreetViewTestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_fnames = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_fnames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_fnames[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            transformed = self.transform(image=np.array(image))
            image = transformed["image"]

        # Parse coordinates from filename
        coords = self.img_fnames[idx][:-4]
        split_coords = coords.split(",")
        if len(split_coords) != 2:
            raise ValueError(
                f"Filename {self.img_fnames[idx]} is not in the expected format latitude,longitude.jpg"
            )

        latitude, longitude = map(float, split_coords)

        coordinates = torch.tensor(
            normalize_coordinates([latitude, longitude]),
            dtype=torch.float32,
        )

        return {"image": image, "coordinates": coordinates}


class MsgPackData(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        record = self.data[index]
        image = get_image(record)

        if self.transform:
            transformed = self.transform(image=np.array(image))
            image = transformed["image"]

        coordinates = torch.tensor(
            normalize_coordinates([record["latitude"], record["longitude"]]),
            dtype=torch.float32,
        )
        return {"image": image, "coordinates": coordinates}

    def __len__(self):
        return len(self.data)


class MsgPackDataWithCountries(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.country_indices = self.__load_country_indices__()

    def __load_country_indices__(self):
        with open("countries.json", "r") as fp:
            return json.load(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(io.BytesIO(item["image"]))

        if self.transform:
            transformed = self.transform(image=np.array(image))
            image = transformed["image"]

        # Get coordinates
        latitude, longitude = item["latitude"], item["longitude"]
        coordinates = torch.tensor(
            normalize_coordinates([latitude, longitude]), dtype=torch.float32
        )

        return {
            "image": image,
            "coordinates": coordinates,
            "country": self.country_indices[idx],
        }


def get_dataloaders_old(config):
    start_idx = config["start_idx"]
    num_shards = config["num_shards"]
    batch_size = config["batch_size"]
    image_size = config["image_size"]
    workers = config["workers"]
    pin_memory = config["pin_memory"]

    AUGMENTATIONS_TRAIN = Compose(
        [
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, p=0.5),
            HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.9
            ),
            CLAHE(p=1.0, clip_limit=2.0),
            ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.8,
            ),
            Resize(image_size, image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    AUGMENTATIONS_VAL = Compose(
        [
            Resize(image_size, image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    all_shard_fnames = glob.glob(os.path.join(DIR, "*.msg"))
    selected_shard_fnames = all_shard_fnames[start_idx : start_idx + num_shards]

    all_data = load_data_from_shards(selected_shard_fnames)
    train_length = int(TRAIN_RATIO * len(all_data))
    val_length = len(all_data) - train_length
    train_data, val_data = random_split(all_data, [train_length, val_length])

    td = MsgPackData(data=train_data, transform=AUGMENTATIONS_TRAIN)
    vd = MsgPackData(data=val_data, transform=AUGMENTATIONS_VAL)

    trainloader = DataLoader(
        td,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    valloader = DataLoader(
        vd,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    print(
        f"Train dataset size {len(trainloader.dataset)}, validation dataset {len(valloader.dataset)}"
    )

    return trainloader, valloader


def get_testloader(config):
    test_dir = "europe/test"
    batch_size = config["batch_size"]
    image_size = config["image_size"]
    workers = config["workers"]
    pin_memory = config["pin_memory"]

    AUGMENTATIONS_TEST = Compose(
        [
            Resize(image_size, image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    # Create dataset for testing
    test_dataset = StreetViewDataset(test_dir, transform=AUGMENTATIONS_TEST)

    # Create dataloader for testing
    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    print(f"Test dataset size: {len(testloader.dataset)}")

    return testloader


def get_streetview_test(config, n=500):
    batch_size = config["batch_size"]
    image_size = config["image_size"]
    workers = config["workers"]
    pin_memory = config["pin_memory"]

    AUGMENTATIONS_TEST = Compose(
        [
            Resize(image_size, image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    test_folder = "test"
    all_data = StreetViewTestDataset(img_dir=test_folder, transform=AUGMENTATIONS_TEST)

    # Select random n elements
    all_data.img_fnames = random.sample(all_data.img_fnames, n)

    evalloader = DataLoader(
        all_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    print(f"Eval dataset size: {len(evalloader.dataset)}")

    return evalloader


def get_dataloaders(config):
    train_dir = "train"
    val_dir = "val"
    batch_size = config["batch_size"]
    image_size = config["image_size"]
    workers = config["workers"]
    pin_memory = config["pin_memory"]

    AUGMENTATIONS_TRAIN = Compose(
        [
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, p=0.5),
            HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.9
            ),
            CLAHE(p=1.0, clip_limit=2.0),
            ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.8,
            ),
            Resize(image_size, image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    AUGMENTATIONS_VAL = Compose(
        [
            Resize(image_size, image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    # Create datasets for training and validation
    train_dataset = StreetViewDataset(train_dir, transform=AUGMENTATIONS_TRAIN)
    val_dataset = StreetViewDataset(val_dir, transform=AUGMENTATIONS_VAL)

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    print(
        f"Train dataset size: {len(trainloader.dataset)}, validation dataset size: {len(valloader.dataset)}"
    )

    return trainloader, valloader


def get_europe(config):
    train_dir = "europe/train"
    val_dir = "europe/val"
    batch_size = config["batch_size"]
    image_size = config["image_size"]
    workers = config["workers"]
    pin_memory = config["pin_memory"]

    AUGMENTATIONS_TRAIN = Compose(
        [
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, p=0.5),
            HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.9
            ),
            CLAHE(p=1.0, clip_limit=2.0),
            ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.8,
            ),
            Resize(image_size, image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    AUGMENTATIONS_VAL = Compose(
        [
            Resize(image_size, image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    # Create datasets for training and validation
    train_dataset = StreetViewDataset(train_dir, transform=AUGMENTATIONS_TRAIN)
    val_dataset = StreetViewDataset(val_dir, transform=AUGMENTATIONS_VAL)

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    print(
        f"Train dataset size: {len(trainloader.dataset)}, validation dataset size: {len(valloader.dataset)}"
    )

    return trainloader, valloader


def get_shard(config):
    start_idx = config["start_idx"]
    num_shards = config["num_shards"]
    val_dir = config["val_dir"]
    batch_size = config["batch_size"]
    image_size = config["image_size"]
    workers = config["workers"]
    pin_memory = config["pin_memory"]

    AUGMENTATIONS_TRAIN = Compose(
        [
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, p=0.5),
            HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.9
            ),
            CLAHE(p=1.0, clip_limit=2.0),
            ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.8,
            ),
            Resize(image_size, image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    AUGMENTATIONS_VAL = Compose(
        [
            Resize(image_size, image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    all_shard_fnames = glob.glob(os.path.join(DIR, "*.msg"))
    selected_shard_fnames = all_shard_fnames[start_idx : start_idx + num_shards]

    all_data = load_data_from_shards(selected_shard_fnames)

    td = MsgPackDataWithCountries(data=all_data, transform=AUGMENTATIONS_TRAIN)
    vd = StreetViewDataset(val_dir, transform=AUGMENTATIONS_VAL)

    trainloader = DataLoader(
        td,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    valloader = DataLoader(
        vd,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    print(
        f"Train dataset size {len(trainloader.dataset)}, validation dataset {len(valloader.dataset)}"
    )

    return trainloader, valloader
