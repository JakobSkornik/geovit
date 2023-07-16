import numpy as np
import os
import torch
import torch.nn as nn


def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = np.transpose(image, (2, 0, 1))
    mean = np.array(mean)
    std = np.array(std)
    denormalized_img = image * std[:, None, None] + mean[:, None, None]
    denormalized_img = np.transpose(denormalized_img, (1, 2, 0))
    denormalized_img = np.clip(denormalized_img, 0, 1)
    return denormalized_img


# [-180, 180] & [-90, 90]  [0, 1]
def normalize_coordinates(coords):
    latitude, longitude = coords
    latitude = (latitude + 90) / 180
    longitude = (longitude + 180) / 360
    return latitude, longitude


def denormalize_coordinates(coords):
    latitude, longitude = coords
    latitude = latitude * 180 - 90
    longitude = longitude * 360 - 180
    return latitude, longitude


def save_model(model, optimizer, epoch, lr, model_version):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr": lr,
        },
        f"{model_version}.pth",
    )


def load_model(model, optimizer, path, mode="train"):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if mode == "train":
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Add the initial_lr to the optimizer's param_groups
        for param_group in optimizer.param_groups:
            param_group["initial_lr"] = checkpoint["lr"]

        start_epoch = checkpoint["epoch"]
        lr = checkpoint["lr"]
    else:
        start_epoch = checkpoint["epoch"]
        lr = None

    return model, optimizer, start_epoch, lr


def get_next_version(model_version):
    base_name = model_version + "_v"
    version = 1
    while os.path.isfile(base_name + str(version) + ".pth"):
        version += 1
    return base_name + str(version)


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
