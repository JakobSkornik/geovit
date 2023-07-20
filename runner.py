import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_loader import get_dataloaders
from model import ExtendedGeoViT
from utils import save_model


def train_one_epoch(
    model,
    trainloader,
    device,
    optimizer,
    criterion,
    pbar,
    grad_accum_steps=8,
):
    scaler = GradScaler()

    running_loss = 0.0
    total_samples = 0

    for param in model.parameters():
        param.grad = None

    for i, data in enumerate(trainloader, 0):
        inputs, coordinates = data["image"].to(device), data["coordinates"].to(device)
        batch_size = inputs.shape[0]
        total_samples += batch_size

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, coordinates)
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (i + 1) % grad_accum_steps == 0 or i + 1 == len(trainloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            for param in model.parameters():
                param.grad = None

        running_loss += loss.item() * batch_size  # Multiply by batch size

        # Update the progress bar
        pbar.set_postfix(
            **{
                "train_loss": running_loss / total_samples,  # Divide by total samples
                "lr": optimizer.param_groups[-1]["lr"],
            }
        )
        pbar.update(batch_size)

    return running_loss / total_samples


def main():
    # Dataloader config
    config = {
        # Dataloader config
        "start_idx": 1,
        "num_shards": 10,
        "batch_size": 32,
        "workers": 3,
        "pin_memory": True,
        "image_size": 224,
        # Training config
        "device": "cuda",
        "num_epochs": 30,
        "initial_lr": 5e-5,
        "pretrained_lr": 5e-6,
        "model_name": f"dev",
        "save": True,
    }

    train_dataloader, val_dataloader = get_dataloaders(config)

    checkpoints = config["save"]

    model = ExtendedGeoViT()
    device = config["device"]
    model.to(device)
    print(f"Using device: {device}")

    # Set up the loss function
    criterion = nn.MSELoss()

    # Define two groups of parameters: one for the ViT base without the classifier and one for the classifier
    no_decay = ["bias", "LayerNorm.weight"]  # Parameters not to apply weight decay on

    vit_parameters = model.vit.named_parameters()
    added_block_parameters = model.additional_blocks.parameters()
    final_classifier_parameters = model.final_classifier.parameters()

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in vit_parameters if not any(nd in n for nd in no_decay)
            ],
            "lr": config["pretrained_lr"],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in vit_parameters if any(nd in n for nd in no_decay)],
            "lr": config["pretrained_lr"],
            "weight_decay": 0.0,
        },
        {
            "params": added_block_parameters,
            "lr": config["initial_lr"],
            "weight_decay": 0.0,
        },
        {
            "params": final_classifier_parameters,
            "lr": config["initial_lr"],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters)

    if checkpoints:
        model_name = config["model_name"]

    # Define the learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

    def validation_loss(valloader):
        total_val_loss = 0
        with torch.no_grad():
            for data in valloader:
                images, labels = data["image"].to(device), data["coordinates"].to(
                    device
                )
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        return total_val_loss / len(valloader)

    # Train the model
    for epoch in range(config["num_epochs"]):
        with tqdm(
            total=len(train_dataloader.dataset),
            desc=f"""Epoch {epoch}/{config["num_epochs"]}""",
            unit="img",
        ) as pbar:
            train_loss = train_one_epoch(
                model, train_dataloader, device, optimizer, criterion, pbar
            )
            val_loss = validation_loss(val_dataloader)
            lr_pretrained = optimizer.param_groups[0]["lr"]
            lr = optimizer.param_groups[-1]["lr"]
            pbar.set_postfix(
                **{"train_loss": train_loss, "val_loss": val_loss, "lr_pretrained": lr_pretrained, "lr": lr}
            )
            pbar.update()
        scheduler.step()

        if checkpoints:
            save_model(model, optimizer, epoch, lr, model_name)

    print("Finished training")
    model.eval()


if __name__ == "__main__":
    main()
