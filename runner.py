import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data_loader import get_dataloaders
from model import GeoViT
from utils import save_model


def train_one_epoch(
    model,
    trainloader,
    device,
    optimizer,
    criterion,
    scheduler,
    pbar,
    grad_accum_steps=4,  # Number of steps to accumulate gradients
):
    scaler = GradScaler()  # Initialize GradScaler for mixed precision training

    running_loss = 0.0
    total_samples = 0  # Keep track of total number of samples

    model.zero_grad()  # Initialize gradients to zero before each backward pass

    for i, data in enumerate(trainloader, 0):
        inputs, coordinates = data["image"].to(device), data["coordinates"].to(device)
        batch_size = inputs.shape[0]  # Get batch size
        total_samples += batch_size  # Add to total

        # Forward pass, backward pass
        with autocast():  # Enable autocasting for mixed precision training
            outputs = model(inputs)
            loss = criterion(outputs, coordinates)

        # Scales the loss, and calls backward() to create scaled gradients
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (i + 1) % grad_accum_steps == 0 or i + 1 == len(trainloader):
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Unscales gradients and calls or skips optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
            # Zero the gradients after updating
            optimizer.zero_grad()

        running_loss += loss.item() * batch_size  # Multiply by batch size

        # Update the progress bar
        pbar.set_postfix(
            **{
                "train_loss": running_loss / total_samples,  # Divide by total samples
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        pbar.update(batch_size)

    # Step the learning rate scheduler
    scheduler.step()

    return running_loss / total_samples


def main():
    # Dataloader config
    config = {
        # Dataloader config
        "start_idx": 0,
        "num_shards": 10,
        "batch_size": 16,
        "workers": 8,
        "image_size": 224,
        # Training config
        "device": "cuda",
        "num_epochs": 10,
        "initial_lr": 1e-5,
        "model_name": f"dev",
    }

    train_dataloader, val_dataloader = get_dataloaders(config)

    checkpoints = True

    model = GeoViT()
    device = config["device"]
    model.to(device)
    print(f"Using device: {device}")

    # Set up the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["initial_lr"])

    if checkpoints:
        model_name = config["model_name"]

    # Define the learning rate scheduler
    scheduler = scheduler = optim.lr_scheduler.StepLR(
        optimizer, gamma=0.97, step_size=1
    )

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
                model, train_dataloader, device, optimizer, criterion, scheduler, pbar
            )
            val_loss = validation_loss(val_dataloader)
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                **{"train_loss": train_loss, "val_loss": val_loss, "lr": lr}
            )
            pbar.update()

        if checkpoints:
            save_model(model, optimizer, epoch, lr, model_name)

    print("Finished training")
    model.eval()


if __name__ == "__main__":
    main()
