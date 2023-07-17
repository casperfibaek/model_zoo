# Standard Library
import os
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import wandb

from .training_utils import cosine_scheduler, convert_torch_to_float


def training_loop(
    num_epochs: int,
    learning_rate: float,
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    metrics: list = None,
    name="model",
    project="model_zoo",
    out_folder="../trained_models/",
    patience=20,
    learning_rate_end=0.00001,
    weight_decay=0.05,
    weight_decay_end=0.0001,
    warmup_epochs=10,
    warmup_lr_start=0.0000001,
    min_epochs=50,
    use_wandb=True,
    save_best_model=False,
    predict_func=None,
) -> None:
    torch.set_default_device(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        print("No CUDA device available.")

    if warmup_epochs > 0:
        print(f"Starting warmup for {warmup_epochs} epochs...")
    else:
        print("Starting training...")

    if use_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=project,
            
            # track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "architecture": name,
                "dataset": "DK_Structure_small",
                "epochs": num_epochs + warmup_epochs,
            }
        )

    model.to(device)

    if weight_decay_end is None:
        weight_decay_end = weight_decay

    wd_schedule_values = cosine_scheduler(
        weight_decay, weight_decay_end, num_epochs + warmup_epochs, warmup_epochs, weight_decay_end,
    )

    lr_schedule_values = cosine_scheduler(
        learning_rate, learning_rate_end, num_epochs + warmup_epochs, warmup_epochs, warmup_lr_start,
    )

    # Loss and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), eps=1e-7)
    scaler = GradScaler()

    # Save the initial learning rate in optimizer's param_groups
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = lr_schedule_values[0]
        param_group['weight_decay'] = wd_schedule_values[0]

    best_epoch = 0
    best_loss = None
    best_model_state = model.state_dict().copy()
    epochs_no_improve = 0

    # Training loop
    for epoch in range(num_epochs + warmup_epochs):
        if epoch == warmup_epochs and warmup_epochs > 0:
            print("Finished warmup. Starting training...")

        model.train()

        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = wd_schedule_values[epoch]
            param_group['lr'] = lr_schedule_values[epoch]

        # Initialize the running loss
        train_loss = 0.0
        train_metrics_values = { metric.__name__: 0.0 for metric in metrics }

        # Initialize the progress bar for training
        epoch_current = epoch + 1 if epoch < warmup_epochs else epoch + 1 - warmup_epochs
        epoch_max = num_epochs if epoch >= warmup_epochs else warmup_epochs

        train_pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch_current}/{epoch_max}")

        for i, (images, labels) in enumerate(train_pbar):
            # Move inputs and targets to the device (GPU)
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Cast to bfloat16
            with autocast(dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            train_loss += loss.item()

            for metric in metrics:
                train_metrics_values[metric.__name__] += metric(outputs, labels)

            train_pbar.set_postfix({
                "loss": f"{train_loss / (i + 1):.4f}",
                **{name: f"{value / (i + 1):.4f}" for name, value in train_metrics_values.items()}
            })

            # Validate at the end of each epoch
            # This is done in the same scope to keep tqdm happy.
            if i == len(train_loader) - 1:

                val_metrics_values = { metric.__name__: 0.0 for metric in metrics }
                # Validate every epoch
                with torch.no_grad():
                    model.eval()

                    val_loss = 0
                    for j, (images, labels) in enumerate(val_loader):
                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = model(images)

                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                        for metric in metrics:
                            val_metrics_values[metric.__name__] += metric(outputs, labels)

                # Append val_loss to the train_pbar
                loss_dict = {
                    "loss": train_loss / (i + 1),
                    **{name: value / (i + 1) for name, value in train_metrics_values.items()},
                    "val_loss": val_loss / (j + 1),
                    **{f"val_{name}": value / (j + 1) for name, value in val_metrics_values.items()},
                }
                loss_dict = { key: convert_torch_to_float(value) for key, value in loss_dict.items() }
                loss_dict_str = { key: f"{value:.4f}" for key, value in loss_dict.items() }

                train_pbar.set_postfix(loss_dict_str, refresh=True)

                if use_wandb:
                    wandb.log(loss_dict)

                if best_loss is None:
                    best_epoch = epoch_current
                    best_loss = val_loss
                    best_model_state = model.state_dict().copy()

                    if predict_func is not None and epoch >= warmup_epochs:
                        predict_func(model, epoch_current)

                elif best_loss > val_loss:
                    best_epoch = epoch_current
                    best_loss = val_loss
                    best_model_state = model.state_dict().copy()

                    if predict_func is not None and epoch >= warmup_epochs:
                        predict_func(model, epoch_current)

                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve == patience and epoch >= warmup_epochs and epoch_current >= min_epochs + patience:
            print(f'Early stopping triggered after {epoch_current} epochs.')
            break

    # Load the best weights
    model.load_state_dict(best_model_state)

    print("Finished Training. Best epoch: ", best_epoch)
    print("")
    print("Starting Testing... (Best val epoch).")
    model.eval()

    # Test the model
    with torch.no_grad():
        test_loss = 0
        for k, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

        print(f"Test Accuracy: {test_loss / (k + 1):.4f}")

    # Save the model
    if save_best_model:
        torch.save(best_model_state, os.path.join(out_folder, f"{name}.pt"))
