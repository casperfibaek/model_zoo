# Standard Library
import os
from glob import glob

# External Libraries
import buteo as beo
import numpy as np
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Internal
from model_CNN_Basic import CNN_BasicEncoder

FOLDER = "./data/patches/"

# Hyperparameters
NUM_EPOCHS = 100
PATIENCE = 100
VAL_SPLIT = 0.1
BATCH_SIZE = 16
NUM_WORKERS = 0 # Increase if you are working on a large dataset. For small ones, multiple workers are slower.

# Cosine annealing scheduler with warm restarts
LEARNING_RATE = 0.001
T_0 = 15  # Number of iterations for the first restart
T_MULT = 2  # Multiply T_0 by this factor after each restart
ETA_MIN = 0.000001  # Minimum learning rate of the scheduler

def callback_normalise(x, y):
    """ Normalise the input data and sum the labels. """
    x_norm = np.empty_like(x, dtype=np.float32)
    np.divide(x, 10000.0, out=x_norm)

    y_summed = np.sum(y, axis=(1, 2)) / y.size

    return torch.from_numpy(x_norm), torch.from_numpy(y_summed)

x_train = beo.MultiArray([np.load(f, mmap_mode="r") for f in glob(os.path.join(FOLDER, "*train_s2.npy"))], shuffle=True)
y_train = beo.MultiArray([np.load(f, mmap_mode="r") for f in glob(os.path.join(FOLDER, "*train_label_area.npy"))])
y_train.set_shuffle_index(x_train.get_shuffle_index())
x_train, x_val = beo.split_multi_array(x_train, split_point=1-VAL_SPLIT)
y_train, y_val = beo.split_multi_array(y_train, split_point=1-VAL_SPLIT)

x_test = beo.MultiArray([np.load(f, mmap_mode="r") for f in glob(os.path.join(FOLDER, "*test_s2.npy"))])
y_test = beo.MultiArray([np.load(f, mmap_mode="r") for f in glob(os.path.join(FOLDER, "*test_label_area.npy"))])

assert len(x_train) == len(y_train) and len(x_test) == len(y_test) and len(x_val) == len(y_val), "Lengths of x and y do not match."

ds_train = beo.Dataset(x_train, y_train, input_is_channel_last=True, output_is_channel_last=False, callback=callback_normalise)
ds_test = beo.Dataset(x_test, y_test, input_is_channel_last=True, output_is_channel_last=False, callback=callback_normalise)
ds_val = beo.Dataset(x_val, y_val, input_is_channel_last=True, output_is_channel_last=False, callback=callback_normalise)

dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS, drop_last=True, generator=torch.Generator(device='cuda'))
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS, generator=torch.Generator(device='cuda'))
dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS, drop_last=True, generator=torch.Generator(device='cuda'))

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
else:
    print("No CUDA device available.")

if __name__ == "__main__":
    print("Starting training...")
    print("")

    model = CNN_BasicEncoder(output_dim=1)
    model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Save the initial learning rate in optimizer's param_groups
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = LEARNING_RATE

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0,
        T_MULT,
        ETA_MIN,
        last_epoch=NUM_EPOCHS - 1,
    )

    best_loss = None
    best_model_state = None
    epochs_no_improve = 0

    # Training loop
    for epoch in range(NUM_EPOCHS):

        # Initialize the running loss
        train_loss = 0.0

        # Initialize the progress bar for training
        train_pbar = tqdm(dl_train, total=len(dl_train), ncols=120, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for i, (images, labels) in enumerate(train_pbar):
            # Move inputs and targets to the device (GPU)
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update the scheduler
            scheduler.step()

            train_loss += loss.item()

            train_pbar.set_postfix({
                "loss": f"{train_loss / (i + 1):.4f}",
            })

            # Validate at the end of each epoch
            # This is done in the same scope to keep tqdm happy.
            if i == len(dl_train) - 1:

                # Validate every epoch
                with torch.no_grad():

                    val_loss = 0
                    for i, (images, labels) in enumerate(dl_val):
                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = model(images)

                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                # Append val_loss to the train_pbar
                train_pbar.set_postfix({
                    "loss": f"{train_loss / len(dl_train):.4f}",
                    "val_loss": f"{val_loss / len(dl_val):.4f}",
                }, refresh=True)

                if best_loss is None:
                    best_loss = val_loss
                elif best_loss > val_loss:
                    best_loss = val_loss
                    best_model_state = model.state_dict().copy()

                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve == PATIENCE:
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            break

    # Load the best weights
    model.load_state_dict(best_model_state)

    print("Finished Training")
    print("")

    # Test the model
    with torch.no_grad():
        test_loss = 0
        for i, (images, labels) in enumerate(dl_test):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

        print(f"Test Accuracy: {test_loss / (i + 1):.4f}")
