import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import buteo as beo
import numpy as np

from load_data import load_data
from model_CNN_Basic import CNN_BasicEncoder
from utils import CustomTensorDataset

y_train, x_train_s2, _x_train_s1 = load_data(tile_size=64, overlaps=3)

# Normalise s2 data
x_train_s2 = (x_train_s2 / 10000.0).astype(np.float32)

x_train, x_val, x_test, y_train, y_val, y_test = beo.split_train_val_test(x_train_s2, y_train, random_state=42)

# Convert labels to single values. We are only interested in the People per tile.
y_train = y_train[:, 1:2, :, :].sum(axis=(2, 3))
y_val = y_val[:, 1:2, :, :].sum(axis=(2, 3))
y_test = y_test[:, 1:2, :, :].sum(axis=(2, 3))

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
else:
    print("No CUDA device available.")

# Hyperparameters
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
BATCH_SIZE = 16

model = CNN_BasicEncoder(output_dim=1)
model.to(device)
# model = torch.compile(model)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Datasets
ds_train = CustomTensorDataset((torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()))
ds_test = CustomTensorDataset((torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()))
ds_val = CustomTensorDataset((torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float()))

# Data loaders
train_loader = DataLoader(dataset=ds_train, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device='cuda'))
test_loader = DataLoader(dataset=ds_test, batch_size=BATCH_SIZE, shuffle=False, generator=torch.Generator(device='cuda'))
val_loader = DataLoader(dataset=ds_val, batch_size=BATCH_SIZE, shuffle=False, generator=torch.Generator(device='cuda'))

# Training loop
for epoch in range(NUM_EPOCHS):

    # Initialize the running loss
    train_loss = 0.0

    # Initialize the progress bar for training
    train_pbar = tqdm(train_loader, total=len(train_loader), ncols=120)

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

        train_loss += loss.item()

        train_pbar.set_postfix({
            "loss": f"{loss.item():.4f}, mean_loss, {train_loss / (i + 1):.4f}",
        })

    # Validate every epoch
    with torch.no_grad():

        val_loss = 0
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

        train_pbar.set_postfix({
            "loss": f"{loss.item():.4f}, mean_loss, {train_loss / (i + 1):.4f}, {val_loss / (i + 1):.4f}",
        })

print("Finished Training")
print("")

# Test the model
with torch.no_grad():
    test_loss = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)
        test_loss += loss.item()

    print(f"Test Accuracy: {test_loss / (i + 1):.4f}")
