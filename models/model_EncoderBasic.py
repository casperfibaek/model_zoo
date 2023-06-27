import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

import sys; sys.path.append("../")
from utils import load_data, training_loop

class CNN_BasicEncoder(nn.Module):
    """
    Basic CNN encoder for the MNIST dataset

    Parameters
    ----------
    output_dim : int, optional
        The output dimension, default: 1
    
    Returns
    -------
    output : torch.Model
        The output model
    """
    def __init__(self, *, output_dim=1, input_dim=10):
        super(CNN_BasicEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, 3, 1, 1)
        self.bn1 = nn.LazyBatchNorm2d(32)
        self.max_pool2d1 = nn.MaxPool2d(2)
        self.conv2 = nn.LazyConv2d(64, 3, 1, 1)
        self.bn2 = nn.LazyBatchNorm2d(64)
        self.max_pool2d2 = nn.MaxPool2d(2)
        self.fc1 = nn.LazyLinear(128)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.LazyLinear(output_dim)

    def forward(self, x):
        # First layer: Conv -> Norm -> Activation -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.max_pool2d1(x)

        # Second Layer: Conv -> Norm -> Activation -> Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.max_pool2d2(x)

        # Flatten the layers and pass through the fully connected layers
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x) # Dropout to prevent overfitting

        output = self.fc2(x)

        return output


def train(
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    name=str,
) -> str:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dl_train, dl_val, dl_test = load_data(
        x="s2",
        y="area",
        with_augmentations=True,
        num_workers=0,
        batch_size=batch_size,
        encoder_only=True,
    )

    model = CNN_BasicEncoder(output_dim=1, input_dim=10)

    wmape = torchmetrics.WeightedMeanAbsolutePercentageError(); wmape.__name__ = "wmape"
    mae = torchmetrics.MeanAbsoluteError(); mae.__name__ = "mae"

    training_loop(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        model=model,
        criterion=nn.MSELoss(),
        device=device,
        metrics=[
            wmape.to(device),
            mae.to(device),
        ],
        train_loader=dl_train,
        val_loader=dl_val,
        test_loader=dl_test,
        name=name,
    )

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore", category=UserWarning)
    from torchinfo import summary

    LEARNING_RATE = 0.001
    NUM_EPOCHS = 250
    BATCH_SIZE = 16
    NAME = "model_CNN_Basic"

    print(f"Summary for: {NAME}")
    summary(CNN_BasicEncoder(output_dim=1), input_size=(BATCH_SIZE, 10, 64, 64))

    train(
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        name=NAME,
    )
