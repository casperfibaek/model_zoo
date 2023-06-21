import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, output_dim=1):
        super(CNN_BasicEncoder, self).__init__()
        self.conv1 = nn.LazyConv2d(32, 3, 1, 1)
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
