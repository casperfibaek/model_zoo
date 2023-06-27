import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

import sys; sys.path.append("../")
from utils import load_data, training_loop

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, *, in_filters, filters, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.LazyConv2d(filters, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.LazyBatchNorm2d()
        self.conv2 = nn.LazyConv2d(filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.LazyBatchNorm2d()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_filters != (self.expansion * filters):
            self.shortcut = nn.Sequential(
                nn.LazyConv2d(self.expansion * filters, kernel_size=1, stride=stride, bias=False),
                nn.LazyBatchNorm2d(self.expansion * filters)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicResNetEncoder(nn.Module):
    def __init__(self, block, num_blocks, output_dim=1, input_dim=10):
        super(BasicResNetEncoder, self).__init__()
        self.in_filters = 64

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.LazyBatchNorm2d()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.fc1 = nn.LazyLinear(128)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.LazyLinear(output_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_filters, planes, stride))
            self.in_filters = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1) # Same as flatten
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        return out



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

    model = BasicResNetEncoder(BasicBlock, [2, 2], input_dim=10, output_dim=1)

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
    NAME = "model_ResNet18"

    print(f"Summary for: {NAME}")
    summary(BasicResNetEncoder(BasicBlock, [2, 2], input_dim=10, output_dim=1), input_size=(BATCH_SIZE, 10, 64, 64))
    
    train(
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        name=NAME,
    )
