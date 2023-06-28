import torch
import torch.nn as nn
import torchmetrics

import sys; sys.path.append("../")
from utils import (
    load_data,
    training_loop,
    TiledMSE,
)




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.LazyConv2d(out_channels, 1, padding=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.relu = nn.ReLU()

        self.conv2 = nn.LazyConv2d(out_channels, 3, padding=1)
        self.bn2 = nn.LazyBatchNorm2d()
        
        self.shortcut = nn.Sequential()
        if out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.LazyConv2d(out_channels, kernel_size=1, stride=1, bias=False),
                nn.LazyBatchNorm2d()
            )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    """
    Basic U-Net

    Parameters
    ----------
    input_dim : int, optional
        The input dimension, default: 10

    output_dim : int, optional
        The output dimension, default: 1

    size_pow2 : int, optional
        The size of the first convolutional layer, default: 5
        Calculated as the power of 2. ie: 2 ** size_pow2
    
    Returns
    -------
    output : torch.Model
        The output model
    """
    def __init__(self, *, input_dim=10, output_dim=1, size_pow2=5):
        super(UNet, self).__init__()

        self.input_conv = nn.Conv2d(input_dim, 2 ** (size_pow2 + 0), kernel_size=7, padding=3)
        
        self.down_conv1 = ResidualBlock(2 ** (size_pow2 + 0), 2 ** (size_pow2 + 0))
        self.down_conv2 = ResidualBlock(2 ** (size_pow2 + 0), 2 ** (size_pow2 + 1))
        self.down_conv3 = ResidualBlock(2 ** (size_pow2 + 1), 2 ** (size_pow2 + 2))
        self.down_conv4 = ResidualBlock(2 ** (size_pow2 + 2), 2 ** (size_pow2 + 3))
        
        self.max_pool = nn.MaxPool2d(2)
        
        self.up_trans1 = nn.LazyConvTranspose2d(2 ** (size_pow2 + 2), kernel_size=2, stride=2)
        self.up_conv1 = ResidualBlock(2 ** (size_pow2 + 3), 2 ** (size_pow2 + 2))
        
        self.up_trans2 = nn.LazyConvTranspose2d(2 ** (size_pow2 + 1), kernel_size=2, stride=2)
        self.up_conv2 = ResidualBlock(2 ** (size_pow2 + 2), 2 ** (size_pow2 + 1))
        
        self.up_trans3 = nn.LazyConvTranspose2d(2 ** (size_pow2 + 0), kernel_size=2, stride=2)
        self.up_conv3 = ResidualBlock(2 ** (size_pow2 + 1), 2 ** (size_pow2 + 0))
        
        self.out = nn.LazyConv2d(output_dim, kernel_size=1)

    def forward(self, x):
        # Large input convolution for context
        conv0 = self.input_conv(x)

        # Downsampling
        conv1 = self.down_conv1(conv0)
        x = self.max_pool(conv1)

        conv2 = self.down_conv2(x)
        x = self.max_pool(conv2)

        conv3 = self.down_conv3(x)
        x = self.max_pool(conv3)
        
        x = self.down_conv4(x)
        
        # Upsamping
        x = self.up_trans1(x)
        x = torch.cat([x, conv3], dim=1)
        
        x = self.up_conv1(x)
        x = self.up_trans2(x)
        x = torch.cat([x, conv2], dim=1)
        
        x = self.up_conv2(x)
        x = self.up_trans3(x)
        x = torch.cat([x, conv1], dim=1)
        
        x = self.up_conv3(x)
        
        # output layer
        out = torch.clamp(self.out(x), 0.0, 100.0)

        return out

def train(
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    name=str,
    predict_func=None,
) -> str:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dl_train, dl_val, dl_test = load_data(
        x="s2",
        y="area",
        with_augmentations=True,
        num_workers=0,
        batch_size=batch_size,
        encoder_only=False,
    )

    model = UNet(input_dim=10, output_dim=1, size_pow2=4)

    wmape = torchmetrics.WeightedMeanAbsolutePercentageError(); wmape.__name__ = "wmape"
    mae = torchmetrics.MeanAbsoluteError(); mae.__name__ = "mae"
    mse = torchmetrics.MeanSquaredError(); mse.__name__ = "mse"

    training_loop(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        model=model,
        criterion=TiledMSE(bias=0.5),
        device=device,
        metrics=[
            mse.to(device),
            wmape.to(device),
            mae.to(device),
        ],
        train_loader=dl_train,
        val_loader=dl_val,
        test_loader=dl_test,
        name=name,
        predict_func=predict_func,
    )

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore", category=UserWarning)
    from torchinfo import summary
    import buteo as beo
    import numpy as np

    LEARNING_RATE = 0.001
    NUM_EPOCHS = 250
    BATCH_SIZE = 16
    NAME = "model_ResNetUnet"

    def predict_func(model, epoch):
        model.eval()
        model.to("cuda")
        
        img_path = "../data/images/naestved_s2.tif"
        img_arr = beo.raster_to_array(img_path, filled=True, fill_value=0, cast=np.float32) / 10000.0

        def predict(arr):
            swap = beo.channel_last_to_first(arr)
            as_torch = torch.from_numpy(swap).float()
            on_device = as_torch.to('cuda')
            predicted = model(on_device)
            on_cpu = predicted.cpu()
            as_numpy = on_cpu.detach().numpy()
            swap_back = beo.channel_first_to_last(as_numpy)

            return swap_back

        with torch.no_grad():
            predicted = beo.predict_array(
                img_arr,
                callback=predict,
                tile_size=64,
                n_offsets=3,
                batch_size=BATCH_SIZE,
            )
        beo.array_to_raster(
            predicted,
            reference=img_path,
            out_path=F"../visualisations/pred_MSE_{epoch}.tif",
        )

    print(f"Summary for: {NAME}")
    summary(UNet(output_dim=1), input_size=(BATCH_SIZE, 10, 64, 64))

    train(
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        name=NAME,
        predict_func=predict_func,
    )
