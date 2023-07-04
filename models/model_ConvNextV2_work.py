import torch
import torch.nn as nn
import torchmetrics

import sys; sys.path.append("../")
from utils import (
    load_data,
    training_loop,
    TiledMSE,
    LayerNorm,
    GRN,
)


"""
    TODO:
        - Iteratively building blocks (dims, depth, kernel_size_at_depth)
        - Add weight initialization
        - Add weight decay
        - Change LayerNorm to BaseImplementation?
        - Look into SENets, InceptionConvNextV2
        - Look into baseimplementations
"""


class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers <- why?
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)


    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + x

        return x


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
        self.size_pow2 = size_pow2

        self.input_conv = nn.Conv2d(input_dim, 2 ** (size_pow2 + 0), kernel_size=7, padding=3)
        
        self.down_conv1 = Block(2 ** (size_pow2 + 0))
        self.down_conv2 = Block(2 ** (size_pow2 + 1))
        self.down_conv3 = Block(2 ** (size_pow2 + 2))
        
        self.bottle1 = Block(2 ** (size_pow2 + 3))
        
        self.down_sample1 = nn.LazyConv2d(2 ** (size_pow2 + 1), kernel_size=2, stride=2)
        self.down_sample2 = nn.LazyConv2d(2 ** (size_pow2 + 2), kernel_size=2, stride=2)
        self.down_sample3 = nn.LazyConv2d(2 ** (size_pow2 + 3), kernel_size=2, stride=2)
        
        self.up_trans1 = nn.LazyConvTranspose2d(2 ** (size_pow2 + 2), kernel_size=2, stride=2)
        self.up_conv1 = Block(2 ** (size_pow2 + 3))
        
        self.up_trans2 = nn.LazyConvTranspose2d(2 ** (size_pow2 + 1), kernel_size=2, stride=2)
        self.up_conv2 = Block(2 ** (size_pow2 + 2))
        
        self.up_trans3 = nn.LazyConvTranspose2d(2 ** (size_pow2 + 0), kernel_size=2, stride=2)
        self.up_conv3 = Block(2 ** (size_pow2 + 1))
        
        self.out = nn.LazyConv2d(output_dim, kernel_size=1)

    def forward(self, x):
        # Large input convolution for context
        conv0 = self.input_conv(x)

        # Downsampling
        conv1 = self.down_conv1(conv0)
        x = self.down_sample1(conv1)

        conv2 = self.down_conv2(x)
        x = self.down_sample2(conv2)

        conv3 = self.down_conv3(x)
        x = self.down_sample3(conv3)

        # Bottleneck
        x = self.bottle1(x)
        
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

    model = UNet(input_dim=10, output_dim=1, size_pow2=6)

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
    NAME = "model_ResNextUnetBase"

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
            out_path=F"../visualisations/pred_ResNextUnetBase_{epoch}.tif",
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