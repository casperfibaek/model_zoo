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


# Reference implementation: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
# https://github.com/huggingface/pytorch-image-models/tree/main/timm/layers
# Update Tensor truncator --> Official implementation should be on 2.0.1

# Check these implementation of ResNextUnet as well
# https://github.com/indzhykulianlab/bism/tree/main/bism/modules

"""
    TODO:
        - What does groups do?
        - Implement Incept ConvNext layers
        - Change the architecture to be image-to-image
            - That is, add a decoder framework.
            - Model seem to be based on channel_last -> should all be converted to channel_last?
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    LayerNorm,
    GRN,
)

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

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, *,
        input_dim=3,
        output_dim=1, 
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        head_init_scale=1.0,
        weight_init_mean=0.0,
        weight_init_std=0.02,
    ):
        super().__init__()

        self.initializer = torch.nn.init.trunc_normal_
        self.initializer_mean = weight_init_mean
        self.initializer_std = weight_init_std
        self.initializer_a = -weight_init_std * 2.0
        self.initializer_b = weight_init_std * 2.0

        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(input_dim, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], output_dim)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            self.initializer(m.weight, self.initializer_mean, self.initializer_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 6, 2], dims=[40, 160, 320], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 9, 3], dims=[96, 384, 768], **kwargs)
    return model


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
        encoder_only=True,
    )

    model = convnextv2_atto(input_dim=10, output_dim=1)

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
    NAME = "model_ResNextV2"

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
            out_path=F"../visualisations/pred_ResNextV2MSE_{epoch}.tif",
        )

    print(f"Summary for: {NAME}")
    summary(convnextv2_atto(input_dim=10, output_dim=1), input_size=(BATCH_SIZE, 10, 64, 64))

    train(
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        name=NAME,
        predict_func=predict_func,
    )