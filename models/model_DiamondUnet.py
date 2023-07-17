import sys; sys.path.append("../")
import torch
import torch.nn as nn
import numpy as np
from utils import get_activation, get_normalization



class CNNBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels, *,
        norm="batch",
        activation="relu",
        padding="same",
        residual=True,
    ):
        super(CNNBlock, self).__init__()

        self.activation = get_activation(activation)
        self.residual = residual
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.residual:
            if in_channels != out_channels:
                self.match_channels = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                    get_normalization(norm, out_channels),
                )

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, padding=0, bias=False)
        self.norm1 = get_normalization(norm, self.out_channels)

        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=self.padding, groups=self.out_channels, bias=False)
        self.norm2 = get_normalization(norm, self.out_channels)

        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=self.padding, groups=1, bias=False)
        self.norm3 = get_normalization(norm, self.out_channels)

    def forward(self, x):
        identity = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))

        if self.residual:
            if x.size(1) != identity.size(1):
                identity = self.match_channels(identity)

            x += identity

        x = self.activation(x)

        return x


class AttentionBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels, *,
        norm="batch",
        activation="relu",
        padding="same",
    ):
        super(AttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = get_activation(activation)
        self.norm = norm
        self.padding = padding
        self.expansion = 4
        self.reduction = 4

        self.compress = nn.Conv2d(self.out_channels, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.attn_c_pool = nn.AdaptiveAvgPool2d(self.reduction)
        self.attn_c_reduction = nn.Linear(self.out_channels * (self.reduction ** 2), self.out_channels * self.expansion)
        self.attn_c_extention = nn.Linear(self.out_channels * self.expansion, self.out_channels)

    def forward(self, x, skip):
        x = x + skip
        x = self.activation(x)

        attn_spatial = self.compress(x)
        attn_spatial = self.sigmoid(attn_spatial)

        attn_channel = self.attn_c_pool(x)
        attn_channel = attn_channel.reshape(attn_channel.size(0), -1)
        attn_channel = self.attn_c_reduction(attn_channel)
        attn_channel = self.activation(attn_channel)
        attn_channel = self.attn_c_extention(attn_channel)
        attn_channel = attn_channel.reshape(x.size(0), x.size(1), 1, 1)
        attn_channel = self.sigmoid(attn_channel)

        return attn_spatial, attn_channel


class DiamondUnet(nn.Module):
    def __init__(self, *, input_dim=10, output_dim=1, input_size=64, depths=None, dims=None, clamp_output=False, clamp_min=0.0, clamp_max=1.0, activation="relu", norm="batch", padding="same"):
        super(DiamondUnet, self).__init__()

        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.dims = [96, 192, 384, 768] if dims is None else dims
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.input_size = input_size
        self.clamp_output = clamp_output
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.activation = get_activation(activation)
        self.norm = norm
        self.padding = padding
        self.dense_bridge = True

        self.dims = [v // 2 for v in self.dims]
        self.sizes = [self.input_size // (2 ** (i + 1)) for i in reversed(range(len(self.depths)))]

        assert len(self.depths) == len(self.dims), "depths and dims must have the same length."
        
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.sigmoid = nn.Sigmoid()

        self.stem = nn.Sequential(
            CNNBlock(self.input_dim, self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding),
        )

        self.bridge = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.dims[-1] * (self.sizes[0] ** 2), (self.dims[-1] * (self.sizes[0] ** 2)) // 2, bias=False),
            nn.BatchNorm1d((self.dims[-1] * (self.sizes[0] ** 2)) // 2),
            self.activation,
            nn.Linear((self.dims[-1] * (self.sizes[0] ** 2)) // 2, self.dims[-1] * (self.sizes[0] ** 2), bias=False),
            nn.BatchNorm1d((self.dims[-1] * (self.sizes[0] ** 2))),
            self.activation,
        )
        
        # Model output
        self.head = nn.Sequential(
            CNNBlock(self.dims[0], self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding),
            nn.Conv2d(self.dims[0], self.output_dim, 1),
        )

        self.attention_blocks = []
        self.encoder_blocks = []
        self.decoder_blocks = []
        self.match_blocks = []
        for idx_i in range(len(self.dims)):
            _encoder_blocks = []
            _decoder_blocks = []
            for idx_j in range(self.depths[idx_i]):

                # Calculate input and output dimensions for each block
                # If we use lazy modules, this goes away.
                indims_encoder = self.dims[idx_i - 1 if idx_i > 0 and idx_j == 0 else idx_i]
                indims_decoder = self.dims[::-1][idx_i + 1 if idx_j > 0 and idx_i < len(self.dims) - 1 else idx_i] 
                outdims_encoder = self.dims[idx_i]
                outdims_decoder = self.dims[::-1][idx_i + 1 if idx_i < len(self.dims) - 1 else idx_i]

                _encoder_blocks.append(
                    CNNBlock(indims_encoder, outdims_encoder, norm=self.norm, activation=self.activation, padding=self.padding)
                )
                _decoder_blocks.append(
                    CNNBlock(indims_decoder, outdims_decoder, norm=self.norm, activation=self.activation, padding=self.padding)
                )
            self.encoder_blocks.append(nn.Sequential(*_encoder_blocks))
            self.decoder_blocks.append(nn.Sequential(*_decoder_blocks))
            self.attention_blocks.append(AttentionBlock(self.dims[idx_i], self.dims[idx_i], norm=self.norm, activation=self.activation, padding=self.padding))

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)
        self.attention_blocks = nn.ModuleList(self.attention_blocks)


    def forward(self, x):
        x = self.stem(x)

        # Encoder
        skip_connections = []
        for idx, block in enumerate(self.encoder_blocks):
            x = block(x)
            skip_connections.append(x)
            x = self.downsample(x)

        # Bridge (Deepest point of the model)
        bridge = self.bridge(x)
        x = bridge.reshape(x.size())
        x = self.upsample(x)

        # Decoder with skip connections and self-attention 
        for idx, block in enumerate(self.decoder_blocks):
            skip = skip_connections.pop()
            attn_spatial, attn_channel = self.attention_blocks[::-1][idx](x, skip)

            if idx > 0:
                x = x + (((skip * attn_spatial) + (skip * attn_channel)) / 2)

            x = block(x)

            if idx < len(self.dims) - 1:
                x = self.upsample(x)

        # Head (The model output)
        x = self.head(x)
        if self.clamp_output:
            x = torch.clamp(x, self.clamp_min, self.clamp_max)

        return x



if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 32
    CHANNELS = 10
    HEIGHT = 64
    WIDTH = 64

    model = DiamondUnet(
        input_dim=10,
        output_dim=1,
        input_size=64,
        depths=[3, 3, 3, 3],
        dims=[96, 192, 384, 768],
    )
    
    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )
