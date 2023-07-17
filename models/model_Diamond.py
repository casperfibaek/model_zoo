import sys; sys.path.append("../")
import torch
import torch.nn as nn
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

        self.match = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0, bias=False),
            get_normalization(self.norm, self.out_channels),
            self.activation,
        )
        self.compress = nn.Conv2d(self.out_channels * 2, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.attn_c_pool = nn.AdaptiveAvgPool2d(self.reduction)
        self.attn_c_reduction = nn.Linear(self.out_channels * (self.reduction ** 2) * 2, self.out_channels * self.expansion)
        self.attn_c_extention = nn.Linear(self.out_channels * self.expansion, self.out_channels)

    def forward(self, x, skip):
        x = self.match(x)
        x = torch.cat([x, skip], dim=1)
        x = self.activation(x)

        attn_spatial = self.compress(x)
        attn_spatial = self.sigmoid(attn_spatial)

        attn_channel = self.attn_c_pool(x)
        attn_channel = attn_channel.reshape(attn_channel.size(0), -1)
        attn_channel = self.attn_c_reduction(attn_channel)
        attn_channel = self.activation(attn_channel)
        attn_channel = self.attn_c_extention(attn_channel)
        attn_channel = attn_channel.reshape(x.size(0), x.size(1) // 2, 1, 1)
        attn_channel = self.sigmoid(attn_channel)

        return attn_spatial, attn_channel

class DiamondNet(nn.Module):
    def __init__(self, *,
        input_dim=10,
        output_dim=1,
        input_size=64,
        depths=None,
        dims=None,
        clamp_output=False,
        clamp_min=0.0,
        clamp_max=1.0,
        activation="relu",
        norm="batch",
        padding="same",
    ):
        super(DiamondNet, self).__init__()

        self.depths = [3, 3, 3, 3] if depths is None else depths
        self.dims = [32, 48, 64, 80] if dims is None else dims
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.clamp_output = clamp_output
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.activation = get_activation(activation)
        self.norm = norm
        self.padding = padding
        self.stem_squeeze = 1
        self.input_size = input_size

        self.sizes = [self.input_size // (2 ** (i + 1)) for i in reversed(range(len(self.depths)))]
        self.stem_size = (self.sizes[0] ** 2) * self.input_dim
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # Instead of using average downsampling, we use strided convolution to make the downsamping
        self.pools = []
        for i in self.sizes:
            self.pools.append(
                nn.Conv2d(
                    self.input_dim,
                    self.input_dim,
                    kernel_size=self.input_size // i + 1,
                    stride=self.input_size // i,
                    padding=(self.input_size // i) // 2,
                    groups=self.input_dim
                )
            )
        self.pools.append(
            nn.Conv2d(
                self.input_dim,
                self.input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.pools = nn.ModuleList(self.pools)

        assert len(self.depths) == len(self.dims), "depths and dims must have the same length."

        # Fully connected stem
        self.stem = nn.Sequential(
            self.pools[0],
            nn.Flatten(),
            nn.Linear(self.stem_size, self.stem_size // self.stem_squeeze, bias=False),
            nn.BatchNorm1d(self.stem_size // self.stem_squeeze),
            self.activation,
            nn.Linear(self.stem_size // self.stem_squeeze, self.stem_size, bias=False),
            nn.BatchNorm1d(self.stem_size),
            self.activation,
        )
        self.stem_match = CNNBlock(self.input_dim, self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding, residual=True)

        # These are the residual blocks (4xdepth) and attention (4x1)
        self.blocks = []
        self.blocks_attn = []
        for i in range(len(self.depths)):
            _blocks = []
            for j in range(self.depths[i]):
                indims = self.dims[i]

                # Figure out what size the input will be and block edges
                if i == 0 and j == 0:
                    indims = self.dims[i] + self.input_dim
                elif i > 0 and j == 0:
                    indims = self.dims[i - 1] + self.input_dim

                block = CNNBlock(
                        indims,
                        self.dims[i],
                        norm=self.norm,
                        activation=self.activation,
                        padding=self.padding,
                    )
                _blocks.append(block)
            self.blocks.append(nn.Sequential(*_blocks))

            block_attn = AttentionBlock(
                self.dims[i - (1 if i > 0 else 0)],
                self.input_dim,
                norm=self.norm,
                activation=self.activation,
                padding=self.padding,
            )
            self.blocks_attn.append(block_attn)

        self.blocks = nn.ModuleList(self.blocks)
        self.blocks_attn = nn.ModuleList(self.blocks_attn)

        # Merge the attention layers from different scales as we go deeper
        self.attention_spatial_merge = []
        self.attention_channel_merge = []
        for i in range(1, len(self.depths)):
            self.attention_spatial_merge.append( 
                nn.Sequential(nn.Conv2d(i + 1, 1, 1), self.sigmoid)
            )
            self.attention_channel_merge.append(
                nn.Sequential(nn.Conv2d(self.input_dim * (i + 1), input_dim, 1), self.sigmoid)
            )
        self.attention_spatial_merge = nn.ModuleList(self.attention_spatial_merge)
        self.attention_channel_merge = nn.ModuleList(self.attention_channel_merge)

        # Model output
        self.head = nn.Sequential(
            CNNBlock(self.dims[-1], self.dims[-1], norm=self.norm, activation=self.activation, padding=self.padding),
            nn.Conv2d(self.dims[-1], self.output_dim, 1),
        )

    def forward(self, x):
        identity = x
        x = self.stem(identity) # <-- Here you can inject 1d features into the model
        x = x.reshape(x.size(0), self.input_dim, self.sizes[0], self.sizes[0])
        x = self.stem_match(x)

        attn_p_spatial = []
        attn_p_channel = []
        for idx, block in enumerate(self.blocks):
            pool = self.pools[idx + 1](identity)
            skip = self.upsample(x)

            attn_spatial, attn_channel = self.blocks_attn[idx](skip, pool)
            attn_p_spatial = list(map(lambda y: self.upsample(y), attn_p_spatial))
            attn_p_spatial.append(attn_spatial); attn_p_channel.append(attn_channel)

            if idx > 0:
                attn_spatial = torch.cat(attn_p_spatial, dim=1)
                attn_spatial = self.attention_spatial_merge[idx - 1](attn_spatial)

                attn_channel = torch.cat(attn_p_channel, dim=1)
                attn_channel = self.attention_channel_merge[idx - 1](attn_channel)

            x = torch.cat([skip, (pool * attn_spatial) + (pool * attn_channel)], dim=1)
            x = block(x)

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

    model = DiamondNet(
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
