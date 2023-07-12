import sys; sys.path.append("../")
import torch
import torch.nn as nn
import numpy as np
from utils import get_activation, get_normalization


class BasicCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, norm="batch", activation="relu", padding="same", residual=True):
        super(BasicCNNBlock, self).__init__()

        self.activation = get_activation(activation)
        self.residual = residual
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.match_channels = nn.Identity()
        if in_channels != out_channels:
            self.match_channels = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                get_normalization(norm, out_channels),
            )

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, padding=0)
        self.norm1 = get_normalization(norm, self.out_channels)

        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=self.padding, groups=self.out_channels)
        self.norm2 = get_normalization(norm, self.out_channels)
        
        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=self.padding, groups=1)
        self.norm3 = get_normalization(norm, self.out_channels)


    def forward(self, x):
        identity = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))

        if self.residual:
            x = x + self.match_channels(identity)

        x = self.activation(x)

        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_dims, cur_depth, *, norm="batch", activation="relu", padding="same"):
        super(AttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_dims = input_dims
        self.padding = padding
        self.cur_depth = cur_depth
        self.activation = get_activation(activation)

        self.attention_x = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.in_channels, self.input_dims, 1, padding=0),
            get_normalization(norm, self.input_dims),
            self.activation,
        )
        self.attention_y = nn.Sequential(
            nn.Conv2d(self.input_dims, self.input_dims, 1, padding=0, groups=self.input_dims),
            get_normalization(norm, self.input_dims),
            self.activation,
        )
        self.attention_y = BasicCNNBlock(self.input_dims, self.input_dims, norm=norm, activation=activation, padding=self.padding, residual=True)
        self.attention_out = BasicCNNBlock(self.input_dims, self.input_dims, norm=norm, activation=activation, padding=self.padding, residual=True)

        self.attention_collapse = nn.Conv2d(self.input_dims, 1, 3, padding=self.padding)
        self.tahn = nn.Tanh()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.fc_pool = nn.AdaptiveAvgPool2d(4)
        self.fc_norm1 = get_normalization(norm, self.input_dims)
        self.fc_norm2 = get_normalization(norm, self.input_dims, dims=1)
        self.linear1 = nn.Linear(in_features=self.input_dims * (4 * 4), out_features=self.input_dims)
        self.linear2 = nn.Linear(in_features=self.input_dims, out_features=self.input_dims)

        self.collapse_spatial = nn.Conv2d(self.cur_depth + 1, 1, 3, padding=self.padding)
        self.collapse_channel = nn.Conv2d(self.input_dims * (self.cur_depth + 1), self.input_dims, 1, padding=self.padding)

    def forward(self, x, skip, previous_attn): # x is the input, y is the original input resampled
        x = self.attention_x(x)
        skip = self.attention_y(skip)
        attn = x + skip

        attn_spatial = self.attention_collapse(attn)
        attn_spatial = self.tahn(attn_spatial)

        attn_channel = self.fc_pool(attn)
        attn_channel = attn_channel.reshape(attn_channel.size(0), -1)
        attn_channel = self.linear1(attn_channel)
        attn_channel = self.fc_norm2(attn_channel)
        attn_channel = self.activation(attn_channel)
        attn_channel = self.linear2(attn_channel)
        attn_channel = attn_channel.reshape(attn_channel.size(0), self.input_dims, 1, 1)
        attn_channel = self.tahn(attn_channel)

        # Consider when activations are being done.
        # Merge all before activations..

        if len(previous_attn) > 0:
            spatials = [attn_spatial]
            channels = [attn_channel]
            for i in range(len(previous_attn)):
                attn_spatial, attn_channel = previous_attn[i]
                _iter = 0

                # This should definitely not be a while loop.
                while attn_spatial.size(2) < x.size(2) and _iter < 10:
                    attn_spatial = self.upsample(attn_spatial)
                    _iter += 1

                spatials.append(attn_spatial)
                channels.append(attn_channel)

            attn_spatial = torch.cat(spatials, dim=1)
            attn_channel = torch.cat(channels, dim=1)
            
            attn_spatial = self.collapse_spatial(attn_spatial)
            attn_channel = self.collapse_channel(attn_channel)

            attn_spatial = self.tahn(attn_spatial)
            attn_channel = self.tahn(attn_channel)

        attn = self.attention_out(skip)
        attn = ((attn * attn_spatial) + (attn * attn_channel)) / 2.0
        attn = self.activation(attn)

        return attn, attn_spatial, attn_channel


class MixerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_dims, cur_depth, *, norm="batch", activation="relu"):
        super(MixerBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_dims = input_dims
        self.cur_depth = cur_depth
        self.activation = get_activation(activation)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.matcher = BasicCNNBlock(in_channels, out_channels - self.input_dims, norm=norm, activation=activation, padding="same", residual=True)
        self.mixer = BasicCNNBlock(out_channels, out_channels, norm=norm, activation=activation, padding="same", residual=True)

        self.raw_block = BasicCNNBlock(self.input_dims, self.input_dims, norm=norm, activation=activation, padding="same", residual=True)
        self.attn_block = AttentionBlock(self.in_channels, self.out_channels, self.input_dims, self.cur_depth, norm=norm, activation=activation, padding="same")

        self.norm1 = get_normalization(norm, in_channels)
        self.norm2 = get_normalization(norm, self.out_channels - self.input_dims)
        self.norm3 = get_normalization(norm, self.out_channels)

        self.stem = nn.Sequential(
            self.upsample,
            self.matcher,
            self.norm2,
            self.activation,
        )

    def forward(self,
        x,
        skip,
        previous_attn,
    ): # x is the input, y is the original input resampled
        identity = x
        attn, attn_spatial, attn_channel = self.attn_block(identity, skip, previous_attn)
        x = self.stem(x)
        x = torch.cat((x, attn), dim=1)
        x = self.mixer(x)

        return x, attn_spatial, attn_channel


class DecoderBlock(nn.Module):
    def __init__(self, depth, in_channels, out_channels, *, norm="batch", activation="relu", padding="same", residual=True):
        super(DecoderBlock, self).__init__()

        self.activation = get_activation(activation)
        self.residual = residual
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.blocks = nn.ModuleList()
        for i in range(depth):
            _in_channel = self.in_channels if i == 0 else self.out_channels
            block = BasicCNNBlock(_in_channel, out_channels, norm=norm, activation=activation, padding=padding, residual=residual)
            self.blocks.append(block)


    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x


class Pyramid(nn.Module):
    """ Pyramid Architecture"""
    def __init__(self, *, input_size=64, input_dim=10, output_dim=1, depths=None, dims=None, stem_squeeze=1, clamp_output=False, clamp_min=0.0, clamp_max=1.0, activation="relu", norm="batch", padding="same"):
        super(Pyramid, self).__init__()

        self.depths = np.array([3, 3, 3, 3]) if depths is None else np.array(depths)
        self.dims = np.array([32, 48, 64, 96]) if dims is None else np.array(dims)
        self.input_size = input_size
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.clamp_output = clamp_output
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.activation = activation
        self.norm = norm
        self.padding = padding
        self.stem_squeeze = stem_squeeze

        assert len(self.depths) == len(self.dims), "depths and dims must have the same length."

        self.sizes = [self.input_size // (2 ** (i + 1)) for i in reversed(range(len(self.depths)))]
        self.pools = [nn.AdaptiveAvgPool2d(i) for i in self.sizes]

        self.stem_size = (self.sizes[0] ** 2) * self.input_dim
        self.channel_match = BasicCNNBlock(self.input_dim, self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding, residual=True)
        self.activation = get_activation(self.activation)

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

        self.head = nn.Sequential(
            BasicCNNBlock(self.dims[-1], self.dims[-1], norm=self.norm, activation=self.activation, padding=self.padding, residual=True),
            nn.Conv2d(self.dims[-1], self.output_dim, kernel_size=1, padding=0, bias=False),
        )

        self.decoder_blocks = []
        self.mixer_blocks = []
        self.squeeze_blocks = []

        for i in range(len(self.depths)):
            self.mixer_blocks.append(
                MixerBlock(
                    self.dims[i] if i == 0 else self.dims[i - 1],
                    self.dims[i],
                    self.input_dim,
                    i,
                    norm=self.norm,
                    activation=self.activation,
                )
            )
            self.decoder_blocks.append(
                DecoderBlock(
                    self.depths[i],
                    self.dims[i],
                    self.dims[i],
                    norm=self.norm,
                    activation=self.activation,
                    padding=self.padding,
                    residual=True,
                )
            )

        self.mixer_blocks = nn.ModuleList(self.mixer_blocks)
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

    def initialize_weights(self, std=0.02):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        x = self.stem(identity)
        x = x.reshape(x.size(0), self.input_dim, self.sizes[0], self.sizes[0])
        x = self.channel_match(x)

        # Pre-calculate the previous attentions instead.

        previous_attn = []
        for i in range(len(self.depths)):
            pool_id = self.pools[i + 1](identity) if i < len(self.depths) - 1 else identity
            x, attn_spatial, attn_channel = self.mixer_blocks[i](x, pool_id, previous_attn)
            previous_attn.append((attn_spatial, attn_channel))
            x = self.decoder_blocks[i](x)

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

    model = Pyramid(
        input_size=64,
        input_dim=10,
        output_dim=1,
        depths=[3, 3, 3, 3],
        dims=[32, 48, 64, 96],
    )
    
    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )

# ====================================================================================================
# Total params: 2,414,564
# Trainable params: 2,414,564
# Non-trainable params: 0
# Total mult-adds (G): 79.20
# ====================================================================================================
# Input size (MB): 5.24
# Forward/backward pass size (MB): 4561.48
# Params size (MB): 4.04
# Estimated Total Size (MB): 4570.77
# ====================================================================================================
