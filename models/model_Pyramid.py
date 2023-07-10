import sys; sys.path.append("../")
import torch
import torch.nn as nn
from utils import get_activation, get_normalization, SE_BlockV3


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
                get_normalization(norm, out_channels), # Has to be different, two learn two different scalars
            )

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, padding=0)
        self.norm1 = get_normalization(norm, self.out_channels)

        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=self.padding, groups=1)
        self.norm2 = get_normalization(norm, self.out_channels)
        
        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=self.padding, groups=self.out_channels)
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



class MixerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_dims, *, norm="batch", activation="relu"):
        super(MixerBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.matcher = BasicCNNBlock(in_channels, out_channels - input_dims, norm=norm, activation=activation, padding="same", residual=True)
        self.raw_block = BasicCNNBlock(input_dims, input_dims, norm=norm, activation=activation, padding="same", residual=True)
        self.mixer = BasicCNNBlock(out_channels, out_channels, norm=norm, activation=activation, padding="same", residual=True)

        self.activation = get_activation(activation)
        self.norm1 = get_normalization(norm, in_channels)
        self.norm2 = get_normalization(norm, self.out_channels - input_dims)
        self.norm3 = get_normalization(norm, self.out_channels)

        self.stem = nn.Sequential(
            self.upsample,
            self.matcher,
            self.norm2,
            self.activation,
        )

    def forward(self, x, y): # x is the input, y is the original input resampled
        x = self.stem(x)
        x = torch.cat((x, self.raw_block(y)), dim=1)
        x = self.mixer(x)

        return x


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

        self.depths = [3, 3, 3, 3] if depths is None else depths
        self.dims = [64, 64, 64, 64] if dims is None else dims
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

        # Could be calculated lazily.
        self.sizes = [self.input_size // (2 ** (i + 1)) for i in reversed(range(len(self.depths)))]
        self.pools = [nn.AdaptiveAvgPool2d(i) for i in self.sizes]

        self.stem_size = (self.sizes[0] ** 2) * self.input_dim
        self.channel_match = BasicCNNBlock(self.input_dim, self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding, residual=True)
        self.activation = get_activation(self.activation)

        self.stem = nn.Sequential(
            self.pools[0],
            SE_BlockV3(
                input_dim,
                reduction_c=1,
                reduction_s=2 if self.sizes[0] < 16 else self.sizes[0] // 8,
                activation=self.activation,
                norm=self.norm,
                first_layer=True,
            ),
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
            self.squeeze_blocks.append(
                SE_BlockV3(
                    self.dims[i],
                    reduction_c=2,
                    reduction_s=2 if self.sizes[i] < 16 else self.sizes[i] // 8,
                    activation=self.activation,
                    norm=self.norm,
                    first_layer=False,
                )
            )

        self.mixer_blocks = nn.ModuleList(self.mixer_blocks)
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)
        self.squeeze_blocks = nn.ModuleList(self.squeeze_blocks)


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

        x = self.stem(x)
        x = x.reshape(x.size(0), self.input_dim, self.sizes[0], self.sizes[0])
        x = self.channel_match(x)

        for i in range(len(self.depths)):
            if i < len(self.depths) - 1:
                x = self.mixer_blocks[i](x, self.pools[i + 1](identity))
            else:
                x = self.mixer_blocks[i](x, identity)
            
            x = self.decoder_blocks[i](x)
            x = self.squeeze_blocks[i](x)

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
    )
    
    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )