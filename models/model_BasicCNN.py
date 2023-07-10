import sys; sys.path.append("../")
import torch
import torch.nn as nn
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



class BasicEncoderBlock(nn.Module):
    """ Encoder block """
    def __init__(self, depth, in_channels, out_channels, norm="batch", activation="relu", padding="same"):
        super(BasicEncoderBlock, self).__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.norm = norm
        self.padding = padding

        self.blocks = []
        for i in range(self.depth):
            _in_channels = self.in_channels if i == 0 else self.out_channels
            block = BasicCNNBlock(_in_channels, self.out_channels, norm=self.norm, activation=self.activation, padding=self.padding)

            self.blocks.append(block)

        self.blocks = nn.Sequential(*self.blocks)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        before_downsample = x
        for i in range(self.depth):
            x = self.blocks[i](x)

        x = self.downsample(x)

        return x, before_downsample


class BasicDecoderBlock(nn.Module):
    """ Decoder block """
    def __init__(self, depth, in_channels, out_channels, *, norm="batch", activation="relu", padding="same"):
        super(BasicDecoderBlock, self).__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_blocks = activation
        self.activation = get_activation(activation)
        self.norm = norm
        self.padding = padding

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.match_channels = BasicCNNBlock(self.in_channels + self.out_channels, self.out_channels, norm=self.norm, activation=self.activation_blocks, padding=self.padding)

        self.blocks = []
        for _ in range(self.depth):
            block = BasicCNNBlock(self.out_channels, self.out_channels, norm=self.norm, activation=self.activation_blocks, padding=self.padding)
            self.blocks.append(block)

        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x, y): # y is the skip connection
        x = self.upsample(x)
        x = torch.cat([x, y], dim=1)
        x = self.match_channels(x)

        for i in range(self.depth):
            x = self.blocks[i](x)

        return x


class BasicUnet(nn.Module):
    """ Basic Architecture """
    def __init__(self, *, input_dim=10, output_dim=1, depths=None, dims=None, clamp_output=False, clamp_min=0.0, clamp_max=1.0, activation="relu", norm="batch", padding="same"):
        super(BasicUnet, self).__init__()

        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.dims = [96, 192, 384, 768] if dims is None else dims
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.clamp_output = clamp_output
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.activation = activation
        self.norm = norm
        self.padding = padding

        self.dims = [v // 2 for v in self.dims]

        assert len(self.depths) == len(self.dims), "depths and dims must have the same length."

        self.stem = nn.Sequential(
            BasicCNNBlock(self.input_dim, self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding),
        )

        self.encoder_blocks = []
        for i in range(len(self.depths)):
            encoder_block = BasicEncoderBlock(
                self.depths[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                self.dims[i],
                norm=self.norm,
                activation=self.activation,
                padding=self.padding,
            )
            self.encoder_blocks.append(encoder_block)

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

        self.decoder_blocks = []

        for i in reversed(range(len(self.encoder_blocks))):
            decoder_block = BasicDecoderBlock(
                self.depths[i],
                self.dims[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                norm=self.norm,
                activation=self.activation,
                padding=self.padding,
            )
            self.decoder_blocks.append(decoder_block)

        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        self.bridge = nn.Sequential(
            BasicCNNBlock(self.dims[-1], self.dims[-1], norm=self.norm, activation=self.activation, padding=self.padding),
        )
        
        self.head = nn.Sequential(
            BasicCNNBlock(self.dims[0], self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding),
            nn.Conv2d(self.dims[0], self.output_dim, kernel_size=1, padding=0),
        )

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
        skip_connections = []
        
        x = self.stem(x)
        for block in self.encoder_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        x = self.bridge(x)

        for block in self.decoder_blocks:
            skip = skip_connections.pop()
            x = block(x, skip)

        x = self.head(x)

        if self.clamp_output:
            x = torch.clamp(x, self.clamp_min, self.clamp_max)

        return x



class Basic(nn.Module):
    """
    Basic Architecture
    """
    def __init__(self, *, input_dim=10, output_dim=1, depths=None, dims=None, clamp_output=False, clamp_min=0.0, clamp_max=1.0, activation="relu", norm="batch", padding="same"):
        super(Basic, self).__init__()

        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.dims = [96, 192, 384, 768] if dims is None else dims
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.clamp_output = clamp_output
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.activation = activation
        self.norm = norm
        self.padding = padding

        assert len(self.depths) == len(self.dims), "depths and dims must have the same length."

        self.stem = BasicCNNBlock(self.input_dim, self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding)

        self.encoder_blocks = []
        for i in range(len(self.depths)):
            encoder_block = BasicEncoderBlock(
                self.depths[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                self.dims[i],
                norm=self.norm,
                activation=self.activation,
                padding=self.padding,
            )
            self.encoder_blocks.append(encoder_block)

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.dims[-1], self.output_dim),
        )

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
        x = self.stem(x)

        for block in self.encoder_blocks:
            x, _ = block(x)

        x = self.head(x)

        if self.clamp_output:
            x = torch.clamp(x, self.clamp_min, self.clamp_max)

        return x

def BasicUnet_atto(**kwargs):
    model = BasicUnet(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def BasicUnet_femto(**kwargs):
    model = BasicUnet(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def BasicUnet_pico(**kwargs):
    model = BasicUnet(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def BasicUnet_nano(**kwargs):
    model = BasicUnet(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def BasicUnet_tiny(**kwargs):
    model = BasicUnet(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def BasicUnet_base(**kwargs):
    model = BasicUnet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def BasicUnet_large(**kwargs):
    model = BasicUnet(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def BasicUnet_huge(**kwargs):
    model = BasicUnet(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model

def Basic_atto(**kwargs):
    model = Basic(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def Basic_femto(**kwargs):
    model = Basic(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def Basic_pico(**kwargs):
    model = Basic(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def Basic_nano(**kwargs):
    model = Basic(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def Basic_tiny(**kwargs):
    model = Basic(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def Basic_base(**kwargs):
    model = Basic(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def Basic_large(**kwargs):
    model = Basic(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def Basic_huge(**kwargs):
    model = Basic(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model



if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 32
    CHANNELS = 10
    HEIGHT = 64
    WIDTH = 64

    model = BasicUnet_pico(
        input_dim=10,
        output_dim=1,
    )
    
    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )