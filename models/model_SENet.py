import sys; sys.path.append("../")
import torch
import torch.nn as nn
from utils import SE_Block, get_activation


class SENetBLock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, activation="gelu"):
        super(SENetBLock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.activation = get_activation(activation)
 
        self.se = SE_Block(out_channels, reduction)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.se(x)

        x += self.shortcut(identity)
        x = self.activation(x)

        return x



class SENetEncoderBlock(nn.Module):
    """ SENet Encoder block """
    def __init__(self, depth, in_channels, out_channels, activation="gelu"):
        super(SENetEncoderBlock, self).__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        self.blocks = []
        for i in range(self.depth):
            if i == 0:
                block = SENetBLock(self.in_channels, self.out_channels, activation=self.activation)
            else:
                block = SENetBLock(self.out_channels, self.out_channels, activation=self.activation)

            self.blocks.append(block)

        self.blocks = nn.Sequential(*self.blocks)
        self.downsample = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        before_downsample = x
        for i in range(self.depth):
            x = self.blocks[i](x)

        x = self.downsample(x)

        return x, before_downsample


class SENetDecoderBlock(nn.Module):
    """ SENet Decoder block """
    def __init__(self, depth, in_channels, out_channels, activation="gelu"):
        super(SENetDecoderBlock, self).__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        self.upsample = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=2, stride=2)

        self.blocks = []
        for _ in range(self.depth):
            block = SENetBLock(self.out_channels, self.out_channels, activation=self.activation)
            self.blocks.append(block)

        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x, y):
        x = self.upsample(x)
        x += y

        for i in range(self.depth):
            x = self.blocks[i](x)

        return x


class SENetUnet(nn.Module):
    """
    Basic SENet Architecture
    """
    def __init__(self, *, input_dim=10, output_dim=1, depths=None, dims=None, stem_kernel_size=7, clamp_output=False, clamp_min=0.0, clamp_max=1.0, activation="gelu"):
        super(SENetUnet, self).__init__()

        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.dims = [96, 192, 384, 768] if dims is None else dims
        self.stem_kernel_size = stem_kernel_size
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.clamp_output = clamp_output
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.activation = activation

        self.dims = [v // 2 for v in self.dims]

        assert len(self.depths) == len(self.dims), "depths and dims must have the same length."

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_dim, self.dims[0], kernel_size=self.stem_kernel_size, padding=self.stem_kernel_size // 2),
            SE_Block(self.dims[0]),
            nn.BatchNorm2d(self.dims[0]),
            get_activation(self.activation)
        )

        self.encoder_blocks = []
        for i in range(len(self.depths)):
            encoder_block = SENetEncoderBlock(
                self.depths[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                self.dims[i],
                activation=self.activation,
            )
            self.encoder_blocks.append(encoder_block)

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

        self.decoder_blocks = []

        for i in reversed(range(len(self.encoder_blocks))):
            decoder_block = SENetDecoderBlock(
                self.depths[i],
                self.dims[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                activation=self.activation,
            )
            self.decoder_blocks.append(decoder_block)

        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        self.bridge = nn.Sequential(
            SENetBLock(self.dims[-1], self.dims[-1], activation=self.activation),
        )
        
        self.head = nn.Sequential(
            SENetBLock(self.dims[0], self.dims[0], activation=self.activation),
            nn.Conv2d(self.dims[0], self.output_dim, kernel_size=3, padding=1),
        )

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)

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



class SENet(nn.Module):
    """
    Basic SENet Architecture
    """
    def __init__(self, *, input_dim=10, output_dim=1, depths=None, dims=None, stem_kernel_size=7, clamp_output=False, clamp_min=0.0, clamp_max=1.0, activation="gelu"):
        super(SENet, self).__init__()

        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.dims = [96, 192, 384, 768] if dims is None else dims
        self.stem_kernel_size = stem_kernel_size
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.clamp_output = clamp_output
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.activation = activation

        assert len(self.depths) == len(self.dims), "depths and dims must have the same length."

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_dim, self.dims[0], kernel_size=self.stem_kernel_size, padding=self.stem_kernel_size // 2),
            SE_Block(self.dims[0]),
            nn.BatchNorm2d(self.dims[0]),
            get_activation(self.activation),
        )

        self.encoder_blocks = []
        for i in range(len(self.depths)):
            encoder_block = SENetEncoderBlock(
                self.depths[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                self.dims[i],
                activation=self.activation,
            )
            self.encoder_blocks.append(encoder_block)

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.dims[-1], self.output_dim),
        )

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)

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


def SENetUnet_atto(**kwargs):
    model = SENetUnet(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def SENetUnet_femto(**kwargs):
    model = SENetUnet(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def SENetUnet_pico(**kwargs):
    model = SENetUnet(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def SENetUnet_nano(**kwargs):
    model = SENetUnet(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def SENetUnet_tiny(**kwargs):
    model = SENetUnet(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def SENetUnet_base(**kwargs):
    model = SENetUnet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def SENetUnet_large(**kwargs):
    model = SENetUnet(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def SENetUnet_huge(**kwargs):
    model = SENetUnet(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model

def SENet_atto(**kwargs):
    model = SENet(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def SENet_femto(**kwargs):
    model = SENet(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def SENet_pico(**kwargs):
    model = SENet(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def SENet_nano(**kwargs):
    model = SENet(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def SENet_tiny(**kwargs):
    model = SENet(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def SENet_base(**kwargs):
    model = SENet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def SENet_large(**kwargs):
    model = SENet(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def SENet_huge(**kwargs):
    model = SENet(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model



if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 32
    CHANNELS = 10
    HEIGHT = 128
    WIDTH = 128

    model = SENetUnet_pico(
        input_dim=10,
        output_dim=1,
        stem_kernel_size=5,
    )
    
    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )