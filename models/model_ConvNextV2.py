import sys; sys.path.append("../")
import torch
import torch.nn as nn
from utils import LayerNorm, GRN



class ConvNextV2Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
    """
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.shortcut = nn.Identity()
        dwconv_in_channels = out_channels
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
            dwconv_in_channels = in_channels

        self.dwconv = nn.Conv2d(dwconv_in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=dwconv_in_channels) # depthwise conv
        self.norm = LayerNorm(out_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(out_channels, 4 * out_channels)
        self.act = nn.GELU()
        self.grn = GRN(4 * out_channels)
        self.pwconv2 = nn.Linear(4 * out_channels, out_channels)


    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.shortcut(identity) + x

        return x


class ConvNextV2EncoderBlock(nn.Module):
    """ ConvNext V2 Encoder block """
    def __init__(self, depth, in_channels, out_channels):
        super(ConvNextV2EncoderBlock, self).__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.blocks = []
        for i in range(self.depth):
            if i == 0:
                block = ConvNextV2Block(self.in_channels, self.out_channels)
            else:
                block = ConvNextV2Block(self.out_channels, self.out_channels)

            self.blocks.append(block)

        self.blocks = nn.Sequential(*self.blocks)
        self.downsample = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        before_downsample = x
        for i in range(self.depth):
            x = self.blocks[i](x)

        x = self.downsample(x)

        return x, before_downsample


class ConvNextV2DecoderBlock(nn.Module):
    """ ConvNext V2 Decoder block """
    def __init__(self, depth, in_channels, out_channels):
        super(ConvNextV2DecoderBlock, self).__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upsample = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=2, stride=2)

        self.blocks = []
        for _ in range(self.depth):
            block = ConvNextV2Block(self.out_channels, self.out_channels)
            self.blocks.append(block)

        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x, y):
        x = self.upsample(x)
        x += y

        for i in range(self.depth):
            x = self.blocks[i](x)

        return x


class ConvNextV2Unet(nn.Module):
    """
    Basic ConvNext Architecture
    """
    def __init__(self, *, input_dim=10, output_dim=1, depths=None, dims=None, stem_kernel_size=7, clamp_output=False, clamp_min=0.0, clamp_max=1.0):
        super(ConvNextV2Unet, self).__init__()

        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.dims = [96, 192, 384, 768] if dims is None else dims
        self.stem_kernel_size = stem_kernel_size
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.clamp_output = clamp_output
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        self.dims = [v // 2 for v in self.dims]

        assert len(self.depths) == len(self.dims), "depths and dims must have the same length."

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_dim, self.dims[0], kernel_size=self.stem_kernel_size, padding=self.stem_kernel_size // 2),
            nn.BatchNorm2d(self.dims[0]),
            nn.ReLU(),
        )

        self.encoder_blocks = []
        for i in range(len(self.depths)):
            encoder_block = ConvNextV2EncoderBlock(
                self.depths[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                self.dims[i],
            )
            self.encoder_blocks.append(encoder_block)

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

        self.decoder_blocks = []

        for i in reversed(range(len(self.encoder_blocks))):
            decoder_block = ConvNextV2DecoderBlock(
                self.depths[i],
                self.dims[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
            )
            self.decoder_blocks.append(decoder_block)

        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        self.bridge = nn.Sequential(
            ConvNextV2Block(self.dims[-1], self.dims[-1]),
        )
        
        self.head = nn.Sequential(
            ConvNextV2Block(self.dims[0], self.dims[0]),
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



class ConvNextV2(nn.Module):
    """
    Basic ConvNext Architecture
    """
    def __init__(self, *, input_dim=10, output_dim=1, depths=None, dims=None, stem_kernel_size=7, clamp_output=False, clamp_min=0.0, clamp_max=1.0):
        super(ConvNextV2, self).__init__()

        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.dims = [96, 192, 384, 768] if dims is None else dims
        self.stem_kernel_size = stem_kernel_size
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.clamp_output = clamp_output
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        assert len(self.depths) == len(self.dims), "depths and dims must have the same length."

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_dim, self.dims[0], kernel_size=self.stem_kernel_size, padding=self.stem_kernel_size // 2),
            nn.BatchNorm2d(self.dims[0]),
            nn.ReLU(),
        )

        self.encoder_blocks = []
        for i in range(len(self.depths)):
            encoder_block = ConvNextV2EncoderBlock(
                self.depths[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                self.dims[i],
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


def ConvNextV2Unet_atto(**kwargs):
    model = ConvNextV2Unet(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def ConvNextV2Unet_femto(**kwargs):
    model = ConvNextV2Unet(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def ConvNextV2Unet_pico(**kwargs):
    model = ConvNextV2Unet(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def ConvNextV2Unet_nano(**kwargs):
    model = ConvNextV2Unet(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def ConvNextV2Unet_tiny(**kwargs):
    model = ConvNextV2Unet(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def ConvNextV2Unet_base(**kwargs):
    model = ConvNextV2Unet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def ConvNextV2Unet_large(**kwargs):
    model = ConvNextV2Unet(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def ConvNextV2Unet_huge(**kwargs):
    model = ConvNextV2Unet(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model

def ConvNextV2_atto(**kwargs):
    model = ConvNextV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def ConvNextV2_femto(**kwargs):
    model = ConvNextV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def ConvNextV2_pico(**kwargs):
    model = ConvNextV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def ConvNextV2_nano(**kwargs):
    model = ConvNextV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def ConvNextV2_tiny(**kwargs):
    model = ConvNextV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def ConvNextV2_base(**kwargs):
    model = ConvNextV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def ConvNextV2_large(**kwargs):
    model = ConvNextV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def ConvNextV2_huge(**kwargs):
    model = ConvNextV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model



if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 32
    CHANNELS = 10
    HEIGHT = 128
    WIDTH = 128

    model = ConvNextV2Unet_pico(
        input_dim=10,
        output_dim=1,
        stem_kernel_size=5,
    )
    
    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )