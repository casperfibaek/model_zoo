import torch
import torch.nn as nn



class ResNetBlock_V1_5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock_V1_5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

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
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += self.shortcut(identity)
        x = self.relu(x)

        return x



class ResNetEncoderBlock(nn.Module):
    """ ResNet Encoder block """
    def __init__(self, depth, in_channels, out_channels):
        super(ResNetEncoderBlock, self).__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.blocks = []
        for i in range(self.depth):
            if i == 0:
                block = ResNetBlock_V1_5(self.in_channels, self.out_channels)
            else:
                block = ResNetBlock_V1_5(self.out_channels, self.out_channels)

            self.blocks.append(block)

        self.blocks = nn.Sequential(*self.blocks)
        self.downsample = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        before_downsample = x
        for i in range(self.depth):
            x = self.blocks[i](x)

        x = self.downsample(x)

        return x, before_downsample


class ResNetDecoderBlock(nn.Module):
    """ ResNet Decoder block """
    def __init__(self, depth, in_channels, out_channels):
        super(ResNetDecoderBlock, self).__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upsample = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=2, stride=2)

        self.blocks = []
        for _ in range(self.depth):
            block = ResNetBlock_V1_5(self.out_channels, self.out_channels)
            self.blocks.append(block)

        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x, y):
        x = self.upsample(x)
        x += y

        for i in range(self.depth):
            x = self.blocks[i](x)

        return x


class ResNetUnet(nn.Module):
    """
    Basic ResNet Architecture
    """
    def __init__(self, *, input_dim=10, output_dim=1, depths=None, dims=None, stem_kernel_size=7, clamp_output=False, clamp_min=0.0, clamp_max=1.0):
        super(ResNetUnet, self).__init__()

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
            encoder_block = ResNetEncoderBlock(
                self.depths[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                self.dims[i],
            )
            self.encoder_blocks.append(encoder_block)

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

        self.decoder_blocks = []

        for i in reversed(range(len(self.encoder_blocks))):
            decoder_block = ResNetDecoderBlock(
                self.depths[i],
                self.dims[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
            )
            self.decoder_blocks.append(decoder_block)

        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        self.bridge = nn.Sequential(
            ResNetBlock_V1_5(self.dims[-1], self.dims[-1]),
        )
        
        self.head = nn.Sequential(
            ResNetBlock_V1_5(self.dims[0], self.dims[0]),
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



class ResNet(nn.Module):
    """
    Basic ResNet Architecture
    """
    def __init__(self, *, input_dim=10, output_dim=1, depths=None, dims=None, stem_kernel_size=7, clamp_output=False, clamp_min=0.0, clamp_max=1.0):
        super(ResNet, self).__init__()

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
            encoder_block = ResNetEncoderBlock(
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


def ResNetUnet_atto(**kwargs):
    model = ResNetUnet(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def ResNetUnet_femto(**kwargs):
    model = ResNetUnet(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def ResNetUnet_pico(**kwargs):
    model = ResNetUnet(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def ResNetUnet_nano(**kwargs):
    model = ResNetUnet(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def ResNetUnet_tiny(**kwargs):
    model = ResNetUnet(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def ResNetUnet_base(**kwargs):
    model = ResNetUnet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def ResNetUnet_large(**kwargs):
    model = ResNetUnet(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def ResNetUnet_huge(**kwargs):
    model = ResNetUnet(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model

def ResNet_atto(**kwargs):
    model = ResNet(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def ResNet_femto(**kwargs):
    model = ResNet(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def ResNet_pico(**kwargs):
    model = ResNet(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def ResNet_nano(**kwargs):
    model = ResNet(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def ResNet_tiny(**kwargs):
    model = ResNet(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def ResNet_base(**kwargs):
    model = ResNet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def ResNet_large(**kwargs):
    model = ResNet(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def ResNet_huge(**kwargs):
    model = ResNet(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model



if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 32
    CHANNELS = 10
    HEIGHT = 128
    WIDTH = 128

    model = ResNetUnet_pico(
        input_dim=10,
        output_dim=1,
        stem_kernel_size=5,
    )
    
    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )