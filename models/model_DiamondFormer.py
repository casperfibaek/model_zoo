import torch
import torch.nn as nn
from timm.layers import DropPath


class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0, scale_learnable=True, bias_learnable=True, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return (self.scale * (self.relu(x) ** 2)) + self.bias


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, residual=True):
        super(CNNBlock, self).__init__()

        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.activation = nn.ReLU()

        self.norm1 = nn.BatchNorm2d(self.out_channels)
        self.norm2 = nn.BatchNorm2d(self.out_channels)
        self.norm3 = nn.BatchNorm2d(self.out_channels)
        self.norm4 = nn.BatchNorm2d(self.out_channels)

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding="same", groups=self.out_channels, bias=False)
        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding="same", groups=1, bias=False)

        if self.residual and in_channels != out_channels:
            self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))

        if self.residual:
            if x.size(1) != identity.size(1):
                identity = self.norm4(self.match_channels(identity))

            x = identity + x

        x = self.activation(x)

        return x


class MLPMixerBlock(nn.Module):
    def __init__(self, chw, output_dim, patch_size, hidden_dims=512, drop_n=0.1, drop_p=0.1):
        super(MLPMixerBlock, self).__init__()
        self.patch_size = patch_size
        self.chw = chw
        self.num_patches = (chw[1] // patch_size) ** 2
        self.output_dim = output_dim
        
        self.cnn_in = CNNBlock(chw[0], chw[0], residual=False)
        self.projection = nn.Linear(chw[0] * (patch_size ** 2), hidden_dims // 2)
        self.reprojection = nn.Linear(hidden_dims // 2, output_dim * (patch_size ** 2))

        self.norm1 = nn.LayerNorm(hidden_dims // 2)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dims // 2, hidden_dims),
            StarReLU(),
            nn.Linear(hidden_dims, hidden_dims // 2)
        )
        
        self.norm2 = nn.LayerNorm(self.num_patches)
        self.token_mlp = nn.Sequential(
            nn.Linear(self.num_patches, hidden_dims),
            StarReLU(),
            nn.Linear(hidden_dims, self.num_patches)
        )

        self.dropout1 = nn.Dropout(drop_n) if drop_n > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_n) if drop_n > 0. else nn.Identity()
        self.drop_path1 = DropPath(drop_p) if drop_p > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_p) if drop_p > 0. else nn.Identity()

    def patchify_batch(self, images):
        patch_size = self.patch_size
        batch_size, channels, height, width = images.shape

        n_patches_y = height // patch_size
        n_patches_x = width // patch_size
        n_patches = n_patches_y * n_patches_x

        channel_last = images.swapaxes(1, -1)
        reshaped = channel_last.reshape(batch_size, n_patches_y, patch_size, n_patches_x, patch_size, channels)
        swaped = reshaped.swapaxes(2, 3)
        blocks = swaped.reshape(batch_size, -1, patch_size, patch_size, channels)
        patches = blocks.reshape(batch_size, n_patches, -1)

        return patches

    def unpatchify_batch(self, patches):
        _, height, width = self.chw
        channels = self.output_dim
        patch_size = self.patch_size
        batch_size, _n_patches, _ = patches.shape
        
        patches = patches.reshape(batch_size, height // patch_size, width // patch_size, patch_size, patch_size, channels)
        patches = patches.swapaxes(2, 3)
        patches = patches.reshape(batch_size, height, width, channels)
        patches = patches.swapaxes(1, -1)

        return patches

    def forward(self, x):
        out = self.cnn_in(x)
        out = self.projection(self.patchify_batch(x))
        out = self.norm1(out)
        out = out + self.drop_path1(self.dropout1(self.channel_mlp(out)))
        out = out.transpose(1, 2)

        out = self.norm2(out)
        out = out + self.drop_path2(self.dropout2(self.token_mlp(out)))
        out = out.transpose(1, 2)

        out = self.reprojection(out)
        out = self.unpatchify_batch(out)
        
        return out


class DiamondFormer(nn.Module):
    def __init__(self, chw, output_dim, patch_size, embed_dim=512, dim=256, depth=3):
        super(DiamondFormer, self).__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.chw = chw
        self.std = .02
        self.num_patches = (chw[1] // patch_size) * (chw[2] // patch_size)

        self.stem = nn.Sequential(CNNBlock(chw[0], 32), CNNBlock(32, 32))

        self.mlp_block1 = MLPMixerBlock(32, 32, patch_size)
        self.mlp_block2 = MLPMixerBlock((32, chw[1], chw[2]), 48, patch_size)
        self.mlp_block3 = MLPMixerBlock((48, chw[1], chw[2]), 64, patch_size)
        self.mlp_block_out = MLPMixerBlock((64, chw[1], chw[2]), 1, patch_size)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.mlp_block1(x)
        x = self.mlp_block2(x)
        x = self.mlp_block3(x)
        x = self.mlp_block_out(x)

        x = torch.clamp(x, 0.0, 100.0)
        
        return x


if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 32
    CHANNELS = 10
    HEIGHT = 64
    WIDTH = 64

    model = DiamondFormer(
        chw=(CHANNELS, HEIGHT, WIDTH),
        output_dim=1,
        patch_size=8,
        embed_dim=1024,
        dim=512,
        depth=6,
    )
    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )
