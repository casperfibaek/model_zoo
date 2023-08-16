import torch
import torch.nn as nn
from timm.layers import DropPath

class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0, scale_learnable=True, bias_learnable=True, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU6(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return (self.scale * (self.relu(x) ** 2)) + self.bias


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, residual=True, drop_n=.0, drop_p=.0):
        super(CNNBlock, self).__init__()

        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.activation1 = StarReLU(inplace=True)
        self.activation2 = StarReLU(inplace=True)
        self.activation3 = StarReLU(inplace=True)

        self.norm1 = nn.BatchNorm2d(self.out_channels)
        self.norm2 = nn.BatchNorm2d(self.out_channels)
        self.norm3 = nn.BatchNorm2d(self.out_channels)
        self.norm4 = nn.BatchNorm2d(self.out_channels)

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding="same", groups=self.out_channels, bias=False)
        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding="same", groups=1, bias=False)

        self.drop1 = nn.Dropout2d(drop_n) if drop_n > 0. else nn.Identity()
        self.drop2 = nn.Dropout2d(drop_n) if drop_n > 0. else nn.Identity()
        self.drop3 = nn.Dropout2d(drop_n) if drop_n > 0. else nn.Identity()
        self.drop_path = DropPath(drop_p) if drop_p > 0. else nn.Identity()

        if self.residual and in_channels != out_channels:
            self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.drop1(self.activation1(self.norm1(self.conv1(x))))
        out = self.drop2(self.activation2(self.norm2(self.conv2(out))))
        out = self.drop3(self.norm3(self.conv3(out)))

        if self.residual:
            if x.size(1) != out.size(1):
                x = self.norm4(self.match_channels(x))

            x = x + self.drop_path(out)

        x = self.activation3(x)

        return x



class MLPMixerBlock(nn.Module):
    def __init__(self, chw, output_dim, patch_size, hidden_dims=512, drop_n=0.1, drop_p=0.1):
        super(MLPMixerBlock, self).__init__()
        self.patch_size = patch_size
        self.chw = chw
        self.num_patches = (chw[1] // patch_size) ** 2
        self.output_dim = output_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dims // 2))
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
        self.drop_path = DropPath(drop_p) if drop_p > 0. else nn.Identity()

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
        out = self.projection(self.patchify_batch(x))
        out = out + self.pos_embed
        out = self.norm1(out)
        out = out + self.dropout1(self.channel_mlp(out))
        out = out.transpose(1, 2)

        out = self.norm2(out)
        out = out + self.dropout2(self.token_mlp(out))
        out = out.transpose(1, 2)

        out = self.reprojection(out)
        out = self.unpatchify_batch(out)

        x = x + self.drop_path(out)
        
        return x


class DiamondFormer(nn.Module):
    def __init__(self, chw, output_dim, patch_size, embed_dim=512, dim=256, depth=3):
        super(DiamondFormer, self).__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.chw = chw
        self.std = .02
        self.num_patches = (chw[1] // patch_size) * (chw[2] // patch_size)

        self.stem_channels = 64
        self.stem = nn.Sequential(
            CNNBlock(chw[0], self.stem_channels),
            CNNBlock(self.stem_channels, self.stem_channels * 2),
            CNNBlock(self.stem_channels * 2, self.stem_channels),
        )

        # self.mlp_block1 = MLPMixerBlock((64, chw[1], chw[2]), 64, patch_size, hidden_dims=embed_dim)
        # self.mlp_block2 = MLPMixerBlock((64, chw[1], chw[2]), 64, patch_size, hidden_dims=embed_dim)
        # self.mlp_block3 = MLPMixerBlock((64, chw[1], chw[2]), 64, patch_size, hidden_dims=embed_dim)

        self.cnn_block1 = CNNBlock(64, 64)
        self.cnn_block2 = CNNBlock(64, 64)
        self.cnn_block3 = CNNBlock(64, 64)

        self.cnn_out = nn.Conv2d(64, 1, 1)
        
    def forward(self, x):
        x = self.stem(x)
        # x = self.mlp_block1(x)
        x = self.cnn_block1(x)
        # x = self.mlp_block2(x)
        x = self.cnn_block2(x)
        # x = self.mlp_block3(x)
        x = self.cnn_block3(x)
        x = self.cnn_out(x)

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
        # embed_dim=512,
        # dim=256,
        # depth=6,
    )
    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )
