import torch
import torch.nn as nn
from timm.models.layers import DropPath


def patchify_batch(images, patch_size):
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

def unpatchify_batch(patches, chw, output_dim, patch_size):
    _, height, width = chw
    channels = output_dim
    patch_size = patch_size
    batch_size, _n_patches, _ = patches.shape
    
    patches = patches.reshape(batch_size, height // patch_size, width // patch_size, patch_size, patch_size, channels)
    patches = patches.swapaxes(2, 3)
    patches = patches.reshape(batch_size, height, width, channels)
    patches = patches.swapaxes(1, -1)

    return patches


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


class TokenMixer(nn.Module):
    def __init__(self, pool_size=3, dim=None):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False,
        )
        self.sep_conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            groups=dim,
            padding="same",
        )

    def forward(self, x):
        y = self.pool(x)
        y = self.sep_conv(y)

        return y - x


class Mlp(nn.Module):
    def __init__(self,
        dim,
        mlp_ratio=4,
        out_features=None,
        act_layer=StarReLU,
        drop=0.0,
        bias=False,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class MetaFormerBlock(nn.Module):
    def __init__(self,
            dim,
            mlp=Mlp,
            drop_p=0.0,
            chw=(10, 64, 64),
            output_dim=1,
            patch_size=8,
        ):
        super().__init__()
        self.dim = dim
        self.chw = chw
        self.output_dim = output_dim
        self.patch_size = patch_size

        self.project_dim = max(int(round((self.dim * (self.patch_size ** 2)) / (self.chw[1] * self.chw[2]))), 4)
        self.token_mixer = TokenMixer(pool_size=3, dim=self.project_dim)

        self.project_out = nn.Linear(dim, self.project_dim * (patch_size ** 2))
        self.project_in = nn.Linear(self.project_dim  * (patch_size ** 2), dim)

        self.drop_path1 = DropPath(drop_p) if drop_p > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(self.project_dim * (patch_size ** 2))
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = mlp(dim)
        self.drop_path2 = DropPath(drop_p) if drop_p > 0. else nn.Identity()

    def _patchify_batch(self, images):
        return patchify_batch(images, self.patch_size)

    def _unpatchify_batch(self, patches):
        return unpatchify_batch(patches, self.chw, self.project_dim, self.patch_size)
        
    def forward(self, x):
        out = self.norm1(self.project_out(x))
        out1 = self._unpatchify_batch(out)
        out2 = self.token_mixer(out1)
        out3 = self._patchify_batch(out2)
        out4 = self.project_in(out3)
        x = x + self.drop_path1(out4)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


class MetaFormer(nn.Module):
    def __init__(self,
        chw,
        output_dim,
        patch_size,
        depth=3,
        embed_dim=512,
        embed_channels=32,
        clamp_output=False,
        clamp_min=0.0,
        clamp_max=1.0,
        drop_n=0.0,
        drop_p=0.0,
    ):
        super(MetaFormer, self).__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.chw = chw
        self.clamp_output = clamp_output
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.drop_n = drop_n
        self.drop_p = drop_p
        self.num_patches = (chw[1] // patch_size) * (chw[2] // patch_size)

        self.stem_channels = embed_channels
        self.stem = nn.Sequential(
            CNNBlock(chw[0], self.stem_channels, drop_n=self.drop_n, drop_p=self.drop_p),
            CNNBlock(self.stem_channels, self.stem_channels * 2, drop_n=self.drop_n, drop_p=self.drop_p),
            CNNBlock(self.stem_channels * 2, self.stem_channels, drop_n=self.drop_n, drop_p=self.drop_p),
        )
        self.head = nn.Sequential(
            CNNBlock(self.stem_channels, self.stem_channels, drop_n=self.drop_n, drop_p=self.drop_p),
            CNNBlock(self.stem_channels, self.stem_channels * 2, drop_n=self.drop_n, drop_p=self.drop_p),
            CNNBlock(self.stem_channels * 2, self.output_dim, drop_n=self.drop_n, drop_p=self.drop_p),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.projection = nn.Linear(self.stem_channels * (patch_size ** 2), embed_dim)
        self.reprojection = nn.Linear(embed_dim, self.stem_channels * (patch_size ** 2))

        self.blocks = nn.ModuleList([
            MetaFormerBlock(
                embed_dim,
                mlp=Mlp,
                chw=self.chw,
                output_dim=self.stem_channels,
                patch_size=self.patch_size,
                drop_p=self.drop_p,
            ) for _ in range(depth)
        ])

    def _patchify_batch(self, images):
        return patchify_batch(images, self.patch_size)

    def _unpatchify_batch(self, patches):
        return unpatchify_batch(patches, self.chw, self.stem_channels, self.patch_size)
        
    def forward(self, x):
        x = self.projection(self._patchify_batch(self.stem(x)))
        for block in self.blocks:
            x = block(x + self.pos_embed)
        x = self._unpatchify_batch(self.reprojection(x))
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

    model = MetaFormer(
        chw=(CHANNELS, HEIGHT, WIDTH),
        output_dim=1,
        patch_size=8,
        embed_dim=512,
    )
    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )


# PoolFormer
# model = MetaFormer(
#     depths=[2, 2, 6, 2],
#     dims=[64, 128, 320, 512],
#     token_mixers=Pooling,
#     norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
#     **kwargs)

# ConvFormer
# model = MetaFormer(
#     depths=[3, 3, 9, 3],
#     dims=[64, 128, 320, 512],
#     token_mixers=SepConv,
#     head_fn=MlpHead,
#     **kwargs)

# CaFormer
# model = MetaFormer(
#     depths=[3, 3, 9, 3],
#     dims=[64, 128, 320, 512],
#     token_mixers=[SepConv, SepConv, Attention, Attention],
#     head_fn=MlpHead,
#     **kwargs)
