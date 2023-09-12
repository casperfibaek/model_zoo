import torch
import torch.nn as nn


class ScaleSkip2D(nn.Module):
    def __init__(self, channels, drop_p=0.1):
        super(ScaleSkip2D, self).__init__()
        self.channels = channels
        self.drop_p = drop_p

        self.skipscale = nn.Parameter(torch.ones(1, self.channels, 1, 1))
        self.skipbias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
        self.dropout = nn.Dropout2d(drop_p) if drop_p > 0. else nn.Identity()

        torch.nn.init.normal_(self.skipscale, mean=1.0, std=.02)
        torch.nn.init.normal_(self.skipbias, mean=0.0, std=.02)

    def forward(self, x, skip_connection):
        y = self.skipscale * self.dropout(skip_connection) + self.skipbias

        return x + y


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0, scale_learnable=True, bias_learnable=True, mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


class CNNBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        *,
        apply_residual=True,
        drop_n=0.0,
        drop_p=0.0,
    ):
        super(CNNBlock, self).__init__()

        self.apply_residual = apply_residual
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.activation1 = nn.ReLU6()
        self.activation2 = nn.ReLU6()
        self.activation3 = nn.ReLU6()

        self.norm1 = nn.BatchNorm2d(self.out_channels)
        self.norm2 = nn.BatchNorm2d(self.out_channels)
        self.norm3 = nn.BatchNorm2d(self.out_channels)
        self.norm4 = nn.BatchNorm2d(self.out_channels)

        self.drop = nn.Dropout2d(drop_n) if drop_n > 0. else nn.Identity()
        self.skipper = ScaleSkip2D(self.out_channels, drop_p=drop_p)

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding="same", groups=self.out_channels, bias=False)
        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding="same", groups=1, bias=False)

        if self.apply_residual and in_channels != out_channels:
            self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        x = self.activation1(self.norm1(self.conv1(x)))
        x = self.activation2(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        x = self.drop(x)

        if self.apply_residual:
            if x.size(1) != identity.size(1):
                identity = self.norm4(self.match_channels(identity))

            x = self.skipper(identity, x)

        x = self.activation3(x)

        return x


class MLPMixerLayer(nn.Module):
    def __init__(self,
        embedding_dims,
        patch_size=16,
        chw=(10, 64, 64),
        expansion=2,
        drop_n=0.0,
    ):
        super(MLPMixerLayer, self).__init__()
        self.embed_dims = embedding_dims
        self.patch_size = patch_size
        self.chw = chw
        self.expansion = expansion

        self.num_patches = (chw[1] // patch_size) * (chw[2] // patch_size)
        self.pixels =  int(chw[1] * chw[2] / self.num_patches)

        self.norm1 = RMSNorm(self.pixels)
        self.drop1 = nn.Dropout1d(drop_n) if drop_n > 0. else nn.Identity()
        self.mlp_token = nn.Sequential(
            nn.Linear(self.pixels, int(self.pixels * self.expansion)),
            StarReLU(),
            nn.Linear(int(self.pixels * self.expansion), self.pixels)
        )

        self.norm2 = RMSNorm(self.num_patches)
        self.drop2 = nn.Dropout1d(drop_n) if drop_n > 0. else nn.Identity()
        self.mlp_patch = nn.Sequential(
            nn.Linear(self.num_patches, int(self.num_patches * self.expansion)),
            StarReLU(),
            nn.Linear(int(self.num_patches * self.expansion), self.num_patches)
        )

        self.norm3 = RMSNorm(self.embed_dims)
        self.drop3 = nn.Dropout1d(drop_n) if drop_n > 0. else nn.Identity()
        self.mlp_channel = nn.Sequential(
            nn.Linear(self.embed_dims, int(self.embed_dims * self.expansion)),
            StarReLU(),
            nn.Linear(int(self.embed_dims * self.expansion), self.embed_dims)
        )


    def patchify_batch(self, tensor):
        tile_size = self.patch_size
        B, C, H, W = tensor.shape

        tensor = tensor.unfold(2, tile_size, tile_size).unfold(3, tile_size, tile_size)
        tensor = tensor.reshape(B, C, H//tile_size, W//tile_size, tile_size*tile_size)
        tensor = tensor.reshape(B, C, H//tile_size * W//tile_size, tile_size*tile_size)

        return tensor
    
    def unpatchify_batch(self, tensor):
        B, C, _, _ = tensor.shape
        H, W = self.chw[1], self.chw[2]
        tile_size = self.patch_size

        tensor = tensor.reshape(B, C, H//tile_size, W//tile_size, tile_size, tile_size)
        tensor = tensor.reshape(B, C, H, W)

        return tensor


    def forward(self, x):
        x = self.patchify_batch(x)
        x = x + self.drop1(self.mlp_token(self.norm1(x)))
        x = x.transpose(3, 2)
        x = x + self.drop2(self.mlp_patch(self.norm2(x)))
        x = x.transpose(1, 3)
        x = x + self.drop3(self.mlp_channel(self.norm3(x)))
        x = x.transpose(1, 3).transpose(3, 2)
        x = self.unpatchify_batch(x)

        return x


class MLPMixer(nn.Module):
    def __init__(self,
        chw,
        output_dim,
        embedding_dims=32,
        expansion=2,
        drop_n=0.0,
        drop_p=0.0,
    ):
        super(MLPMixer, self).__init__()
        self.chw = chw
        self.output_dim = output_dim
        self.embedding_dims = embedding_dims
        self.expansion = expansion
        self.drop_n = drop_n
        self.drop_p = drop_p
        self.std = .05

        self.stem = nn.Sequential(
            CNNBlock(chw[0], self.embedding_dims, drop_n=0.0, drop_p=0.0),
            CNNBlock(self.embedding_dims, self.embedding_dims, drop_n=drop_n, drop_p=drop_p),
            CNNBlock(self.embedding_dims, self.embedding_dims, drop_n=drop_n, drop_p=drop_p),
        )

        self.mixer_layers = []
        for v in [16, 8, 4, 2]:
            self.mixer_layers.append(
                MLPMixerLayer(
                    self.embedding_dims,
                    patch_size=v,
                    chw=chw,
                    expansion=self.expansion,
                )
            )
        self.mixer = nn.Sequential(*self.mixer_layers)

        self.skipper = ScaleSkip2D(self.embedding_dims, drop_p=drop_p)

        self.head = nn.Sequential(
            CNNBlock(self.embedding_dims, self.embedding_dims, drop_n=drop_n, drop_p=drop_p),
            nn.Conv2d(self.embedding_dims, self.output_dim, 1, padding=0),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = self.std
            torch.nn.init.trunc_normal_(m.weight, std=std, a=-std * 2, b=std * 2)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, identity):
        skip = self.stem(identity)
        x = self.mixer(skip)
        x = self.skipper(skip, x)
        x = self.head(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 16
    CHANNELS = 10
    HEIGHT = 64
    WIDTH = 64

    model = MLPMixer(
        chw=(10, 64, 64),
        output_dim=1,
        embedding_dims=32,
        expansion=2,
        drop_n=0.0,
        drop_p=0.0,
    )
    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )
