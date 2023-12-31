import torch
import torch.nn as nn


class ScaleSkip1D(nn.Module):
    def __init__(self, channels, drop_n=0.1):
        super(ScaleSkip1D, self).__init__()
        self.channels = channels
        self.drop_n = drop_n

        self.skipscale = nn.Parameter(torch.ones(1, self.channels, 1))
        self.skipbias = nn.Parameter(torch.zeros(1, self.channels, 1))
        self.dropout = nn.Dropout(drop_n) if drop_n > 0. else nn.Identity()

        torch.nn.init.normal_(self.skipscale, mean=1.0, std=.02)
        torch.nn.init.normal_(self.skipbias, mean=0.0, std=.02)

    def forward(self, x, skip_connection):
        scale = torch.clamp(self.skipscale, -10.0, 10.0)
        bias = torch.clamp(self.skipbias, -1.0, 1.0)
        y = scale * self.dropout(skip_connection) + bias

        return x + y


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
        scale = torch.clamp(self.skipscale, -10.0, 10.0)
        bias = torch.clamp(self.skipbias, -1.0, 1.0)
        y = scale * self.dropout(skip_connection) + bias

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
    def __init__(self, dim, num_patches, hidden_dim, drop_n=0.0, drop_p=0.0):
        super(MLPMixerLayer, self).__init__()

        self.dim = dim
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.drop_n = drop_n
        self.drop_p = drop_p

        self.norm1 = RMSNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
        
        self.norm2 = RMSNorm(num_patches)
        self.patches_mlp = nn.Sequential(
            nn.Linear(num_patches, hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Linear(hidden_dim, num_patches)
        )

        self.skipper1 = ScaleSkip1D(num_patches, drop_n=drop_n)
        self.skipper2 = ScaleSkip1D(dim, drop_n=drop_n)

    def forward(self, x):
        out = self.norm1(x)
        out = self.skipper1(out, self.token_mlp(out))
        out = out.transpose(1, 2)

        out = self.norm2(out)
        out = self.skipper2(out, self.patches_mlp(out))
        out = out.transpose(1, 2)

        return out


class MLPMixer(nn.Module):
    def __init__(self,
        chw,
        output_dim,
        patch_size,
        dim,
        depth,
        embed_dim,
        expansion=4,
        drop_n=0.1,
        drop_p=0.1,
        clamp_output=False,
        clamp_min=0.0,
        clamp_max=1.0,
    ):
        super(MLPMixer, self).__init__()
        self.chw = chw
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.embed_dim = embed_dim
        self.expansion = expansion
        self.drop_n = drop_n
        self.drop_p = drop_p
        self.clamp_output = clamp_output
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.std = .02
        self.expansion = expansion

        self.num_patches = (chw[1] // patch_size) * (chw[2] // patch_size)

        self.stem_channels = (dim // (patch_size ** 2)) * self.expansion
        self.stem = nn.Sequential(
            CNNBlock(chw[0], self.stem_channels, drop_n=0.0, drop_p=0.0),
            CNNBlock(self.stem_channels, self.stem_channels * 2, drop_n=drop_n, drop_p=drop_p),
            CNNBlock(self.stem_channels * 2, self.stem_channels, drop_n=drop_n, drop_p=drop_p),
        )

        import pdb; pdb.set_trace()
        self.projection = nn.Linear(self.stem_channels * (patch_size ** 2), dim)
        self.mixer_layers = nn.ModuleList([
            MLPMixerLayer(dim, self.num_patches, embed_dim, drop_n=drop_n, drop_p=drop_p) for _ in range(depth)
        ])
        self.reproject = nn.Sequential(
            nn.Linear(dim, int(self.stem_channels * (patch_size ** 2))),
            RMSNorm(int(self.stem_channels * (patch_size ** 2))),
            nn.ReLU6(),
        )
        self.skipper = ScaleSkip2D(self.stem_channels, drop_p=drop_p)

        self.head = nn.Sequential(
            CNNBlock(self.stem_channels, self.stem_channels // 2, drop_n=drop_n, drop_p=drop_p),
            nn.Conv2d(self.stem_channels // 2, self.output_dim, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = self.std
            torch.nn.init.trunc_normal_(m.weight, std=std, a=-std * 2, b=std * 2)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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
    
    def unpatchify_batch(self, patches, channels=0):
        _, height, width = self.chw
        channels = channels if channels > 0 else self.stem_channels
        patch_size = self.patch_size
        batch_size, _n_patches, _ = patches.shape
        
        patches = patches.reshape(batch_size, height // patch_size, width // patch_size, patch_size, patch_size, channels)
        patches = patches.swapaxes(2, 3)
        patches = patches.reshape(batch_size, height, width, channels)
        patches = patches.swapaxes(1, -1)

        return patches
    
    def forward_stem(self, x):
        x = self.stem(x)

        return x
    
    def forward_project(self, x):
        x = self.patchify_batch(x)
        x = self.projection(x)

        return x

    def forward_trunc(self, x):
        for layer in self.mixer_layers:
            x = layer(x)

        return x

    def forward_reproject(self, x):
        x = self.reproject(x)
        x = self.unpatchify_batch(x)

        return x
    
    def forward_head(self, x):
        x = self.head(x)

        return x

    def forward(self, identity):
        skip = self.forward_stem(identity)
        x = self.forward_project(skip)
        x = self.forward_trunc(x)
        x = self.forward_reproject(x)
        x = self.skipper(skip, x)
        x = self.forward_head(x)

        if self.clamp_output:
            x = torch.clamp(x, self.clamp_min, self.clamp_max)
        
        return x


if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 32
    CHANNELS = 10
    HEIGHT = 64
    WIDTH = 64

    model = MLPMixer(
        chw=(10, 64, 64),
        output_dim=10,
        patch_size=4,
        embed_dim=256,
        expansion=4,
        dim=128,
        depth=5,
        clamp_output=False,
        clamp_min=0.0,
        clamp_max=1.0,
    )
    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )
