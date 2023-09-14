import torch
import torch.nn as nn


class ScaleSkip2D(nn.Module):
    def __init__(self, channels, drop_p=0.1):
        super(ScaleSkip2D, self).__init__()
        self.channels = channels
        self.drop_p = drop_p

        self.y_skipscale = nn.Parameter(torch.ones(1, self.channels, 1, 1))
        self.y_skipbias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
        self.y_dropout = nn.Dropout2d(drop_p) if drop_p > 0. else nn.Identity()

        self.x_skipscale = nn.Parameter(torch.ones(1, self.channels, 1, 1))
        self.x_skipbias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
        self.x_dropout = nn.Dropout2d(drop_p) if drop_p > 0. else nn.Identity()

        torch.nn.init.normal_(self.y_skipscale, mean=1.0, std=.02)
        torch.nn.init.normal_(self.y_skipbias, mean=0.0, std=.02)
        torch.nn.init.normal_(self.x_skipscale, mean=1.0, std=.02)
        torch.nn.init.normal_(self.x_skipbias, mean=0.0, std=.02)

    def forward(self, x, skip_connection):
        y = self.y_skipscale * self.y_dropout(skip_connection) + self.y_skipbias
        x = self.x_skipscale * self.x_dropout(x) + self.x_skipbias

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
    def __init__(self, scale_value=1.0, bias_value=0.0, scale_learnable=True, bias_learnable=True, inplace=False):
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

        if self.apply_residual:
            if x.size(1) != identity.size(1):
                identity = self.norm4(self.match_channels(identity))

            x = identity + x

        x = self.activation3(x)

        return x


class MLPMixerLayer(nn.Module):
    def __init__(self,
        embed_dims,
        patch_size=16,
        chw=(10, 64, 64),
        expansion=2,
        drop_n=0.0,
        offset=False,
    ):
        super(MLPMixerLayer, self).__init__()
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.chw = chw
        self.expansion = expansion
        self.offset = offset

        if self.offset:
            self.chw = (chw[0], chw[1] + self.patch_size, chw[2] + self.patch_size)

        self.num_patches_height = self.chw[1] // self.patch_size
        self.num_patches_width = self.chw[2] // self.patch_size
        self.num_patches = self.num_patches_height * self.num_patches_width
        self.tokens = round((self.chw[1] * self.chw[2]) / self.num_patches)

        star = True
        rms = True

        self.mix_channel = nn.Sequential(
            RMSNorm(self.embed_dims) if rms else nn.LayerNorm(self.embed_dims),
            nn.Linear(self.embed_dims, int(self.embed_dims * self.expansion)),
            StarReLU() if star else nn.ReLU(),
            nn.Linear(int(self.embed_dims * self.expansion), self.embed_dims),
            nn.Dropout(drop_n) if drop_n > 0. else nn.Identity(),
        )

        self.mix_patch = nn.Sequential(
            RMSNorm(self.num_patches) if rms else nn.LayerNorm(self.num_patches),
            nn.Linear(self.num_patches, int(self.num_patches * self.expansion)),
            StarReLU() if star else nn.ReLU(),
            nn.Linear(int(self.num_patches * self.expansion), self.num_patches),
            nn.Dropout(drop_n) if drop_n > 0. else nn.Identity(),
        )

        self.mix_token = nn.Sequential(
            RMSNorm(self.tokens) if rms else nn.LayerNorm(self.tokens),
            nn.Linear(self.tokens, int(self.tokens * self.expansion)),
            StarReLU() if star else nn.ReLU(),
            nn.Linear(int(self.tokens * self.expansion), self.tokens),
            nn.Dropout(drop_n) if drop_n > 0. else nn.Identity(),
        )

        self.mix_local = nn.Sequential(
            nn.BatchNorm3d(self.num_patches),
            nn.Conv3d(self.num_patches, self.num_patches, kernel_size=3, stride=1, padding="same", groups=self.num_patches),
            StarReLU() if star else nn.ReLU(),
        )


    def patchify_batch(self, tensor):
        B, C, _H, _W = tensor.shape
        patch_size = self.patch_size
        num_patches_height = self.num_patches_height
        num_patches_width = self.num_patches_width
        num_patches = self.num_patches

        # Reshape and extract patches
        reshaped = tensor.reshape(B, C, num_patches_height, patch_size, num_patches_width, patch_size)
        transposed = reshaped.permute(0, 2, 4, 1, 3, 5)
        final_patches = transposed.reshape(B, num_patches, C, patch_size ** 2)

        return final_patches


    def unpatchify_batch(self, patches):
        B, _P, C, _T = patches.shape
        _C, H, W = self.chw
        patch_size = self.patch_size
        num_patches_height = self.num_patches_height
        num_patches_width = self.num_patches_width


        # Reverse the patchify process
        reshaped = patches.reshape(B, num_patches_height, num_patches_width, C, patch_size, patch_size)
        transposed = reshaped.permute(0, 3, 1, 4, 2, 5)
        final_tensor = transposed.reshape(B, C, H, W)

        return final_tensor


    def forward(self, x):
        B, C, _H, _W = x.shape

        if self.offset:
            x = torch.nn.functional.pad(x, (self.patch_size // 2, self.patch_size // 2, self.patch_size // 2, self.patch_size // 2), mode="constant", value=0)

        x = self.patchify_batch(x)
        # x: batch, num_Patches, channels, patch_Size * patch_Size

        mix_channel = x.transpose(2, 3)
        mix_channel = self.mix_channel(mix_channel)
        mix_channel = mix_channel.transpose(2, 3)
        x = x + mix_channel

        mix_patch = x.transpose(1, 3)
        mix_patch = self.mix_patch(mix_patch)
        mix_patch = mix_patch.transpose(1, 3)
        x = x + mix_patch

        mix_token = self.mix_token(x)
        x = x + mix_token

        mix_local = x.reshape(B, self.num_patches, C, self.patch_size, self.patch_size)
        mix_local = self.mix_local(mix_local)
        mix_local = mix_local.reshape(B, self.num_patches, C, int(self.patch_size ** 2))
        x = x + mix_local

        x = self.unpatchify_batch(x)
        # x: Batch, Channels, Height, Width

        if self.offset:
            x = x[:, :, self.patch_size // 2:-self.patch_size // 2, self.patch_size // 2:-self.patch_size // 2]

        return x


class MLPMixer(nn.Module):
    def __init__(self,
        chw,
        output_dim,
        embedding_dims=[16, 16, 16],
        patch_sizes=[8, 4, 2],
        overlaps=[False, False, False],
        expansion=2,
        drop_n=0.0,
        drop_p=0.0,
    ):
        super(MLPMixer, self).__init__()
        self.chw = chw
        self.output_dim = output_dim
        self.embedding_dims = embedding_dims
        self.patch_sizes = patch_sizes
        self.overlaps = overlaps
        self.expansion = expansion
        self.drop_n = drop_n
        self.drop_p = drop_p
        self.std = .05

        assert isinstance(self.embedding_dims, list), "embedding_dims must be a list."
        assert isinstance(self.patch_sizes, list), "patch_sizes must be a list."
        assert isinstance(self.overlaps, list), "overlaps must be a list."
        assert len(self.embedding_dims) == len(self.patch_sizes) == len(self.overlaps), "embedding_dims, patch_sizes, and overlaps must be the same length."

        self.stem = nn.Sequential(
            # nn.Conv2d(self.chw[0], self.embedding_dims[0], 1, padding=0),
            CNNBlock(chw[0], self.embedding_dims[0]),
            CNNBlock(self.embedding_dims[0], self.embedding_dims[0]),
            CNNBlock(self.embedding_dims[0], self.embedding_dims[0]),
        )

        self.mixer_layers = []
        self.matcher_layers = []
        self.skip_layers = [nn.Identity()]
        self.skip_layers_2 = [nn.Identity()]
        for i, v in enumerate(patch_sizes):
            if self.embedding_dims[i] != self.embedding_dims[i - 1] and i < len(patch_sizes) - 1 and i != 0:
                self.matcher_layers.append(
                    nn.Conv2d(self.embedding_dims[i - 1], self.embedding_dims[i], 1, padding=0)
                )
            else:
                self.matcher_layers.append(nn.Identity())

            self.mixer_layers.append(
                MLPMixerLayer(
                    self.embedding_dims[i],
                    patch_size=v,
                    chw=chw,
                    expansion=self.expansion,
                    drop_n=drop_n,
                    offset=self.overlaps[i],
                )
            )

            if i != 0:
                self.skip_layers.append(
                    ScaleSkip2D(self.embedding_dims[i], drop_p=drop_p)
                )
                if self.embedding_dims[i] != self.embedding_dims[0]:
                    self.skip_layers_2.append(
                        nn.Conv2d(self.embedding_dims[0], self.embedding_dims[i], 1, padding=0)
                    )
                else:
                    self.skip_layers_2.append(nn.Identity())

        self.matcher_layers = nn.ModuleList(self.matcher_layers)
        self.mixer_layers = nn.ModuleList(self.mixer_layers)
        self.skip_layers = nn.ModuleList(self.skip_layers)
        self.skip_layers_2 = nn.ModuleList(self.skip_layers_2)

        self.head = nn.Sequential(
            CNNBlock(self.embedding_dims[-1], self.embedding_dims[-1]),
            nn.Conv2d(self.embedding_dims[-1], self.output_dim, 1, padding=0),
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
        x = skip

        for i, layer in enumerate(self.mixer_layers):
            x = self.matcher_layers[i](x)
            skip_match = self.skip_layers_2[i](skip)

            # Only add skip-connections after the first layer
            if i != 0:
                x = self.skip_layers[i](layer(x), skip_match)
            else:
                x = layer(x)

        x = self.head(x)

        x = torch.clamp(x, 0.0, 100.0)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 16
    CHANNELS = 10
    HEIGHT = 64
    WIDTH = 64

    torch.set_default_device("cuda")

    model = MLPMixer(
        chw=(10, 64, 64),
        output_dim=1,
        # embedding_dims=32,
        # expansion=2,
        # drop_n=0.0,
        # drop_p=0.0,
    )
    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )
