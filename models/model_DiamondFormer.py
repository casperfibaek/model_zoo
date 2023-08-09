import torch
import torch.nn as nn


class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0, scale_learnable=True, bias_learnable=True, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return (self.scale * (self.relu(x) ** 2)) + self.bias


class MLPMixerLayer(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim, drop=0.2):
        super(MLPMixerLayer, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            StarReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
        self.norm2 = nn.LayerNorm(num_patches)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_patches, hidden_dim),
            StarReLU(),
            nn.Linear(hidden_dim, num_patches)
        )

        self.dropout1 = nn.Dropout(drop) if drop > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop) if drop > 0. else nn.Identity()
        
    def forward(self, x):
        out = self.norm1(x)
        out = out + self.dropout1(self.channel_mlp(out))
        out = out.transpose(1, 2)

        out = self.norm2(out)
        out = out + self.dropout2(self.token_mlp(out))
        out = out.transpose(1, 2)
        
        return out


class DiamondFormer(nn.Module):
    def __init__(self,
        chw,
        output_dim=1,
        patch_size=16,
        depth=3,
        embed_dim=512,
        clamp_output=False,
        clamp_min=0.0,
        clamp_max=1.0,
    ):
        super(DiamondFormer, self).__init__()
        torch.set_default_device("cuda")

        self.chw = chw
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.clamp_output = clamp_output
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.std = 0.02
        self.input_dim = chw[0]
        self.input_size = chw[1]
        self.depth = depth
        self.embed_dim = embed_dim

        self.sizes = [self.input_size // (2 ** (i + 1)) for i in reversed(range(self.depth))]

        self.patch_sizes = [self.patch_size]
        for i in range(self.depth):
            self.patch_sizes.append(max(self.patch_sizes[-1] // 2, 2))
        self.patch_sizes = self.patch_sizes[::-1]

        self.pools = []
        for i in self.sizes:
            self.pools.append(
                nn.Conv2d(
                    self.input_dim,
                    self.input_dim,
                    kernel_size=self.input_size // i + 1,
                    stride=self.input_size // i,
                    padding=(self.input_size // i) // 2,
                    groups=self.input_dim
                )
            )
        self.pools.append(
            nn.Conv2d(
                self.input_dim,
                self.input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.input_dim,
            )
        )
        self.pools = nn.ModuleList(self.pools)
        self.sizes.append(self.input_size)
        self.projections = [nn.Linear((p ** 2) * chw[0], embed_dim) for p in self.patch_sizes]
        self.reprojection = nn.Linear(embed_dim, (self.patch_sizes[-1] ** 2) * self.output_dim)
        self.pos_embeddings = [nn.Parameter(torch.zeros(1, (s // p) ** 2, embed_dim)) for s, p in zip(self.sizes, self.patch_sizes)]
        self.blocks = nn.ModuleList([
            nn.Sequential(*[
                MLPMixerLayer(embed_dim, (s // p) ** 2, embed_dim) for _ in range(1)
            ]) for s, p in zip(self.sizes, self.patch_sizes)
        ])

    def patchify_batch(self, images, patch_size):
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
    
    def unpatchify_batch(self, patches, shape, patch_size):
        channels, height, width = shape
        batch_size, _n_patches, _ = patches.shape
        
        patches = patches.reshape(batch_size, height // patch_size, width // patch_size, patch_size, patch_size, channels)
        patches = patches.swapaxes(2, 3)
        patches = patches.reshape(batch_size, height, width, channels)
        patches = patches.swapaxes(1, -1)

        return patches

    def forward(self, x):
        identity = x
        block = x
        for ii, pp in enumerate(self.pools):
            pooled = pp(x)
            patches = self.patchify_batch(pooled, self.patch_sizes[ii])
            projection = self.projections[ii](patches)
            
            patch_embeddings = projection + self.pos_embeddings[ii]
            if ii > 0:
                patch_embeddings = patch_embeddings + block

            block = self.blocks[ii](patch_embeddings)

        reprojection = self.reprojection(block)
        spatial = self.unpatchify_batch(reprojection, (self.output_dim, self.chw[1], self.chw[2]), self.patch_sizes[-1])
        x = spatial

        if self.clamp_output:
            x = torch.clamp(x, self.clamp_min, self.clamp_max)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 32
    CHANNELS = 10
    HEIGHT = 64
    WIDTH = 64

    torch.set_default_device("cuda")

    model = DiamondFormer(
        (CHANNELS, HEIGHT, WIDTH),
        output_dim=1,
        patch_size=16,
    )

    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )
