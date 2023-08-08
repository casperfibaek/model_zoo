import torch
import torch.nn as nn


class MLPMixerLayer(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim):
        super(MLPMixerLayer, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
        self.norm2 = nn.LayerNorm(num_patches)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_patches, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_patches)
        )
        
    def forward(self, x):
        out = self.norm1(x)
        out = out + self.channel_mlp(out)
        out = out.transpose(1, 2)

        out = self.norm2(out)
        out = out + self.token_mlp(out)
        out = out.transpose(1, 2)
        
        return out

class MLPMixer(nn.Module):
    def __init__(self, chw, output_dim, patch_size, dim, depth, embed_dim):
        super(MLPMixer, self).__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.chw = chw
        self.num_patches = (chw[1] // patch_size) * (chw[2] // patch_size)

        self.projection = nn.Linear(chw[0] * (patch_size ** 2), dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.mixer_layers = nn.ModuleList([
            MLPMixerLayer(dim, self.num_patches, embed_dim) for _ in range(depth)
        ])
        self.decoder = nn.Linear(dim, int(output_dim * (patch_size ** 2)), bias=True)

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
        x = self.projection(self.patchify_batch(x))
        x = x + self.pos_embed
        
        for layer in self.mixer_layers:
            x = layer(x)

        x = self.decoder(x)

        x = self.unpatchify_batch(x)
        x = torch.clamp(x, 0.0, 100.0)
        
        return x
    


if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 32
    CHANNELS = 10
    HEIGHT = 64
    WIDTH = 64

    model = MLPMixer(
        chw=(CHANNELS, HEIGHT, WIDTH),
        output_dim=1,
        patch_size=8,
        embed_dim=2048,
        dim=512,
        depth=8,
    )
    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )
