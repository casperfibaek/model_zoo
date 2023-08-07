import numpy as np

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed
import buteo as beo
import os


def patchify_batch(images, patch_size=16):
    """ Patchify without loops and copies. """
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


def unpatchify_batch(patches, shape, patch_size):
    """ Unpatchify without loops and copies. """
    _, channels, height, width = shape
    batch_size, _n_patches, _ = patches.shape
    
    patches = patches.reshape(batch_size, height // patch_size, width // patch_size, patch_size, patch_size, channels)
    patches = patches.swapaxes(2, 3)
    patches = patches.reshape(batch_size, height, width, channels)
    patches = patches.swapaxes(1, -1)

    return patches


class ViT(nn.Module):
    def __init__(self,
        bchw=(32, 10, 64, 64),
        output_dim=1,
        patch_size=16,
        embed_dim=512,
        n_layers=2,
        n_heads=8,
    ):
        super(ViT, self).__init__()
        self.bchw = bchw
        self.patch_size = patch_size
        self.n_patches  = bchw[2] // patch_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.output_dim = output_dim

        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

        assert bchw[2] % patch_size == 0, "Image size must be divisible by the patch size."
        assert bchw[3] == bchw[2], "Image must be square."
        assert embed_dim % n_heads == 0, "Hidden dimension must be divisible by the number of heads."

        self.patch_embed = PatchEmbed(img_size=bchw[2], patch_size=patch_size, in_chans=bchw[1], embed_dim=embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, (self.n_patches ** 2) + 1, embed_dim) * .02)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.blocks = [Block(embed_dim, n_heads) for _ in range(self.n_layers)]

        self.output = nn.Linear(self.embed_dim, bchw[1] * patch_size * patch_size)
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(bchw[1], bchw[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(bchw[1]),
            nn.Conv2d(bchw[1], self.output_dim, kernel_size=1),
        )

    def forward(self, x):
        x = self.patch_embed(x)

        # Add class token
        x = torch.cat([self.class_token.repeat(x.shape[0], 1, 1), x], dim=1)

        # Add Positional Embeddings
        x = x + self.position_embedding

        # Pass through Transformer Encoder
        for block in self.blocks:
            x = block(x)

        # Return to the orignal shape
        x = self.output(x)
        x = x[:, 1:, :] # Remove class token
        x = unpatchify_batch(x, self.bchw, self.patch_size)

        # Collapse dimensions
        x = self.output_conv(x)

        x = torch.clamp(x, 0.0, 100.0)

        return x


if __name__ == "__main__":
    FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/model_zoo/data/images/"
    PATCH_SIZE = 8

    image = os.path.join(FOLDER, "naestved_s2.tif")
    arr = beo.raster_to_array(image, filled=True, fill_value=0.0, cast=np.float32) / 10000.0
    arr = beo.array_to_patches(arr, 64)
    arr = beo.channel_last_to_first(arr)
    tensor = torch.from_numpy(arr)
    batch = tensor[:32, ::]

    model = ViT(bchw=batch.shape, patch_size=PATCH_SIZE)
    output = model(batch)
