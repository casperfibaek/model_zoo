import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
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
    batch_size, channels, height, width = shape
    
    patches = patches.reshape(batch_size, height // patch_size, width // patch_size, patch_size, patch_size, channels)
    patches = patches.swapaxes(2, 3)
    patches = patches.reshape(batch_size, height, width, channels)
    patches = patches.swapaxes(1, -1)

    return patches


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            if j % 2 == 0:
                result[i][j] = np.sin(i / (10000 ** (j / d)))
            else:
                result[i][j] = np.cos(i / (10000 ** ((j - 1) / d)))

    return result


class ViT(nn.Module):
    def __init__(self,
        bchw=(32, 10, 64, 64),
        output_dim=1,
        patch_size=16,
        n_blocks=2,
        hidden_d=512,
        n_heads=8,
    ):
        super(ViT, self).__init__()
        self.bchw = bchw
        self.patch_size = patch_size
        self.n_patches  = bchw[2] // patch_size
        self.embed_dim = hidden_d
        self.n_heads = n_heads
        self.n_layers = n_blocks
        self.output_dim = output_dim

        assert bchw[2] % patch_size == 0, "Image size must be divisible by the patch size."
        assert bchw[3] == bchw[2], "Image must be square."
        assert hidden_d % n_heads == 0, "Hidden dimension must be divisible by the number of heads."

        # self.projection_conv = nn.Conv2d(bchw[1], self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.projection = nn.Linear(patch_size * patch_size * bchw[1], self.embed_dim)
        # self.position_embedding = nn.Parameter(torch.randn(1, self.n_patches ** 2, self.embed_dim))
        self.position_embedding = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches ** 2, self.embed_dim)))

        encoder_norm = nn.LayerNorm(self.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers, norm=encoder_norm)
        
        self.output = nn.Linear(self.embed_dim, bchw[1] * patch_size * patch_size)
        self.output_conv = nn.Conv2d(bchw[1], self.output_dim, kernel_size=1)
     

    def forward(self, x):
        # Apply the linear transformation to each patch
        x = patchify_batch(x, self.patch_size)

        # Apply linear transformation to each patch
        x = torch.cat([self.projection(p) for p in x.split(1, dim=1)], dim=1)

        # # Apply convolution to project patches
        # x = self.projection_conv(x)

        # # Reshape x to fit the expected shape of the transformer encoder
        # b, d, h, w = x.size()
        # x = x.permute(0, 2, 3, 1) # Rearrange dimensions
        # x = x.reshape(b, h * w, d)

        # Add Positional Embeddings
        x = x + self.position_embedding

        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)

        # Return to the orignal shape
        x = self.output(x)
        x = unpatchify_batch(x, self.bchw, self.patch_size)

        # Collapse dimensions
        x = self.output_conv(x)

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

    import pdb; pdb.set_trace()
