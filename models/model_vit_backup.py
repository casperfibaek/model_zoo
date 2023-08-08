# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import torch
import torch.nn as nn

from utils_transformers import get_2d_sincos_pos_embed
from timm.models.vision_transformer import PatchEmbed, Block


def unpatchify_batch(patches, shape, patch_size):
    """ Unpatchify without loops and copies. """
    channels, height, width = shape
    batch_size, _n_patches, _ = patches.shape
    
    patches = patches.reshape(batch_size, height // patch_size, width // patch_size, patch_size, patch_size, channels)
    patches = patches.swapaxes(2, 3)
    patches = patches.reshape(batch_size, height, width, channels)
    patches = patches.swapaxes(1, -1)

    return patches

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


class VitEncoder(nn.Module):
    """ VisionTransformer """
    def __init__(self,
        chw:tuple=(10, 64, 64),
        patch_size:int=16,
        embed_dim:int=768,
        depth:int=3,
        num_heads:int=16,
        mlp_ratio:float=4.,
        norm_layer:nn.Module=nn.LayerNorm,
    ):
        super().__init__()

        self.chw = chw
        self.emded_dim = embed_dim

        self.patch_embed = PatchEmbed(chw[1], patch_size, chw[0], embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)

        # Add positional embedding without class token
        x = x + self.pos_embed[:, 1:, :]

        # Append class token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x


class Vit_basic(nn.Module):
    def __init__(self, chw:tuple=(10, 64, 64), out_chans:int=1, patch_size:int=16,
                 embed_dim:int=512, depth:int=1, num_heads:int=16,
                 mlp_ratio:float=4., norm_layer:nn.Module=nn.LayerNorm):
        super().__init__()

        self.chw = chw
        self.patch_size = patch_size
        self.out_chans = out_chans

        self.vit_encoder = VitEncoder(chw, patch_size, embed_dim, depth, num_heads, mlp_ratio, norm_layer)
        self.decoder_pred = nn.Linear(embed_dim, int(out_chans * patch_size ** 2), bias=True)

    def forward(self, x):
        x = self.vit_encoder(x)
        x = self.decoder_pred(x)
        
        x = x[:, 1:, :] # remove cls token

        x = unpatchify_batch(x, (self.out_chans, self.chw[1], self.chw[2]), self.patch_size)
        x = torch.clamp(x, 0.0, 100.0)

        return x
