import torch
import torch.nn as nn

from timm.models.vision_transformer import Block


class ViT(nn.Module):
    """ Basic VisionTransformer """
    def __init__(self,
        chw:tuple=(10, 64, 64),
        output_dim:int=1,
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
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.num_patches = (chw[1] // patch_size) * (chw[2] // patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)); torch.nn.init.normal_(self.cls_token, std=.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self.projection = nn.Linear(patch_size * patch_size * chw[0], embed_dim)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)
        ])
        self.decoder = nn.Linear(embed_dim, int(output_dim * (patch_size ** 2)), bias=True)

        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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

    def forward(self, x):
        # Patch embedding and linear projection
        x = self.projection(self.patchify_batch(x))

        # Add positional embedding without class token
        x = x + self.pos_embed[:, 1:, :]

        # Append class token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block in self.blocks:
            x = block(x)

        # Return to original shape
        x = self.decoder(self.norm(x))

        # Remove cls token
        x = x[:, 1:, :]

        x = self.unpatchify_batch(x)
        x = torch.clamp(x, 0.0, 100.0)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 32
    CHANNELS = 10
    HEIGHT = 64
    WIDTH = 64

    model = ViT(
        chw=(CHANNELS, HEIGHT, WIDTH),
        output_dim=1,
        patch_size=8,
        embed_dim=768,
        depth=3,
        num_heads=16,
    )
    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )
