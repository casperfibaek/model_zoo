import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F


def patchify(images, n_patches):
    n, c, h, w = images.shape
    device = images.device

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2).to(device)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()

    return patches


def unpatchify(n, c, h, w, n_patches, tensors):
    device = tensors.device
    images = torch.zeros(n, c, h, w).to(device)
    patch_size = h // n_patches

    for idx, tensor in enumerate(tensors):
        patch_idx = 0
        for i in range(n_patches):
            for j in range(n_patches):
                patch = tensor[patch_idx].unflatten(dim=0, sizes=(c, patch_size, patch_size))
                images[idx, :, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size] = patch
                patch_idx = patch_idx + 1
    
    return images


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            if j % 2 == 0:
                result[i][j] = np.sin(i / (10000 ** (j / d)))
            else:
                result[i][j] = np.cos(i / (10000 ** ((j - 1) / d)))

    return result


class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                # splits patch image features into (n, d_head) inputs (one for each head)
                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5)) # @ is matrix multiplication
                seq_result.append(attention @ v)

            result.append(torch.hstack(seq_result))

        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))

        return out


class MyViT(nn.Module):
  def __init__(self, chw=(9, 128, 128), n_patches=8, n_blocks=2, hidden_d=1024, n_heads=2):
      # Super constructor
      super(MyViT, self).__init__()

      # Attributes
      self.chw = chw # (C, H, W)
      self.n_patches = n_patches
      self.n_blocks = n_blocks
      self.n_heads = n_heads
      self.hidden_d = hidden_d
      self.norm_layer = nn.LayerNorm(hidden_d)

      assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
      assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
      self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

      # 1) Linear mapper
      self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
      self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

      # 2) Learnable classifiation token
      self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

      # 3) Positional embedding
      self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches ** 2 + 1,
                                                                           self.hidden_d)))
      self.pos_embed.requires_grad = False

      # 4) Transformer encoder blocks
      self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

  def forward(self, images):
      patches = patchify(images, self.n_patches)
      tokens = self.linear_mapper(patches)

      # Adding classification token to the tokens # more efficent way to do it?
      # classification token is put as the first token of each sequence
      tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

      # Adding positional embedding
      pos_embed = self.pos_embed.repeat(tokens.shape[0], 1, 1)
      out = tokens + pos_embed

      # Transformer Blocks
      for block in self.blocks:
          out = block(out)

      out = self.norm_layer(out)
      return out

class MyViT_decoder(nn.Module):
  def __init__(self, input_d, n_patches=8, n_blocks=2, hidden_d=1024, n_heads=2):
      # Super constructor
      super(MyViT_decoder, self).__init__()

      # Attributes
      self.input_d = input_d
      self.hidden_d = hidden_d
      self.n_patches = n_patches
      self.n_blocks = n_blocks
      self.n_heads = n_heads
      self.norm_layer = nn.LayerNorm(self.hidden_d)


      # 1) Linear mapper
      self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

      # 3) Positional embedding
      self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches ** 2 + 1,
                                                                           self.hidden_d)))
      self.pos_embed.requires_grad = False

      # 4) Transformer encoder blocks
      self.blocks = nn.ModuleList([MyViTBlock(self.hidden_d, n_heads) for _ in range(n_blocks)])

  def forward(self, x):
      tokens = self.linear_mapper(x)

      # Adding positional embedding
      pos_embed = self.pos_embed.repeat(tokens.shape[0], 1, 1)
      out = tokens + pos_embed

      # Transformer Blocks
      for block in self.blocks:
          out = block(out)

      out = self.norm_layer(out)
      # remove classification token put as the first token of each sequence
      out = out[:, 1:, :]
      return out

class ViT(nn.Module):
    def __init__(self, chw=(10, 64, 64), n_patches=4, n_blocks=2, hidden_d=768, n_heads=12, clamp_output=True, clamp_output_min=0.0, clamp_output_max=1.0):
        # Super constructor
        super(ViT, self).__init__()
        self.chw = chw
        self.n_patches = n_patches
        self.clamp_output = clamp_output
        self.clamp_output_min = clamp_output_min
        self.clamp_output_max = clamp_output_max
        patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        self.vit_encoder = MyViT(chw, n_patches, n_blocks, hidden_d, n_heads)
        self.decoder_pred = nn.Linear(hidden_d, int(1 * patch_size[0] ** 2), bias=True)

    def forward(self, x):
        x = self.vit_encoder(x)
        x = x[:, 1:, :]
        x = self.decoder_pred(x)

        if self.clamp_output:
            x = torch.clamp(x, self.clamp_output_min, self.clamp_output_max)

        return x

class autoencoderViT(nn.Module):

    def __init__(self, chw=(10, 128, 128), n_patches=8, n_blocks=2, hidden_d=768, n_heads=12,
                 decoder_n_blocks=2, decoder_hidden_d=512, decoder_n_heads=16, clamp_output=True, clamp_output_min=0.0, clamp_output_max=1.0):
        # Super constructor
        super(autoencoderViT, self).__init__()
        self.chw = chw
        self.n_patches = n_patches
        self.clamp_output = clamp_output
        self.clamp_output_min = clamp_output_min
        self.clamp_output_max = clamp_output_max
        patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        self.vit_encoder = MyViT(chw, n_patches, n_blocks, hidden_d, n_heads)
        self.vit_decoder = MyViT_decoder(input_d=hidden_d, n_patches=n_patches, n_blocks=decoder_n_blocks, hidden_d=decoder_hidden_d, n_heads=decoder_n_heads)
        self.decoder_pred = nn.Linear(decoder_hidden_d, int(1 * patch_size[0] ** 2), bias=True)  # decoder to patch


    def forward(self, x):
        x = self.vit_encoder(x)
        x = self.vit_decoder(x)
        x = self.decoder_pred(x)

        if self.clamp_output:
            x = torch.clamp(x, self.clamp_output_min, self.clamp_output_max)

        return x

class vit_mse_losses(nn.Module):
    def __init__(self, n_patches=8) -> None:
        super().__init__()
        self.n_patches = n_patches

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = patchify(target, n_patches=self.n_patches)

        return F.mse_loss(input, target, reduction='mean')
