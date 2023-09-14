import numpy as np

img = np.arange(0, 1024).reshape(1, 1, 32, 32)

patch_size = 8

def patchify_batch(self, tensor):
    B, C, H, W = tensor.shape
    patch_size = self.patch_size

    # Calculate number of patches along height and width
    num_patches_height = H // patch_size
    num_patches_width = W // patch_size
    num_patches = num_patches_height * num_patches_width

    # Reshape and extract patches
    reshaped = tensor.reshape(B, C, num_patches_height, patch_size, num_patches_width, patch_size)
    transposed = reshaped.transpose(0, 2, 4, 1, 3, 5)
    final_patches = transposed.reshape(B, num_patches, C, patch_size ** 2)

    return final_patches


def unpatchify_batch(self, patches):
    B, num_patches, C, _ = patches.shape
    patch_size = self.patch_size

    # Calculate number of patches along height and width
    num_patches_height = num_patches_width = int((num_patches)**0.5)

    H = num_patches_height * patch_size
    W = num_patches_width * patch_size

    # Reverse the patchify process
    reshaped = patches.reshape(B, num_patches_height, num_patches_width, C, patch_size, patch_size)
    transposed = reshaped.transpose(0, 3, 1, 4, 2, 5)
    final_tensor = transposed.reshape(B, C, H, W)

    return final_tensor

patchify_batch(img)