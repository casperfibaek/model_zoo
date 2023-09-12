# Standard Library
import os
from glob import glob

# External Libraries
import buteo as beo
import numpy as np

# PyTorch
import torch
from torch.utils.data import DataLoader


def callback_preprocess(x, y):
    x_norm = np.empty_like(x, dtype=np.float32)
    np.divide(x, 10000.0, out=x_norm)

    y = y.astype(np.float32, copy=False)

    return x_norm, y

def callback_postprocess_encoder(x, y):
    x = beo.channel_last_to_first(x)
    y = np.array([np.sum(y) / y.size], dtype=np.float32)

    return torch.from_numpy(x), torch.from_numpy(y)

def callback_postprocess_decoder(x, y):
    x = beo.channel_last_to_first(x)
    y = beo.channel_last_to_first(y)

    return torch.from_numpy(x), torch.from_numpy(y)

def callback_encoder(x, y):
    x, y = callback_preprocess(x, y)
    x, y = callback_postprocess_encoder(x, y)

    return x, y

def callback_decoder(x, y):
    x, y = callback_preprocess(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y


masking_layer = beo.MaskImages(
    masking_functions=[
        beo.MaskPixels3D(p=0.05),
        beo.MaskElipse3D(p=0.33, max_height=0.25, max_width=0.25),
        # beo.MaskLines3DBezier(p=0.025),
        beo.MaskRectangle3D(p=0.2, max_height=0.25, max_width=0.25),
        beo.MaskChannels(p=0.1, max_channels=5),
    ],
    method=3,
    val=0,
    max_val=1.0,
    min_val=0.0,
    per_channel=True,
    inplace=False,
)

def callback_preprocess_ae(x, _y):
    x_norm = np.empty_like(x, dtype=np.float32)
    np.divide(x, 10000.0, out=x_norm)

    return x_norm, x_norm

def callback_postprocess_ae(x, y):
    x = masking_layer(x)

    x = beo.channel_last_to_first(x)
    y = beo.channel_last_to_first(y)

    return torch.from_numpy(x), torch.from_numpy(y)


def callback_ae(x, y):
    x_norm = np.empty_like(x, dtype=np.float32)
    np.divide(x, 10000.0, out=x_norm)

    x = x_norm
    y = x_norm

    x = beo.channel_last_to_first(x)
    y = beo.channel_last_to_first(y)

    return torch.from_numpy(x), torch.from_numpy(y)


def load_data(*, x="s2", y="area", with_augmentations=False, num_workers=0, batch_size=16, folder="../data/patches/", encoder_only=False):
    """
    Loads the data from the data folder.
    """
    x_train = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*train_{x}.npy")))])
    y_train = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*train_label_{y}.npy")))])

    x_val = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*val_{x}.npy")))])
    y_val = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*val_label_{y}.npy")))])

    x_test = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*test_{x}.npy")))])
    y_test = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*test_label_{y}.npy")))])

    assert len(x_train) == len(y_train) and len(x_test) == len(y_test) and len(x_val) == len(y_val), "Lengths of x and y do not match."

    if with_augmentations:
        ds_train = beo.DatasetAugmentation(
            x_train, y_train,
            callback_pre_augmentation=callback_preprocess,
            callback_post_augmentation=callback_postprocess_encoder if encoder_only else callback_postprocess_decoder,
            augmentations=[
                beo.AugmentationRotationXY(p=0.2, inplace=True),
                beo.AugmentationMirrorXY(p=0.2, inplace=True),
                beo.AugmentationCutmix(p=0.2, inplace=True),
                beo.AugmentationChannelScale(p=0.2, inplace=True),
                # beo.AugmentationBlur(p=0.2, inplace=True),
                # beo.AugmentationSharpen(p=0.2, inplace=True),
                beo.AugmentationNoiseNormal(p=0.2, inplace=True),
                beo.AugmentationNoiseUniform(p=0.2, inplace=True),
            ],
        )
    else:
        ds_train = beo.Dataset(x_train, y_train, callback=callback_encoder if encoder_only else callback_decoder)

    ds_test = beo.Dataset(x_test, y_test, callback=callback_encoder if encoder_only else callback_decoder)
    ds_val = beo.Dataset(x_val, y_val, callback=callback_encoder if encoder_only else callback_decoder)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True, generator=torch.Generator(device='cuda'))
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=True, generator=torch.Generator(device='cuda'))
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=True, generator=torch.Generator(device='cuda'))

    return dl_train, dl_test, dl_val


def load_data_ae(*, x="s2", y="area", with_augmentations=False, num_workers=0, batch_size=16, folder="../data/patches/", encoder_only=False):
    """ Loads the data from the data folder. """
    x_train = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*train_{x}.npy")))])
    y_train = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*train_{x}.npy")))])

    x_val = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*val_{x}.npy")))])
    y_val = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*val_{x}.npy")))])

    x_test = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*test_{x}.npy")))])
    y_test = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(folder, f"*test_{x}.npy")))])

    assert len(x_train) == len(y_train) and len(x_test) == len(y_test) and len(x_val) == len(y_val), "Lengths of x and y do not match."

    if with_augmentations:
        ds_train = beo.DatasetAugmentation(
            x_train, y_train,
            callback_pre_augmentation=callback_preprocess_ae,
            callback_post_augmentation=callback_postprocess_ae,
            augmentations=[
                beo.AugmentationRotationXY(p=0.2, inplace=True),
                beo.AugmentationMirrorXY(p=0.2, inplace=True),
                beo.AugmentationCutmix(p=0.2, inplace=True),
                # beo.AugmentationBlur(p=0.2, inplace=True),
                # beo.AugmentationSharpen(p=0.2, inplace=True),
                beo.AugmentationChannelScale(p=0.2, inplace=True),
                beo.AugmentationNoiseNormal(p=0.2, inplace=True),
                beo.AugmentationNoiseUniform(p=0.2, inplace=True),
            ],
        )
    else:
        ds_train = beo.Dataset(x_train, y_train, callback=callback_ae)

    ds_test = beo.Dataset(x_test, y_test, callback=callback_ae)
    ds_val = beo.Dataset(x_val, y_val, callback=callback_ae)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True, generator=torch.Generator(device='cuda'))
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=True, generator=torch.Generator(device='cuda'))
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=True, generator=torch.Generator(device='cuda'))

    return dl_train, dl_test, dl_val
