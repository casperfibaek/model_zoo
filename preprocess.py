import os
from glob import glob
from time import sleep
from tqdm import tqdm

import buteo as beo
import numpy as np

FOLDER = "./data/images/"
FOLDER_OUT = "./data/patches/"

PROCESS_S1 = True
PROCESS_S2 = True
OVERLAPS = 1
PATCH_SIZE = 64
VAL_SPLIT = 0.1

# If empty, all locations are used.
TRAIN_LOCATIONS = []
TEST_LOCATIONS = ["naestved"]

for x in TRAIN_LOCATIONS:
    if x in TEST_LOCATIONS:
        raise ValueError("Location in both train and test.")

images = os.listdir(FOLDER)

total = 0
for img in images:
    if "_mask.tif" in img:
        total += 1

processed = 0
for img in images:
    if "mask" not in img:
        continue

    location = img.split("_")[0]

    path_mask = os.path.join(FOLDER, f"{location}_mask.tif")

    path_label_area = os.path.join(FOLDER, f"{location}_label_area.tif")
    path_label_volume = os.path.join(FOLDER, f"{location}_label_volume.tif")
    path_label_people = os.path.join(FOLDER, f"{location}_label_people.tif")
    path_label_lc = os.path.join(FOLDER, f"{location}_label_lc.tif")

    path_s1 = os.path.join(FOLDER, f"{location}_s1.tif")
    path_s2 = os.path.join(FOLDER, f"{location}_s2.tif")

    if location in TEST_LOCATIONS:
        pass
    elif location in TRAIN_LOCATIONS:
        pass
    elif len(TRAIN_LOCATIONS) == 0:
        pass
    else:
        processed += 1
        continue

    if location in TEST_LOCATIONS:
        TARGET = "test"
    else:
        TARGET = "train"

    mask_arr = beo.raster_to_array(path_mask)
    metadata = beo.raster_to_metadata(path_mask)

    if metadata["height"] < PATCH_SIZE or metadata["width"] < PATCH_SIZE:
        processed += 1
        continue

    initial_patches = beo.array_to_patches(
        mask_arr,
        tile_size=PATCH_SIZE,
        n_offsets=OVERLAPS,
        border_check=True,
    )

    # Mask any tiles with any masked values.
    mask_bool = ~np.any(initial_patches == 0, axis=(1, 2, 3))
    mask_bool_sum = mask_bool.sum()
    mask_random = np.random.permutation(np.arange(mask_bool_sum))
    idx_val = int(mask_bool_sum * (1 - VAL_SPLIT))

    if PROCESS_S1:
        patches_s1 = beo.array_to_patches(
            beo.raster_to_array(path_s1),
            tile_size=PATCH_SIZE,
            n_offsets=OVERLAPS,
            border_check=True,
        )[mask_bool][mask_random]

        if TARGET == "train":
            patches_s1_val = patches_s1[idx_val:]
            patches_s1 = patches_s1[:idx_val]

            np.save(os.path.join(FOLDER_OUT, f"{location}_val_s1.npy"), patches_s1_val)

        np.save(os.path.join(FOLDER_OUT, f"{location}_{TARGET}_s1.npy"), patches_s1)

    if PROCESS_S2:
        patches_s2 = beo.array_to_patches(
            beo.raster_to_array(path_s2),
            tile_size=PATCH_SIZE,
            n_offsets=OVERLAPS,
            border_check=True,
        )[mask_bool][mask_random]

        if TARGET == "train":
            patches_s2_val = patches_s2[idx_val:]
            patches_s2 = patches_s2[:idx_val]

            np.save(os.path.join(FOLDER_OUT, f"{location}_val_s2.npy"), patches_s2_val)

        np.save(os.path.join(FOLDER_OUT, f"{location}_{TARGET}_s2.npy"), patches_s2)

    patches_label_area = beo.array_to_patches(
        beo.raster_to_array(path_label_area),
        tile_size=PATCH_SIZE,
        n_offsets=OVERLAPS,
        border_check=True,
    )[mask_bool][mask_random]

    if TARGET == "train":
        patches_label_area_val = patches_label_area[idx_val:]
        patches_label_area = patches_label_area[:idx_val]

        np.save(os.path.join(FOLDER_OUT, f"{location}_val_label_area.npy"), patches_label_area_val)

    np.save(os.path.join(FOLDER_OUT, f"{location}_{TARGET}_label_area.npy"), patches_label_area)

    patches_label_volume = beo.array_to_patches(
        beo.raster_to_array(path_label_volume),
        tile_size=PATCH_SIZE,
        n_offsets=OVERLAPS,
        border_check=True,
    )[mask_bool][mask_random]

    if TARGET == "train":
        patches_label_volume_val = patches_label_volume[idx_val:]
        patches_label_volume = patches_label_volume[:idx_val]

        np.save(os.path.join(FOLDER_OUT, f"{location}_val_label_volume.npy"), patches_label_volume_val)

    np.save(os.path.join(FOLDER_OUT, f"{location}_{TARGET}_label_volume.npy"), patches_label_volume)

    patches_label_people = beo.array_to_patches(
        beo.raster_to_array(path_label_people),
        tile_size=PATCH_SIZE,
        n_offsets=OVERLAPS,
        border_check=True,
    )[mask_bool][mask_random]

    if TARGET == "train":
        patches_label_people_val = patches_label_people[idx_val:]
        patches_label_people = patches_label_people[:idx_val]

        np.save(os.path.join(FOLDER_OUT, f"{location}_val_label_people.npy"), patches_label_people_val)

    np.save(os.path.join(FOLDER_OUT, f"{location}_{TARGET}_label_people.npy"), patches_label_people)

    patches_label_lc = beo.array_to_patches(
        beo.raster_to_array(path_label_lc),
        tile_size=PATCH_SIZE,
        n_offsets=OVERLAPS,
        border_check=True,
    )[mask_bool][mask_random]

    if TARGET == "train":
        patches_label_lc_val = patches_label_lc[idx_val:]
        patches_label_lc = patches_label_lc[:idx_val]

        np.save(os.path.join(FOLDER_OUT, f"{location}_val_label_lc.npy"), patches_label_lc_val)

    np.save(os.path.join(FOLDER_OUT, f"{location}_{TARGET}_label_lc.npy"), patches_label_lc)

    assert patches_label_area.shape[0] == patches_label_volume.shape[0], "Number of patches do not match."
    assert patches_label_area.shape[0] == patches_label_people.shape[0], "Number of patches do not match."
    assert patches_label_area.shape[0] == patches_label_lc.shape[0], "Number of patches do not match."
    assert patches_label_area_val.shape[0] == patches_label_volume_val.shape[0], "Number of patches do not match."
    assert patches_label_area_val.shape[0] == patches_label_people_val.shape[0], "Number of patches do not match."
    assert patches_label_area_val.shape[0] == patches_label_lc_val.shape[0], "Number of patches do not match."

    if PROCESS_S1:
        assert patches_label_area.shape[0] == patches_s1.shape[0], "Number of patches do not match."
        assert patches_label_area_val.shape[0] == patches_s1_val.shape[0], "Number of patches do not match."

    if PROCESS_S2:
        assert patches_label_area.shape[0] == patches_s2.shape[0], "Number of patches do not match."
        assert patches_label_area_val.shape[0] == patches_s2_val.shape[0], "Number of patches do not match."

    processed += 1
    print(f"Processed {location}. ({processed}/{total}).")
