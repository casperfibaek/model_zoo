""" Get simple EO data and labels for testing model architectures. """

import os
from typing import Union, List, Tuple, Optional

import buteo as beo
import numpy as np


def load_data(
    tile_size: int = 64,
    overlaps: int = 3,
    shuffle: bool = True,
    seed: Union[int, float] = 42,
    labels: Optional[List[str]] = None, # 'area', 'people', 'volume'
    data: Optional[List[str]] = None, # 's1', 's2'
    channel_last: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads data for testing.
    
    Parameters
    ----------
    tile_size : int, optional
        The size of the tiles, default: 64
    
    overlaps : int, optional
        The number of overlaps, default: 3
    
    shuffle : bool, optional
        Whether to shuffle the data, default: True
    
    seed : Union[int, float], optional
        The seed for the random number generator, default: 42

    labels : Optional[List[str]], optional
        The labels to load, default: ["area", "people", "volume"]

    data : Optional[List[str]], optional
        The data to load, default: ["s2"]
    
    channel_last : bool, optional
        Whether to use channel last, default: False
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (labels, data_s2, data_s1). Note: data_s1 is None if "s1" is not in data, etc..

    """
    folder = "./data/"

    if labels is None:
        labels = ["area", "people", "volume"]

    if data is None:
        data = ["s2"]

    # Load the labels
    arr_labels = beo.raster_to_array(
        [os.path.join(folder, f"CPH_LABEL_{label}.tif") for label in labels],
        filled=True,
        fill_value=0,
    )

    patches_labels = beo.array_to_patches(
        arr_labels,
        tile_size=tile_size,
        n_offsets=overlaps,
        border_check=True,
    )

    # Load S1 data if needed
    arr_data_s1 = None
    patches_data_s1 = None
    if "s1" in data:
        arr_data_s1 = beo.raster_to_array(
            [os.path.join(folder, "CPH_s1.tif") for d in data],
            filled=True,
            fill_value=0,
        )

        patches_data_s1 = beo.array_to_patches(
            arr_data_s1,
            tile_size=tile_size,
            n_offsets=overlaps,
            border_check=True,
        )

    # Load S2 data if needed
    arr_data_s2 = None
    patches_data_s2 = None
    if "s2" in data:
        arr_data_s2 = beo.raster_to_array(
            [os.path.join(folder, "CPH_s2.tif") for d in data],
            filled=True,
            fill_value=0,
        )

        patches_data_s2 = beo.array_to_patches(
            arr_data_s2,
            tile_size=tile_size,
            n_offsets=overlaps,
            border_check=True,
        )

    # Load the mask
    arr_mask = beo.raster_to_array(
        os.path.join(folder, "CPH_MASK.tif"),
        filled=True,
        fill_value=0,
    )

    patches_mask = beo.array_to_patches(
        arr_mask,
        tile_size=tile_size,
        n_offsets=overlaps,
        border_check=True,
    )

    mask_bool = ~np.any(patches_mask == 0, axis=(1, 2, 3))

    patches_labels = patches_labels[mask_bool]
    if "s1" in data:
        patches_data_s1 = patches_data_s1[mask_bool]
    if "s2" in data:
        patches_data_s2 = patches_data_s2[mask_bool]

    if shuffle:
        np.random.seed(seed)
        random_mask = np.random.permutation(len(patches_labels))

        patches_labels = patches_labels[random_mask]

        if "s1" in data:
            patches_data_s1 = patches_data_s1[random_mask]

        if "s2" in data:
            patches_data_s2 = patches_data_s2[random_mask]

    if not channel_last:
        if "s1" in data:
            patches_data_s1 = beo.channel_last_to_first(patches_data_s1)

        if "s2" in data:
            patches_data_s2 = beo.channel_last_to_first(patches_data_s2)

        patches_labels = beo.channel_last_to_first(patches_labels)

    if "s1" in data:
        assert patches_labels.shape[-2:] == patches_data_s1.shape[-2:], (
            f"{patches_labels.shape} != {patches_data_s1.shape}"
        )
        assert patches_labels.shape[0] == patches_data_s1.shape[0], (
            f"{patches_labels.shape} != {patches_data_s1.shape}"
        )

    if "s2" in data:
        assert patches_labels.shape[-2:] == patches_data_s2.shape[-2:], (
            f"{patches_labels.shape} != {patches_data_s2.shape}"
        )
        assert patches_labels.shape[0] == patches_data_s2.shape[0], (
            f"{patches_labels.shape} != {patches_data_s2.shape}"
        )

    return (
        patches_labels,
        patches_data_s2,
        patches_data_s1,
    )
