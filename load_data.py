""" Get simple EO data and labels for testing model architectures. """

import os
from typing import Union, List, Tuple, Optional, Callable
from glob import glob
import torch

import numpy as np

# Buteo
import buteo as beo
from buteo.array.utils_array import channel_first_to_last, channel_last_to_first
from buteo.ai.augmentation_funcs import (
    augmentation_mirror,
    augmentation_mirror_xy,
    augmentation_rotation,
    augmentation_rotation_xy,
    augmentation_noise_uniform,
    augmentation_noise_normal,
    augmentation_channel_scale,
    augmentation_contrast,
    augmentation_drop_channel,
    augmentation_drop_pixel,
    augmentation_blur,
    augmentation_blur_xy,
    augmentation_sharpen,
    augmentation_sharpen_xy,
    augmentation_cutmix,
    augmentation_mixup,
    augmentation_misalign
)


def load_data(
    labels: Optional[List[str]] = None, # 'area', 'volume', 'people', "lc"
    data: Optional[List[str]] = None, # 's1', 's2'
    folder: str = "./data/patches/",
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load data for the test-bed.
    Labels are stacked in the following order:
        [area, volume, people, lc]
    Data is stacked in the following order:
        [s2, s1]
    
    Returns {
        "train_s1": Optional[Generator],
        "train_s2": Optional[generator],
        "train_label_area": Optional[Generator],
        "train_label_volume": Optional[Generator],
        "train_label_people": Optional[Generator],
        "train_label_lc": Optional[Generator],

        "test_s1": Optional[Generator],
        "test_s2": Optional[Generator],
        "test_label_area": Optional[Generator],
        "test_label_volume": Optional[Generator],
        "test_label_people": Optional[Generator],
        "test_label_lc": Optional[Generator],
    }
    """
    obj = {
        "train_s1": [],
        "train_s2": [],
        "train_label_area": [],
        "train_label_volume": [],
        "train_label_people": [],
        "train_label_lc": [],

        "test_s1": [],
        "test_s2": [],
        "test_label_area": [],
        "test_label_volume": [],
        "test_label_people": [],
        "test_label_lc": [],
    }
    valid_labels = ["area", "volume", "people", "lc"]
    valid_data = ["s1", "s2"]

    if labels is None:
        labels = valid_labels

    if data is None:
        data = valid_data
    
    if len(labels) == 0:
        raise ValueError("No labels selected.")
    
    if len(data) == 0:
        raise ValueError("No data selected.")
    
    for label in labels:
        if label not in valid_labels:
            raise ValueError(f"Invalid label: {label}")
        
    images = sorted(glob(os.path.join(folder, f"*_label_*.npy")))
    for img in images:
        name_split = os.path.splitext(os.path.basename(img))[0].split("_")
        label_use = name_split[1]
        label_type = name_split[-1]

        obj[f"{label_use}_label_{label_type}"].append(np.load(img, mmap_mode="r"))

    for d in data:
        if d not in valid_data:
            raise ValueError(f"Invalid data: {d}")
        
    images = sorted(
        glob(os.path.join(folder, f"*_s1.npy")) + glob(os.path.join(folder, f"*_s2.npy")),
    )
    for img in images:
        name_split = os.path.splitext(os.path.basename(img))[0].split("_")
        label_use = name_split[1]
        label_type = name_split[-1]

        obj[f"{label_use}_{label_type}"].append(np.load(img, mmap_mode="r"))

    for key in obj:
        if len(obj[key]) == 0:
            obj[key] = None

    return obj


class MultiArray:
    """
    This is a class that takes in a tuple of list of arrays and glues them together
    without concatenating them. This is useful for when you have a large
    dataset that you want to load into memory, but you don't want to
    concatenate them because that would take up too much memory.

    The function works for saved numpy arrays loading using mmap_mode="r".

    Parameters
    ----------
    array_or_list_of_arrays : array or list of arrays
        The arrays to glue together. Can be len 1.

    Returns
    -------
    MultiArray
        The multi array. Lazily loaded.

    Examples
    --------
    ```python
    >>> from glob import glob
    >>> folder = "./data_patches/"

    >>> patches = sorted(glob(folder + "train*.npy"))
    >>> multi_array = MultiArray([np.load(p, mmap_mode="r") for p in patches])
    >>> single_image = multi_array[0]

    >>> print(single_image.shape)
    (128, 128, 10)
    >>> print(len(multi_array).shape)
    (32, 128, 128, 10)
    ```
    """
    def __init__(self, array_or_list_of_arrays):
        self.input_was_array = isinstance(array_or_list_of_arrays, (np.ndarray, np.memmap))
        self.list_of_arrays = array_or_list_of_arrays if not self.input_was_array else [array_or_list_of_arrays]

        self.one_deep = False
        if isinstance(self.list_of_arrays[0], (np.ndarray, np.memmap)):
            self.one_deep = True
            self.list_of_arrays = [self.list_of_arrays]

        if not self.one_deep:
            # All arrays in the list, must be the same length.
            first_len = 0
            for idx, lst in enumerate(self.list_of_arrays):
                if isinstance(lst, (np.ndarray, np.memmap)):
                    lst = [lst]

                assert isinstance(lst, list), "All arrays must be in a list."

                tup_len = 0
                for arr in lst:
                    tup_len += arr.shape[0]

                if idx == 0:
                    first_len = tup_len
                else:
                    assert tup_len == first_len, "All arrays must have the same length."

        # Calculate the cumulative sizes once, to speed up future calculations
        self.cumulative_sizes = np.cumsum([0] + [arr.shape[0] for arr in self.list_of_arrays[0]])
        self._shape = (self.cumulative_sizes[-1],) + self.list_of_arrays[0][0].shape[1:]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._get_single_item(idx)

        raise TypeError("Invalid argument type.")

    def _get_single_item(self, idx):
        if idx < 0:  # added this block to support negative indexing
            idx = self.__len__() + idx

        array_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
        array_tuple = [arrays[array_idx] for arrays in self.list_of_arrays]
        idx_within_array = idx - self.cumulative_sizes[array_idx]

        return_list = [array[idx_within_array] for array in array_tuple]

        if self.one_deep:
            return_list = return_list[0]

        if self.input_was_array:
            return_list = return_list[0]

        return return_list
    
    def __len__(self):
        return self.cumulative_sizes[-1]


class Dataset():
    """
    A dataset that does not apply any augmentations to the data.
    Allows a callback to be passed and can convert between
    channel formats.

    Parameters
    ----------
    X : np.ndarray
        The data to read.

    y : np.ndarray
        The labels for the data.

    callback : callable, optional
        A callback to apply to the data before returning.
        Inside the callback, the format will always be channel first.

    input_is_channel_last : bool, default: True
        Whether the data is in channel last format.

    output_is_channel_last : bool, default: False
        Whether the output should be in channel last format.

    Returns
    -------
    Dataset
        A dataset yielding batches of data. For Pytorch,
        convert the batches to tensors before ingestion.
    """
    def __init__(self,
        X: Union[Union[np.ndarray, MultiArray], List[Union[np.ndarray, MultiArray]]],
        y: Union[Union[np.ndarray, MultiArray], List[Union[np.ndarray, MultiArray]]],
        callback: Callable = None,
        input_is_channel_last: Optional[bool] = None,
        output_is_channel_last: Optional[bool] = None,
    ):
        self.x_train = X
        self.y_train = y

        self.callback = callback
        self.channel_last = False
        self.input_is_channel_last = input_is_channel_last
        self.output_is_channel_last = output_is_channel_last

        assert len(self.x_train) == len(self.y_train), "X and y must have the same length."
        assert input_is_channel_last is not None, "Input channel format must be specified."
        assert output_is_channel_last is not None, "Output channel format must be specified."

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        sample_x = self.x_train[index]
        sample_y = self.y_train[index]

        converted_x = False
        if not isinstance(sample_x, list):
            sample_x = [sample_x]
            converted_x = True
        
        converted_y = False
        if not isinstance(sample_y, list):
            sample_y = [sample_y]
            converted_y = True

        # Copy the data. For Pytorch.
        for i in range(len(sample_x)):
            sample_x[i] = sample_x[i].copy()
        
        for j in range(len(sample_y)):
            sample_y[j] = sample_y[j].copy()

        if self.input_is_channel_last:
            for i in range(len(sample_x)):
                sample_x[i] = channel_last_to_first(sample_x[i])
            
            for j in range(len(sample_y)):
                sample_y[j] = channel_last_to_first(sample_y[j])

        # Apply callback if specified
        if self.callback is not None:
            preconv_x = sample_x
            if converted_x:
                preconv_x = sample_x[0]
            
            preconv_y = sample_y
            if converted_y:
                preconv_y = sample_y[0]

            sample_x, sample_y = self.callback(preconv_x, preconv_y)

            if converted_x:
                sample_x = [sample_x]
            
            if converted_y:
                sample_y = [sample_y]

        # Convert output format if necessary
        if self.output_is_channel_last:
            for i in range(len(sample_x)):
                sample_x[i] = channel_first_to_last(sample_x[i])
            
            for j in range(len(sample_y)):
                sample_y[j] = channel_first_to_last(sample_y[j])

        if converted_x:
            sample_x = sample_x[0]
        
        if converted_y:
            sample_y = sample_y[0]

        return sample_x, sample_y



class AugmentationDataset():

    def __init__(self,
        X: Union[Union[np.ndarray, MultiArray], List[Union[np.ndarray, MultiArray]]],
        y: Union[Union[np.ndarray, MultiArray], List[Union[np.ndarray, MultiArray]]],
        augmentations: Optional[List] = None,
        callback: Callable = None,
        callback_pre: Callable = None,
        input_is_channel_last: Optional[bool] = None,
        output_is_channel_last: Optional[bool] = None,
    ):
        self.x_train = X
        self.y_train = y

        self.augmentations = augmentations or []
        self.callback = callback
        self.callback_pre = callback_pre
        self.input_is_channel_last = input_is_channel_last
        self.output_is_channel_last = output_is_channel_last

        # Read the first sample to determine if it is multi input
        self.x_is_multi_input = isinstance(self.x_train[0], list) and len(self.x_train[0]) > 1
        self.y_is_multi_input = isinstance(self.y_train[0], list) and len(self.y_train[0]) > 1

        assert len(self.x_train) == len(self.y_train), "X and y must have the same length."
        assert input_is_channel_last is not None, "Input channel format must be specified."
        assert output_is_channel_last is not None, "Output channel format must be specified."

        # If X is more than one array, then we need to make sure that the
        # number of list of augmentations is the same as the number of arrays.
        if self.x_is_multi_input or self.y_is_multi_input:
            if len(self.augmentations) == 0:
                pass
            elif not isinstance(self.augmentations[0], list):
                x_len = len(self.x_train[0]) if self.x_is_multi_input else 1
                y_len = len(self.y_train[0]) if self.y_is_multi_input else 1

                self.augmentations = [self.augmentations] * (x_len + y_len)
            else:
                assert len(self.augmentations) == len(X), "Number of augmentations must match number of arrays."

        test_augs = [self.augmentations] if not (self.x_is_multi_input or self.y_is_multi_input) else self.augmentations
        for aug_outer in test_augs:
            for aug in aug_outer:

                # Check if augmentation is valid
                if "name" not in aug:
                    raise ValueError("Augmentation name not specified.")

                if "p" not in aug and "chance" not in aug:
                    raise ValueError("Augmentation chance not specified.")

                aug_name = aug["name"]
                if "chance" in aug:
                    aug_change = aug["chance"]
                elif "p" in aug:
                    aug_change = aug["p"]

                assert aug_change is not None, "Augmentation chance cannot be None."
                assert 0 <= aug_change <= 1, "Augmentation chance must be between 0 and 1."
                assert aug_name is not None, "Augmentation name cannot be None."

                # Check if augmentation is valid for multi input
                if (aug_name[-2:] == "xy" or aug_name in ["cutmix", "mixup"]) and (self.x_is_multi_input or self.y_is_multi_input):
                    raise ValueError("Augmentation that target labels are not supported for multi input. (_xy augmentations)")

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        sample_x = self.x_train[index]
        sample_y = self.y_train[index]

        # Remeber to copy the data, otherwise it will be changed inplace
        converted_x = False
        if not isinstance(sample_x, list):
            sample_x = [sample_x]
            converted_x = True
        
        converted_y = False
        if not isinstance(sample_y, list):
            sample_y = [sample_y]
            converted_y = True

        # Copy the data. For Pytorch.
        for i in range(len(sample_x)):
            sample_x[i] = sample_x[i].copy()
        
        for j in range(len(sample_y)):
            sample_y[j] = sample_y[j].copy()

        if self.input_is_channel_last:
            for i in range(len(sample_x)):
                sample_x[i] = channel_last_to_first(sample_x[i])
            
            for j in range(len(sample_y)):
                sample_y[j] = channel_last_to_first(sample_y[j])

        # Apply callback_pre if specified. For normalisation.
        if self.callback_pre is not None:
            preconv_x = sample_x
            if converted_x:
                preconv_x = sample_x[0]
            
            preconv_y = sample_y
            if converted_y:
                preconv_y = sample_y[0]

            sample_x, sample_y = self.callback_pre(preconv_x, preconv_y)

            if converted_x:
                sample_x = [sample_x]
            
            if converted_y:
                sample_y = [sample_y]

        if self.x_is_multi_input or self.y_is_multi_input:
            # Apply augmentations
            for i in range(len(sample_x)):
                for aug in self.augmentations[i]:
                    aug_name = aug["name"]
                    if "chance" in aug:
                        aug_change = aug["chance"]
                    elif "p" in aug:
                        aug_change = aug["p"]
                    func = None

                    # Check if augmentation should be applied
                    if np.random.rand() > aug_change:
                        break

                    # Mapping augmentation names to their respective functions
                    if aug_name == "rotation":
                        func = augmentation_rotation
                    elif aug_name == "mirror":
                        func = augmentation_mirror
                    elif aug_name == "channel_scale":
                        func = augmentation_channel_scale
                    elif aug_name == "noise_uniform":
                        func = augmentation_noise_uniform
                    elif aug_name == "noise_normal":
                        func = augmentation_noise_normal
                    elif aug_name == "contrast":
                        func = augmentation_contrast
                    elif aug_name == "drop_pixel":
                        func = augmentation_drop_pixel
                    elif aug_name == "drop_channel":
                        func = augmentation_drop_channel
                    elif aug_name == "blur":
                        func = augmentation_blur
                    elif aug_name == "sharpen":
                        func = augmentation_sharpen
                    elif aug_name == "misalign":
                        func = augmentation_misalign

                    if func is None:
                        raise ValueError(f"Augmentation {aug['name']} not supported.")

                    kwargs = {key: value for key, value in aug.items() if key not in ["name", "chance", "inplace", "p"]}

                    sample_x[i] = func(sample_x[i], channel_last=False, inplace=True, **kwargs)
        else:
            x = sample_x[0]
            y = sample_y[0]

            # Apply augmentations
            for aug in self.augmentations:
                aug_name = aug["name"]
                if "chance" in aug:
                    aug_change = aug["chance"]
                elif "p" in aug:
                    aug_change = aug["p"]
                func = None

                # Check if augmentation should be applied
                if np.random.rand() > aug_change:
                    break

                # Mapping augmentation names to their respective functions
                if aug_name == "rotation":
                    func = augmentation_rotation
                elif aug_name == "rotation_xy":
                    func = augmentation_rotation_xy
                elif aug_name == "mirror":
                    func = augmentation_mirror
                elif aug_name == "mirror_xy":
                    func = augmentation_mirror_xy
                elif aug_name == "channel_scale":
                    func = augmentation_channel_scale
                elif aug_name == "noise_uniform":
                    func = augmentation_noise_uniform
                elif aug_name == "noise_normal":
                    func = augmentation_noise_normal
                elif aug_name == "contrast":
                    func = augmentation_contrast
                elif aug_name == "drop_pixel":
                    func = augmentation_drop_pixel
                elif aug_name == "drop_channel":
                    func = augmentation_drop_channel
                elif aug_name == "blur":
                    func = augmentation_blur
                elif aug_name == "blur_xy":
                    func = augmentation_blur_xy
                elif aug_name == "sharpen":
                    func = augmentation_sharpen
                elif aug_name == "sharpen_xy":
                    func = augmentation_sharpen_xy
                elif aug_name == "misalign":
                    func = augmentation_misalign
                elif aug_name == "cutmix":
                    func = augmentation_cutmix
                elif aug_name == "mixup":
                    func = augmentation_mixup

                if func is None:
                    raise ValueError(f"Augmentation {aug['name']} not supported.")

                kwargs = {key: value for key, value in aug.items() if key not in ["name", "chance", "inplace", "p"]}

                # Augmentations that apply to both image and label
                if aug_name in ["rotation_xy", "mirror_xy", "blur_xy", "sharpen_xy"]:
                    x, y = func(x, y, channel_last=False, inplace=True, **kwargs)

                # Augmentations that needs two images
                elif aug_name in ["cutmix", "mixup"]:
                    idx_source = np.random.randint(len(self.x_train))
                    xx = self.x_train[idx_source]
                    yy = self.y_train[idx_source]

                    if isinstance(xx, list):
                        xx = xx[0]
                    if isinstance(yy, list):
                        yy = yy[0]

                    if self.input_is_channel_last:
                        xx = channel_last_to_first(xx)
                        yy = channel_last_to_first(yy)

                    x, y = func(x, y, xx, yy, channel_last=False, inplace=True, **kwargs)

                # Augmentations that only apply to image
                else:
                    x = func(x, channel_last=False, inplace=True, **kwargs)
                
                sample_x = [x]
                sample_y = [y]

        # Apply callback if specified
        if self.callback is not None:
            if converted_x:
                sample_x = sample_x[0]
        
            if converted_y:
                sample_y = sample_y[0]

            sample_x, sample_y = self.callback(sample_x, sample_y)
        
        if self.callback is None and converted_x:
            sample_x = sample_x[0]
        
        if self.callback is None and converted_y:
            sample_y = sample_y[0]

        if self.output_is_channel_last:
            for i in range(len(sample_x)):
                sample_x[i] = channel_first_to_last(sample_x[i])
            
            for j in range(len(sample_y)):
                sample_y[j] = channel_first_to_last(sample_y[j])

        if converted_x:
            sample_x = sample_x[0]
        
        if converted_y:
            sample_y = sample_y[0]

        return sample_x, sample_y



if __name__ == "__main__":
    arr = load_data()

    # These are lists of multiple memmapped numpy arrays.
    # Notice: It handles multi-modal data.
    x_train = MultiArray([arr["train_s1"], arr["train_s2"]])
    y_train = MultiArray(arr["train_label_area"])

    x_test = MultiArray([arr["test_s1"], arr["test_s2"]])
    y_test = MultiArray(arr["test_label_area"])

    # Do your normalisation in the pre-augmentation callback
    def callback_pre(x, y):
        x_norm_s1, _statdict = beo.scaler_truncate(x[0], -50.0, 20.0)
        x_norm_s2 = (x[1] / 10000.0).astype(np.float32)

        return [x_norm_s1, x_norm_s2], y

    # Do the conversion to tensors in the callback (post-augmentation)
    def callback(x, y):
        return (
            (
                torch.from_numpy(x[0]),
                torch.from_numpy(x[1]),
            ),
            torch.from_numpy(y),
        )

    train_ds = AugmentationDataset(
        x_train,
        y_train,
        augmentations=[
            # { "name": "cutmix", "chance": 0.2 }, # Not supported for multi-modal data.
            { "name": "noise_uniform", "chance": 0.2 },
        ],
        input_is_channel_last=True,
        output_is_channel_last=False,
        callback_pre=callback_pre,
        callback=callback,
    )

    (t_sample_s1, t_sample_s2), t_label = train_ds[0]

    def callback_test(x, y):
        x_norm_s1, _statdict = beo.scaler_truncate(x[0], -50.0, 20.0)
        x_norm_s2 = (x[1] / 10000.0).astype(np.float32)

        x_norm_s1 = torch.from_numpy(x_norm_s1)
        x_norm_s2 = torch.from_numpy(x_norm_s2)
        y = torch.from_numpy(y)

        return (x_norm_s1, x_norm_s2), y

    test_ds = Dataset(
        x_test,
        y_test,
        input_is_channel_last=True,
        output_is_channel_last=False,
        callback=callback_test,
    )
    
    (sample_s1, sample_s2), label = test_ds[0]

    import pdb; pdb.set_trace()