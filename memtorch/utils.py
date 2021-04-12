import codecs
import gzip
import lzma
import math
import os
import os.path
import string
import warnings
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.error import URLError

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import datasets, transforms
from torchvision.datasets.utils import (download_and_extract_archive,
                                        download_url, extract_archive,
                                        verify_str_arg)

import memtorch


# Hotfix from https://github.com/seemethere/vision/blob/76e325d04475d18b53ea51ede618ee54a30c2706/torchvision/datasets/mnist.py
# ----------------------------------------------------------------------------------------------------------------------------
class MNIST(datasets.VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = [
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ]

    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(MNIST, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(
            os.path.join(self.processed_folder, data_file)
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return os.path.exists(
            os.path.join(self.processed_folder, self.training_file)
        ) and os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    download_and_extract_archive(
                        url, download_root=self.raw_folder, filename=filename, md5=md5
                    )
                except URLError as error:
                    print("Failed to download (trying next):\n{}".format(error))
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))

        # process and save as torch files
        print("Processing...")

        training_set = (
            read_image_file(os.path.join(self.raw_folder, "train-images-idx3-ubyte")),
            read_label_file(os.path.join(self.raw_folder, "train-labels-idx1-ubyte")),
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, "t10k-images-idx3-ubyte")),
            read_label_file(os.path.join(self.raw_folder, "t10k-labels-idx1-ubyte")),
        )
        with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)

        print("Done!")

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


def open_maybe_compressed_file(path: Union[str, IO]) -> Union[IO, gzip.GzipFile]:
    """Return a file object that possibly decompresses 'path' on the fly.
    Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    if path.endswith(".xz"):
        return lzma.open(path, "rb")
    return open(path, "rb")


SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype(">i2"), "i2"),
    12: (torch.int32, np.dtype(">i4"), "i4"),
    13: (torch.float32, np.dtype(">f4"), "f4"),
    14: (torch.float64, np.dtype(">f8"), "f8"),
}


def read_sn3_pascalvincent_tensor(
    path: Union[str, IO], strict: bool = True
) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def read_label_file(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 1
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 3
    return x


# ----------------------------------------------------------------------------------------------------------------------------


def convert_range(old_value, old_min, old_max, new_min, new_max):
    """Method to convert values between two ranges.

    Parameters
    ----------
    old_value : object
        Old value(s) to convert. May be a single number, vector or tensor.
    old_min : float
        Minimum old value.
    old_max : float
        Maximum old value.
    new_min : float
        Minimum new value.
    new_max : float
        Maximum new value.

    Returns
    -------
    object
        New value(s).
    """
    return (
        ((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)
    ) + new_min


def clip(value, lower, upper):
    """Method to clip a float between lower and upper bounds.

    Parameters
    ----------
    value : float
        Value to clip.
    lower : float
        Lower bound.
    upper : float
        Upper bound.

    Returns
    -------
    float
        Clipped float.
    """
    return lower if value < lower else upper if value > upper else value


def pad_tensor(tensor, tile_shape):
    """Method to zero-pad a tensor.

    Parameters
    ----------
    tensor : torch.tensor
        Tensor to zero-pad.
    tile_shape : (int, int)
        Tile shape to pad tensor for.

    Returns
    -------
    torch.tensor
        Padded tensor.
    """
    assert (
        len(tensor.shape) == 1 or len(tensor.shape) == 2
    ), "tensor.shape must be 1 or 2 dimensional."
    if len(tensor.shape) == 1:
        tensor_padded = torch.zeros(
            (math.ceil(tensor.shape[0] / tile_shape[0]) * tile_shape[0])
        )
        tensor_padded[0 : tensor.shape[0]] = tensor
    elif len(tensor.shape) == 2:
        tensor_padded = torch.zeros(
            (
                math.ceil(tensor.shape[0] / tile_shape[0]) * tile_shape[0],
                math.ceil(tensor.shape[1] / tile_shape[1]) * tile_shape[1],
            )
        )
        tensor_padded[0 : tensor.shape[0], 0 : tensor.shape[1]] = tensor

    return tensor_padded


def LoadMNIST(batch_size=32, validation=True):
    """Method to load the MNIST dataset.

    Parameters
    ----------
    batch_size : int
        Batch size.
    validation : bool
        Load the validation set (True).

    Returns
    -------
    list of torch.utils.data
        The train, validiation, and test loaders.
    """
    root = "data"
    transform = transforms.Compose([transforms.ToTensor()])
    full_train_set = MNIST(root=root, train=True, transform=transform, download=True)
    test_set = MNIST(root=root, train=False, transform=transform, download=True)
    if validation:
        train_size = int(0.8 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        train_set, validation_set = torch.utils.data.random_split(
            full_train_set, [train_size, validation_size]
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=2
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_set, batch_size=batch_size, shuffle=True, num_workers=2
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            full_train_set, batch_size=batch_size, shuffle=True, num_workers=2
        )
        validation_loader = None

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=int(batch_size / 2), shuffle=False, num_workers=2
    )
    return train_loader, validation_loader, test_loader


def LoadCIFAR10(batch_size=32, validation=True):
    """Method to load the CIFAR-10 dataset.

    Parameters
    ----------
    batch_size : int
        Batch size.
    validation : bool
        Load the validation set (True).

    Returns
    -------
    list of torch.utils.data
        The train, validiation, and test loaders.
    """
    root = "data"
    transform = transforms.Compose([transforms.ToTensor()])
    full_train_set = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform
    )
    if validation:
        train_size = int(0.8 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        train_set, validation_set = torch.utils.data.random_split(
            full_train_set, [train_size, validation_size]
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=2
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_set, batch_size=batch_size, shuffle=True, num_workers=2
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            full_train_set, batch_size=batch_size, shuffle=True, num_workers=2
        )
        validation_loader = None

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=int(batch_size / 2), shuffle=False, num_workers=2
    )
    return train_loader, validation_loader, test_loader
