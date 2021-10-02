import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import toml
import torch
from torchvision import transforms

from text_recognizer.data.base_data_module import BaseDataModule, _download_raw_dataset, load_and_print_info
from text_recognizer.data.util import BaseDataset

NUM_SPECIAL_TOKENS = 4
SAMPLE_TO_BALANCE = True
TRAIN_FRAC = 0.8

RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "emnist"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "download" / "emnist"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "byclass.h5"
ESSENTIALS_FILENAME = Path(__file__).parents[0].resolve() / "emnist_essentials.json"


def _sample_to_balance(x, y):
    np.random.seed(42)
    num_to_sample = int(np.bincount(y.flatten()).mean())
    all_sampled_inds = []
    for label in np.unique(y.flatten()):
        inds = np.where(y == label)[0]
        sampled_inds = np.unique(np.random.choice(inds, num_to_sample))
        all_sampled_inds.append(sampled_inds)
    ind = np.concatenate(all_sampled_inds)
    x_sampled = x[ind]
    y_sampled = y[ind]
    return x_sampled, y_sampled


def _augment_emnist_characters(characters: Sequence[str]) -> Sequence[str]:
    """Augment the mapping with extra symbols."""
    # Extra characters from the IAM dataset
    iam_characters = [
        " ",
        "!",
        '"',
        "#",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        ":",
        ";",
        "?",
    ]

    # Also add special tokens:
    # - CTC blank token at index 0
    # - Start token at index 1
    # - End token at index 2
    # - Padding token at index 3
    # NOTE: Don't forget to update NUM_SPECIAL_TOKENS if changing this!
    return ["<B>", "<S>", "<E>", "<P>", *characters, *iam_characters]


def _process_raw_dataset(filename: str, dirname: Path):
    print("Unzipping EMNIST")
    curdir = os.getcwd()
    os.chdir(dirname)
    zip_file = zipfile.ZipFile(filename, "r")
    zipfile.extract("matlab/emnist-byclass.mat")

    from scipy.io import loadmat

    print("Loading training data from .mat file")
    data = loadmat("matlab/emnist-byclass.mat")
    x_train = data["dataset"]["train"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_train = data["dataset"]["train"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS
    x_test = data["dataset"]["test"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_test = data["dataset"]["test"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS

    if SAMPLE_TO_BALANCE:
        print("Balancing classes to reduce amount of data")
        x_train, y_train = _sample_to_balance(x_train, y_train)
        x_test, y_test = _sample_to_balance(x_test, y_test)

    print("Saving HDF5 in compressed format.")
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    with h5py.File(PROCESSED_DATA_FILENAME, "w") as f:
        f.create_dataset("x_train", data=x_train, dtype="u1", compression="lzf")
        f.create_dataset("y_train", data=y_train, dtype="u1", compression="lzf")
        f.create_dataset("x_test", data=x_test, dtype="u1", compression="lzf")
        f.create_dataset("y_test", data=y_test, dtype="u1", compression="lzf")

    print("saving dataset parameters to text__recognizer/datasets")
    mapping = {int(k): chr(v) for k, v in data["dataset"]["mapping"][0, 0]}
    characters = _augment_emnist_characters(mapping.values())
    essentials = {"characters": characters, "input_shape": list(x_train.shape[1:])}
    with open(ESSENTIALS_FILENAME, "w") as f:
        json.dump(essentials, f)

    print("Cleaning up.")
    shutil.rmtree("matlab")
    os.chdir(curdir)


def _download_and_process_emnist():
    metadata = toml.load(METADATA_FILENAME)
    _download_raw_dataset(metadata, DL_DATA_DIRNAME)
    _process_raw_dataset(metadata["filename"], DL_DATA_DIRNAME)


class EMNIST(BaseDataModule):
    def __init__(self, args=None):
        super().__init__(args)
        if not os.path.exists(ESSENTIALS_FILENAME):
            _download_and_process_emnist()

        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)

        self.mapping = list(essentials["characters"])
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dims = (1, *essentials["input_shape"])
        self.output_dims = (1,)

    def prepare_data(self):
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_and_process_emnist()
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_trainval = f["x_train"][:]
                self.y_trainval = f["y_train"][:].squeeze().astype(int)

            data_trainval = BaseDataset(self.x_trainval, self.y_trainval, transform=self.transform)
            train_size = int(TRAIN_FRAC * len(data_trainval))
            val_size = len(data_trainval) - train_size
            self.data_train, self.data_val = torch.utils.data.random_split(
                data_trainval, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )

        if stage == "test" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_test = f["x_test"][:]
                self.y_test = f["y_test"][:].squeeze().astype(int)
            self.data_test = BaseDataset(self.x_test, self.y_test, transform=self.transform)

    def __repr__(self):
        basic = f"EMNIST Dataset\nNum classes:{len(self.mapping)}\nMapping: {self.mapping}\nDims:{self.dims}\n"
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = {
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape,x.dtype.x.min(),x.mean(),x.std(),x.max())}\n"
            f"Batch y stats: {(y.shape,y.dtype,y.min(),y.max())}\n"
        }
        return basic + data


if __name__ == "__main__":
    load_and_print_info(EMNIST)
