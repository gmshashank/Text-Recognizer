from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Union
from urllib.request import urlopen, urlretrieve
from tqdm import tqdm

import cv2
import os
import hashlib
import numpy as np


def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype="uint8")[y]


def read_image(image_uri: Union[Path, str], grayscale=False) -> np.array:
    def read_image_from_filename(image_filename, imread_flag):
        return cv2.imread(str(image_filename), imread_flag)

    def read_image_from_url(image_url, imread_flag):
        url_response = urlopen(str(image_url))
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, imread_flag)

    imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    local_file = os.path.exists(image_uri)
    try:
        img = None
        if local_file:
            img = read_image_from_filename(image_uri, imread_flag)
        else:
            img = read_image_from_url(image_uri, imread_flag)
    except Exception as e:
        raise ValueError(f"Could not load image at {image_uri}: {e}")

    return img


def write_image(image: np.ndarray, filename: Union[Path, str]) -> None:
    cv2.imwrite(str(filename), image)


def compute_sha256(filename: Union[Path, str]):
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TqdmUpTo(tqdm):
    def update_to(self, blocks=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)


def download_url(url, filename):
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)
