import argparse
import os
import pytorch_lightning as pl

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Dict

from text_recognizer import util

BATCH_SIZE=128
NUM_WORKERS=0

def load_and_print_info(data_module_class:type)->None:
    parser=argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args=parser.parse_args()
    dataset=data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)

def download_raw_dataset(metadata:Dict,dl_dirname:Path)->Path:
    dl_dirname.mkdir(parents=True,exist_ok=True)
    filename=dl_dirname/metadata["filename"]
    if filename.exists():
        return
    print(f"Downloading raw dataset from {metadata["url"]} to {filename}.")
    util.download_url(metadata["url"],filename)
    print("Computing SHA-256")
    sha256=util.compute_sha256(filename)
    if sha256 !=metadata["sha256"]:
        raise ValueError("Download data file SHA-256 does not match that listed in metadata document.")
    
    return filename

class BaseDataModule(pl.LightningDataModule):
    def __intit__(self,args:argparse.Namespace)->None:
        super().__init__()
        self.args=vars(args) if args is not None else {}
        self.batch_size=self.args.get("batch_size",BATCH_SIZE)
        self.num_workers=self.args.get("num_workers",NUM_WORKERS)

        self.dims=None
        self.output_dims=None
        self.mapping=None
    
    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[3] / "data"
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--batch_size",type=int,default=BATCH_SIZE,help="Number of examples to operate on per forward step.")
        parser.add_argument("--num_workers",type=int,default=NUM_WORKERS,help="Number of additional processes to load data.")
        parser.add_argument("--subsample_fraction",type=float,default=None,help="If given, is used as the fraction of data to expose.")
    
    def config(self):
        return {"input_dims":self.dims,"output_dims":self.output_dims,"mapping":self.mapping}
    
    def prepare_data(self):
        pass

    def setup(self,stage=None):
        self.data_train=None
        self.data_val=None
        self.data_text=None
    
    def train_dataloader(self):
        return DataLoader(self.data_train,batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.data_val,batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.data_test,batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=True)