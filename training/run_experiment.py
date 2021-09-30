import argparse
import importlib
import numpy as np
from pytorch_lightning import callbacks
import torch
import pytorch_lightning as pl
import sys

sys.path.append(".")

from text_recognizer import lit_models

# In order to enusre reproducible experiments.
np.random.seed(42)
torch.manual_seed(42)


def _import_class(module_and_class_name: str) -> type:
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    # Setup Python ArgumentParser with data, model, trainer and other arguments
    parser = argparse.ArgumentParser(add_help=False)

    # Add trainer specific arguments such as --max_epochs,--gpus,--precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic Arguments
    parser.add_argument("--data_class", type=str, default="MNIST")
    parser.add_argument("--model_class", type=str, default="MLP")

    # Get the data and model classes to add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"text_recognizer.data.{temp_args.data_class}")
    model_class = _import_class(f"text_recognizer.models.{temp_args.model_class}")

    # Get data, model and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"text_recognizer.data.{args.data_class}")
    model_class = _import_class(f"text_recognizer.models.{args.model_class}")
    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)

    lit_model = lit_models.BaseLitModel(args=args, model=model)
    loggers = [pl.loggers.TensorBoardLogger("training/logs")]
    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
    ]
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks, logger=loggers, default_root_dir="training/logs"
    )
    trainer.tune(lit_model, datamodule=data)
    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
    main()
    print("Done")
