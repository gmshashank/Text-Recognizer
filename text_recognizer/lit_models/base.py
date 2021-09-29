import torch
import pytorch_lightning as pl


class BaseLitModel(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.optimizer_class = getattr(torch.optim, args.optimizer)
        self.lr = args.lr
        if not args.loss == "transformer":
            self.loss_fn = getattr(torch.nn.functional, args.loss)
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--optimizer",
            type=str,
            default="Adam",
            help="optimizer class from torch.optim",
        )
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument(
            "--loss",
            type=str,
            default="cross entropy",
            help="loss function from torch.nn.functional",
        )
        return parser
