import torch
from typing import Any, Callable, Sequence, Tuple, Union

SequenceOrTensor = Union[Sequence, torch.Tensor]


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: SequenceOrTensor,
        targets: SequenceOrTensor,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        if len(data) != len(targets):
            raise ValueError("Data and targets ust be of equal length")
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitme__(self, index: int) -> Tuple[Any, Any]:
        datum, target = self.data[index], self.targets[index]
        if self.transform is not None:
            datum = self.transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target
