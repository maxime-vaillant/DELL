import os
from abc import abstractmethod

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class BaseSpikeDataModule(LightningDataModule):
    # Names the metrics of each validation dataloader, in the order `val_dataloader()` returns them.
    # Only M3EDDataModule overrides this, to split validation into car and quadruped.
    VAL_STAGES: tuple = ('val',)

    def __init__(
        self,
        batch_size: int,
        data_path: str = None,
        num_workers: int = 8,
        num_steps: int = 10,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path or os.path.expanduser("~/data")
        self.num_workers = num_workers
        self.num_steps = num_steps

        self.train_ds: Dataset = None
        self.test_ds: Dataset = None

    # `_encode` and `_collate` are not instance methods: `collate_fn` is pickled for worker
    # processes, and a bound method would drag the datamodule (and the model's parameters) along.
    @staticmethod
    def _encode(x: torch.Tensor) -> torch.Tensor:
        # (B, T, 2, H, W) -> (T, B, 2, H, W)
        return x.permute(1, 0, 2, 3, 4).float()

    @classmethod
    def _collate(cls, batch):
        x, y = torch.utils.data.dataloader.default_collate(batch)

        spikes = cls._encode(x)

        return spikes, y

    @abstractmethod
    def setup(self, stage=None): ...

    def _make_loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn=self._collate,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False)
