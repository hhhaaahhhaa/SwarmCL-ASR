import torch
from torch.utils.data import Dataset, DataLoader
import lightning as pl


class DataModule(pl.LightningDataModule):
    """ Simple wrapper class to wrap datasets into pytorch lightning datamodules. """
    def __init__(
        self,
        train_dataset: Dataset=None,
        val_dataset: Dataset=None,
        test_dataset: Dataset=None,
        batch_size: int=1,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        return self.val_loader
