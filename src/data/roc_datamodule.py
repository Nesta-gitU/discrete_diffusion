from typing import Any, Optional
import os
from types import SimpleNamespace
import logging

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.roc_dataset import ROCdataset  # Adjust the import path as needed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ROCdatamodule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        block_size: int = 128,
        vocab_size: int = 10000,
        root_dir: str = "data/e2e",
        num_workers: int = 0,
        overfit_one_batch: bool = False,
        epoch_length: int = 8000,
        pin_memory: bool = True,
    ) -> None:
        """
        LightningDataModule for the E2E dataset.
        
        Args:
            batch_size (int): Batch size.
            block_size (int): Fixed length of each token sequence.
            vocab_size (int): Vocabulary size for the tokenizer.
            root_dir (str): Directory where the data files reside. Expected to contain:
                            - src1_train.txt
                            - src1_valid.txt
                            - src1_test.txt
                            - debug.txt (for debug split)
            num_workers (int): Number of subprocesses to use for data loading.
            overfit_one_batch (bool): If True, each __getitem__ returns the same sample.
            epoch_length (int): Used only if the dataset is not yet prepared.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[ROCdataset] = None
        self.data_val: Optional[ROCdataset] = None
        self.data_test: Optional[ROCdataset] = None

    def prepare_data(self) -> None:
        """
        No downloading or extraction is required because the data is assumed to be already available.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Instantiates and prepares the train, validation, and test datasets.
        """
        # Create a simple config object for the dataset.
        data_args = SimpleNamespace(
            roc_train=os.path.join(self.hparams.root_dir, "data", "roc"),  # This folder should contain src1_train.txt, etc.
            debug_path=os.path.join(self.hparams.root_dir, "debug.txt"),
        )

        # Prepare the training dataset.
        self.data_train = ROCdataset(
            split="train",
            vocab_size=self.hparams.vocab_size,
            block_size=self.hparams.block_size,
            epoch_length=self.hparams.epoch_length,
            overfit_one_batch=self.hparams.overfit_one_batch,
        )
        tokenizer, nlp_tokenizer = self.data_train.prepare(data_args)
        self.tokenizer = tokenizer

        # Prepare the validation dataset.
        self.data_val = ROCdataset(
            split="valid",
            vocab_size=self.hparams.vocab_size,
            block_size=self.hparams.block_size,
            epoch_length=self.hparams.epoch_length,
            overfit_one_batch=self.hparams.overfit_one_batch,
        )
        self.data_val.prepare(data_args, tokenizer=tokenizer, nlp_tokenizer=nlp_tokenizer)

        # Prepare the test dataset.
        self.data_test = ROCdataset(
            split="test",
            vocab_size=self.hparams.vocab_size,
            block_size=self.hparams.block_size,
            epoch_length=self.hparams.epoch_length,
            overfit_one_batch=self.hparams.overfit_one_batch,
        )
        self.data_test.prepare(data_args, tokenizer=tokenizer, nlp_tokenizer=nlp_tokenizer)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,  # Typically, you want to shuffle training data.
            pin_memory=self.hparams.pin_memory,
            drop_last=True,  # Drop the last incomplete batch
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=self.hparams.pin_memory
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=self.hparams.pin_memory
        )
