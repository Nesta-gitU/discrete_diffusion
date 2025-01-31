from typing import Any, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.data.components.text8_dataset import Text8Dataset
import logging
import itertools
import os 

from src.data.components.custom_tokenizers import CharTokenizer
from tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO)


class Text8DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        block_size: int = 256,
        vocab_size: int = 100000,
        root_dir: str = "data/text8",
        num_workers: int = 0,
        overfit_one_batch: bool = False,
        pin_memory: bool = False,
        character_level: bool = False,
        reload_data: bool = False,
        epoch_length: int = 8000,
    ) -> None:
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.logger = logging.getLogger(__name__)

        # data transformations
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    def prepare_data(self) -> None:
        Text8Dataset.download_and_extract(root_dir=self.hparams.root_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices. -> I deleted this for now
        
        # load and split datasets only if not loaded already
        if not self.data_train:
            print("loading data...")
            if not self.hparams.reload_data:
                if self.hparams.character_level and os.path.exists(os.path.join(self.hparams.root_dir, 'data', 'text8', 'char_train.bin')):
                        print("char_data data already loaded, skipping...")
                        self.tokenizer = CharTokenizer.load("char_tokenizer_text8.pkl")
                elif os.path.exists(os.path.join(self.hparams.root_dir, 'data', 'text8', 'bpe_train.bin')) and not self.hparams.character_level:
                    print("bpe_data data already loaded, skipping...")
                    self.tokenizer = Tokenizer.from_file("tokenizer_text8.json")
                else:
                    print("loading data with function...")
                    self.tokenizer = Text8Dataset.prepare(root_dir=self.hparams.root_dir, character_level=self.hparams.character_level, vocab_size=self.hparams.vocab_size)
            else:
                print("relaoading data with function...")
                print(self.hparams.character_level)
                self.tokenizer = Text8Dataset.prepare(root_dir=self.hparams.root_dir, character_level=self.hparams.character_level, vocab_size=self.hparams.vocab_size)



            self.data_train = Text8Dataset(split= "train" ,block_size=self.hparams.block_size, root_dir=self.hparams.root_dir, 
                overfit_one_batch=self.hparams.overfit_one_batch, character_level=self.hparams.character_level, epoch_length=self.hparams.epoch_length)
            self.data_val = Text8Dataset(split= "val", block_size=self.hparams.block_size, root_dir=self.hparams.root_dir, 
                overfit_one_batch= self.hparams.overfit_one_batch, character_level=self.hparams.character_level, epoch_length=self.hparams.epoch_length)
            self.data_test = Text8Dataset(split="test", block_size=self.hparams.block_size, root_dir=self.hparams.root_dir, 
                overfit_one_batch=self.hparams.overfit_one_batch, character_level=self.hparams.character_level, epoch_length=self.hparams.epoch_length)


            #made some insane changes here
            

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
