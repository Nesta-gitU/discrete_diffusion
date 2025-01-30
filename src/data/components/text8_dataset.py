import requests
import zipfile
from pathlib import Path
import logging
from typing import Optional

import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Union
from torch import Tensor

import numpy as np 
import os
import pickle

from src.data.components.custom_tokenizers import CharTokenizer

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class Text8Dataset(TorchDataset):
    data_lists: list[Tensor]
    file_url = "http://mattmahoney.net/dc/text8.zip"
    data_dir: Path

    def __init__(
        self,
        split: str,
        block_size: int,
        root_dir: str,
        overfit_one_batch: bool = False,
        return_index: bool = False,
        character_level: bool = False,
        epoch_length: int = 8000,
    ):
        
        self.epoch_length = epoch_length
        self.return_index = return_index
        self.block_size = block_size
        self.overfit_one_batch = overfit_one_batch
        self.text8_dir = Text8Dataset.get_data_dir(Path(root_dir)) / "text8"
        if character_level:
            self.text8_url = Text8Dataset.get_data_dir(Path(root_dir)) / "text8" / f"char_{split}.bin"
        else:
            self.text8_url = Text8Dataset.get_data_dir(Path(root_dir)) / "text8" / f"bpe_{split}.bin"

    #length is arbitrary so lets just make it the number of chuncks if you stack them after eachother
    #probably just shouldnt use the concept of epoch at all. 
    def __len__(self): 
        return self.epoch_length

    def __getitem__(self, index: int):

        #create a memmap for the current split, create a new one each time to prevent memory leakage.
        data = np.memmap(self.text8_url, dtype=np.uint16, mode='r')

        #the get item function should return a ternsor of block size but not for a whole batch, the loader will handle that. 
        if not self.overfit_one_batch:
            i = torch.randint(len(data) - self.block_size, (1,)).item() #this is questionable, because it does not garuantee seeing the whole dataset in an epoch. idk how to make better
        else:
            #i = 0  #this is for debugging purposes, so we can see the same string over and over again.
            # sample an index between 0-64
            i = 0
       
        x = torch.from_numpy((data[i:i+self.block_size]).astype(np.int64))
        
        
        return x

        
    @classmethod
    def prepare(cls, root_dir: str, character_level: bool = True, vocab_size: int = 10000):
        text8_file_path = Text8Dataset.get_data_dir(Path(root_dir)) / "text8" / "text8"

        with open(text8_file_path, 'r') as f:
            data = f.read()
        print(f"length of dataset in characters: {len(data):,}")

        n = len(data)
        test_data = data[:int(n*0.8)] 
        train_data = data[int(n*0.8):int(n*0.9)] 
        val_data = data[int(n*0.9):]

        #initialize a tokenizer here
        if character_level:
            tokenizer = CharTokenizer(data)
            tokenizer.save("char_tokenizer_text8.pkl")

            train_ids = tokenizer.encode(train_data)
            val_ids = tokenizer.encode(val_data)
            test_ids = tokenizer.encode(test_data)
        else:
            tokenizer = Tokenizer(BPE())
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
            tokenizer.train_from_iterator([data], trainer=trainer)
            tokenizer.save("tokenizer_text8.json")

            train_ids = tokenizer.encode_batch([train_data])[0].ids
            val_ids = tokenizer.encode_batch([val_data])[0].ids
            test_ids = tokenizer.encode_batch([test_data])[0].ids
        
        print(f"train has {len(train_ids):,} tokens")
        print(f"test has {len(test_ids):,} tokens")

        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        test_ids = np.array(test_ids, dtype=np.uint16)
        if character_level:
            train_ids.tofile(os.path.join(os.path.dirname(text8_file_path), 'char_train.bin'))
            val_ids.tofile(os.path.join(os.path.dirname(text8_file_path), 'char_val.bin'))
            test_ids.tofile(os.path.join(os.path.dirname(text8_file_path), 'char_test.bin'))
        else:
            train_ids.tofile(os.path.join(os.path.dirname(text8_file_path), 'bpe_train.bin'))
            val_ids.tofile(os.path.join(os.path.dirname(text8_file_path), 'bpe_val.bin'))
            test_ids.tofile(os.path.join(os.path.dirname(text8_file_path), 'bpe_test.bin'))

        return tokenizer

    @classmethod
    def get_data_dir(cls, root_dir: Path):
        data_dir = root_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @classmethod
    def download_and_extract(
        cls, root_dir: str, logger: Optional[logging.Logger] = None 
    ):
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
        root_dir = Path(root_dir)
        data_dir = cls.get_data_dir(root_dir)

        out_folder = data_dir / "text8" #should not be the same as datadir otherwise the if below goes wrong.
        if out_folder.exists():
            logger.info(f"Data already downloaded and extracted at {out_folder}")
            return out_folder

        logger.info(f"Downloading and extracting data to {out_folder}")
        r = requests.get(Text8Dataset.file_url, allow_redirects=True)
        with open(data_dir / "text8.zip", "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(data_dir / "text8.zip", "r") as zip_ref:
            zip_ref.extractall(out_folder) #since this file has only one inside it it doesnt unzip into a folder by itself. 
        # remove the zip file
        (data_dir / "text8.zip").unlink()
        return out_folder
