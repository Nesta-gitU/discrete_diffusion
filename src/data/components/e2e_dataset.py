import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset as TorchDataset
from torch import Tensor

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class E2EDataset(TorchDataset):
    def __init__(
        self,
        split: str = "train",
        vocab_size: int = 10000,
        block_size: int = 128,
        epoch_length: int = 8000,  # Used if data is not prepared yet.
        overfit_one_batch: bool = False,
    ):
        """
        Args:
            split (str): Which split to use: "train", "valid", "test", or "debug".
            vocab_size (int): Vocabulary size for the BPE tokenizer.
            block_size (int): Fixed length of token sequences to return.
            epoch_length (int): Arbitrary length if data is not yet prepared.
            overfit_one_batch (bool): If True, __getitem__ always returns the sample at index 0.
        """
        self.split = split
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.epoch_length = epoch_length
        self.overfit_one_batch = overfit_one_batch

        self.tokenizer: Optional[Tokenizer] = None
        self.sentence_lst: Optional[list[str]] = None
        # We'll store self.data as a list of lists (each inner list holds token IDs for one sentence)
        self.data: Optional[list[list[int]]] = None

    def __len__(self) -> int:
        if self.data is not None:
            return len(self.data)
        return self.epoch_length

    def __getitem__(self, index: int) -> Tensor:
        """
        Returns a sample (a sentence) as a tensor of token IDs padded or truncated to block_size.
        Padding is done using the tokenizer's special [PAD] token.
        If overfit_one_batch is True, always returns the sample at index 0.
        """
        if self.data is None:
            raise ValueError("Data not prepared. Please call the prepare() method first.")

        # In overfit mode, always return the first sample.
        if self.overfit_one_batch:
            index = 0

        # Retrieve the sample's token IDs (a list of ints)
        sample_ids = self.data[index]
        sample_tensor = torch.tensor(sample_ids, dtype=torch.long)

        # Truncate if the sample is longer than block_size.
        if sample_tensor.size(0) > self.block_size:
            sample_tensor = sample_tensor[: self.block_size]
        # Pad with [PAD] token if the sample is shorter than block_size.
        elif sample_tensor.size(0) < self.block_size:
            pad_len = self.block_size - sample_tensor.size(0)
            # Retrieve the pad token id from the tokenizer.
            pad_id = self.tokenizer.token_to_id("[PAD]")
            pad_tensor = torch.full((pad_len,), pad_id, dtype=torch.long)
            sample_tensor = torch.cat([sample_tensor, pad_tensor], dim=0)

        return sample_tensor

    def prepare(self, data_args, tokenizer=None) -> Tokenizer:
        """
        Prepares the dataset by:
          1. Loading sentences from the appropriate file based on the chosen split.
          2. Training a BPE tokenizer (with a Whitespace pre-tokenizer) on these sentences.
          3. Encoding each sentence into token ids and storing them as separate samples.

        Args:
            data_args: A configuration object (or namespace) with attributes:
                       - e2e_train (str): Directory path for the train/valid/test files.
                       - debug_path (str): File path for the debug split.

        Returns:
            The trained tokenizer.
        """
        print("preparing data...")
        sentence_lst = []

        if self.split == "train":
            print("Loading from the TRAIN set")
            path = f"{data_args.e2e_train}/src1_train.txt"
        elif self.split == "valid":
            print("Loading from the VALID set")
            path = f"{data_args.e2e_train}/src1_valid.txt"
        elif self.split == "test":
            print("Loading from the TEST set")
            path = f"{data_args.e2e_train}/src1_test.txt"
        elif self.split == "debug":
            print("Loading from the DEBUG set")
            path = data_args.debug_path
            with open(path, "r") as ff:
                for line in ff:
                    # Assuming each line is a JSON list and the first element is the sentence.
                    sentence = json.loads(line)[0]
                    sentence_lst.append(sentence)
            # Duplicate the sentences as in your original code.
            sentence_lst = sentence_lst + sentence_lst
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # For train/valid/test splits, assume each line contains a '||' separator.
        if self.split in ["train", "valid", "test"]:
            with open(path, "r") as ff:
                for row in ff:
                    parts = row.split("||")
                    # Use parts[1].strip() if the separator exists to extract the sentence portion.
                    if len(parts) > 1:
                        text = parts[1].strip()
                    else:
                        text = row.strip()
                    sentence_lst.append(text)

        self.sentence_lst = sentence_lst

        # Initialize and train a BPE tokenizer using the Hugging Face Tokenizers library.
        if tokenizer is None:
            tokenizer = Tokenizer(BPE())
            tokenizer.pre_tokenizer = Whitespace()
            # Include both "[UNK]" and "[PAD]" as special tokens.
            trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=["[UNK]", "[PAD]"])
            tokenizer.train_from_iterator(sentence_lst, trainer=trainer)
            
        self.tokenizer = tokenizer

        # Encode each sentence separately.
        encoded_ids = []
        for sentence in sentence_lst:
            encoding = tokenizer.encode(sentence)
            encoded_ids.append(encoding.ids)
        self.data = encoded_ids

        return tokenizer
