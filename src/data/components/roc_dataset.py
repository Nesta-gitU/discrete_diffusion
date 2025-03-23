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

import spacy
from spacy.lang.en import English
from collections import Counter
from .custom_tokenizers import TokenizerFromDict


class ROCdataset(TorchDataset):
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
        return len(self.data)

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

        print('loading dataset from ROCStory')
        nlp = English()
        tokenizer = nlp.tokenizer
        sentence_lst = []
        print(f'loading from {data_args.roc_train}')
        if self.split == 'train':
            print('loading form the TRAIN set')
            path = f'{data_args.roc_train}/roc_train.json'
        elif self.split == 'valid':
            print('loading form the VALID set')
            path = f'{data_args.roc_train}/roc_valid.json'
        elif self.split == 'test':
            path = f'{data_args.roc_train}/roc_test.json'

        with open(path, 'r') as roc_reader:
            #i=0
            for row in roc_reader:
                sentences = json.loads(row)[0].strip()
                word_lst = [x.text for x in tokenizer(sentences)]
                sentence_lst.append(word_lst)

        self.sentence_lst = sentence_lst
        #print vocab size
        #get a vocab dict
        counter = Counter()
        for input_ids in sentence_lst:
            counter.update(input_ids)
        
        vocab_dict = {'PAD': 0, 'EOS': 1, 'UNK': 2}
        for k, v in counter.items():
            if v > 10:
                vocab_dict[k] = len(vocab_dict)
        
        tokenizer = TokenizerFromDict(vocab_dict)
        self.tokenizer = tokenizer

        print(f"Vocab size: {len(vocab_dict)}")

        encoded_ids = []
        for sentence in sentence_lst:
            input_ids = [0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in sentence] + [1]
            encoded_ids.append(input_ids)

        encoded_ids = self._collate_batch_helper(encoded_ids, 0, self.block_size)

        self.data = encoded_ids

        return tokenizer
    
    def _collate_batch_helper(self, examples, pad_token_id, max_length, return_mask=False):
        result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
        mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
        for i, example in enumerate(examples):
            curr_len = min(len(example), max_length)
            result[i][:curr_len] = example[:curr_len]
            mask_[i][:curr_len] = [1] * curr_len
        if return_mask:
            return result, mask_
        return result