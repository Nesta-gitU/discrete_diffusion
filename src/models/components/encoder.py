import torch
from torch import nn
import pickle
from pathlib import Path

class IdentityEncoder(nn.Module):
    
    def init(self,
             root_dir: str):
        super().__init__()
        self.root_dir = root_dir
        self.vocab_size = self.get_vocab_size()
    
    def get_vocab_size(self):
        #open the meta file and get the vocab size
        url = Path(self.root_dir) / 'meta.pkl'
        with open(url, 'rb') as f:
            meta = pickle.load(f)
        return meta['vocab_size']

    def forward(self, x):
        return torch.nn.functional.one_hot(x, num_classes=self.get_vocab_size()).permute(1,0).float()

class SimpleEmbeddingEncoder(nn.Module):
    
    def init(self,
             root_dir: str,
             embedding_dim: int):
        super().__init__()

        self.root_dir = root_dir
        self.vocab_size = self.get_vocab_size()
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
    
    def get_vocab_size(self):
        #open the meta file and get the vocab size
        url = Path(self.root_dir) / 'meta.pkl'
        with open(url, 'rb') as f:
            meta = pickle.load(f)
        return meta['vocab_size']

    def forward(self, x):
        return self.embedding(x)
