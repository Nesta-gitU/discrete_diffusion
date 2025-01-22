import torch
from torch import nn
import pickle
from pathlib import Path

class IdentityEncoder(nn.Module):
    
    def __init__(self,
             vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size


    def forward(self, x):
        return torch.nn.functional.one_hot(x, num_classes=self.vocab_size).permute(1,0).float()

class SimpleEmbeddingEncoder(nn.Module):
    
    def __init__(self,
             vocab_size: int,
             embedding_dim: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
    

    def forward(self, x):
        return self.embedding(x)
