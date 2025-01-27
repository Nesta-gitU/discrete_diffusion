import torch
from torch import nn
import pickle
from pathlib import Path

class IdentityEncoder(nn.Module):
    
    def __init__(self,
             vocab_size: int,
             embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        #spoof a nn.Embedding layer without learnable values and initialized as the identity matrix
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = False
        self.embedding.weight.data = torch.eye(self.vocab_size)

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.one_hot(x, num_classes=self.vocab_size).float()

class SimpleEmbeddingEncoder(nn.Module):
    
    def __init__(self,
             vocab_size: int,
             embedding_dim: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    

    def forward(self, x):
        return self.embedding(x)
