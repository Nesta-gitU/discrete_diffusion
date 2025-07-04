import torch
from torch import nn
from torch import Tensor

class SimilarityDecoder(nn.Module):
    # THIS FUNCTION WILL NOT HAVE PARAMETERS AT FIRST

    def __init__(self):
        super().__init__()
    
    def forward(self, z_0: Tensor, true_word_embs:Tensor) -> Tensor:
        """
        returns logits for the true word embeddings
        """
        # compare z_1 to e and assign the index of the one to which it is the closest. 
        # return the index.

        similarities = z_0 @ true_word_embs.T #we could technically add a bias term props to gpt
        
        return similarities

class LmHeadDecoder(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int):
        super().__init__()
        
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, z_0: Tensor, true_word_embs) -> Tensor:
        """
        returns logits for the true word embeddings
        """
        return self.lm_head(z_0)