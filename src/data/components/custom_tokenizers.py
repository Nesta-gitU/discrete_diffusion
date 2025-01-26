import torch 
import pickle

class CharTokenizer:
    def __init__(self, data):
        chars = sorted(set(data))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.vocab_size = vocab_size
        
        self.string_to_index = { ch:i for i,ch in enumerate(chars) }
        self.index_to_string = { i:ch for i,ch in enumerate(chars) }

    def encode(self, string: str) -> torch.Tensor | list[int]:
        out = [self.string_to_index[ch] for ch in string]
        return out

    def decode(self, indices: torch.Tensor | list[int]) -> str:
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return ''.join([self.index_to_string[i] for i in indices])
    
    @classmethod
    def load(cls, path: str):
        #self.tokenizer = CharTokenizer().load("char_tokenizer_text8.pkl")
        with open(path, 'rb') as f:
            self = pickle.load(f)
        
        return self

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        return self