import torch 
import torch.nn.functional as F

def top_k(logits, k, temperature=1.0):
    """
    top_k sampling
    :param logits: tensor of shape [batch_size, block_size, vocab_size]
    :param k: int, number of words to consider when sampling
    :param temperature: float, temperature
    """
    print(logits.shape, "logit_shape")
    logits = logits / temperature

    probs = F.softmax(logits, dim=-1)

    v, _ = torch.topk(probs, k)
    
    mask = probs >= v[:, :, -1].unsqueeze(-1)
    probs = probs * mask
    print(probs.shape, "probs_shape")

    probs = probs / probs.sum(dim=-1, keepdim=True)
    idx = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).view(*probs.shape[:-1], 1).squeeze(-1) #fold into batch to sample then fold back 
    print(idx.shape, "idx_shape")

    return idx
    

def top_p(logits, p, temperature=1.0):
    """
    top_p sampling
    :param logits: tensor of shape [batch_size, block_size, vocab_size]
    :param p: float, cumulative probability threshold
    :param temperature: float, temperature
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

def argmax_sample(logits):
    """
    argmax sampling
    :param logits: tensor of shape [batch_size, block_size, vocab_size]
    """
    return logits.argmax(dim=-1).squeeze(-1) # shape [batch_size, block_size]
