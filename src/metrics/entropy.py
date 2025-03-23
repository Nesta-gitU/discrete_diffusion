
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def get_per_token_entropy(logits: torch.Tensor, padding_mask: torch.Tensor = None, temperature: float = 1.0, sum_tokens=False) -> float:
    """
    Computes the average entropy per sequence for a batch of generated sequences.
    
    Args:
        logits (torch.Tensor): Logits from the model with shape (batch_size, seq_len, vocab_size).
        padding_mask (torch.Tensor, optional): Mask indicating padding tokens with shape (batch_size, seq_len).
                                               If provided, padding tokens are excluded from entropy calculation.
        temperature (float): Temperature factor for scaling logits. A higher value flattens the distribution.
        sum_tokens (bool): If True, sum the entropy over tokens (instead of averaging per token) for each sequence.
    
    Returns:
        float: The averaged entropy per sequence.
    """
    # Apply temperature scaling to logits
    scaled_logits = logits / temperature
    # Convert scaled logits to probabilities using softmax
    probs = F.softmax(scaled_logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
    
    # Calculate per-token entropy in nats
    token_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # Shape: (batch_size, seq_len)

    if padding_mask is not None:
        token_entropy = token_entropy * padding_mask
        non_padding_counts = torch.sum(padding_mask, dim=1)  # Shape: (batch_size,)
    else:
        non_padding_counts = torch.full((logits.size(0),), logits.size(1), dtype=logits.dtype, device=logits.device)

    if sum_tokens:
        # Sum token entropies per sequence (no normalization by token count)
        per_sequence_entropy = torch.sum(token_entropy, dim=1)
    else:
        # Average token entropies per sequence
        per_sequence_entropy = torch.sum(token_entropy, dim=1) / (non_padding_counts + 1e-8)
    
    # Average over all sequences
    avg_entropy = torch.mean(per_sequence_entropy) * logits.size(1)

    #this unfortunately doesnt make sense because now the more you pad the lower your entropy
    
    return avg_entropy.item()


