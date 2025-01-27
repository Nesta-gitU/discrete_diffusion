from typing import Any, Optional, Dict, Sequence, Tuple

import torch
from torch import Tensor
import wandb
import pickle

from lightning import Trainer, Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from pathlib import Path
from src.utils.differential_equations import sde_drift, solve_de



def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    wandb_logger = None
    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            if wandb_logger is not None:
                raise ValueError('More than one WandbLogger was found in the list of loggers')

            wandb_logger = logger

    return wandb_logger


class CollapseLogger(Callback):
    def __init__(
        self,
        log_train_every_n_steps: Optional[int] = None,
    ):
        self.log_train_every_n_steps = log_train_every_n_steps

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.log_train_every_n_steps is not None:
            if not (trainer.global_step % self.log_train_every_n_steps == 0):
                return
        
        logger = get_wandb_logger(trainer)
        if logger is None:
            return
        
        with torch.no_grad():
            #get embeddings from the encoder
            embeddings = pl_module.model.encoder.embedding.weight

            # Normalize the embeddings to unit vectors
            embeddings = torch.nn.functional.normalize(embeddings, dim=0)  # Shape remains [embedding_dim, vocab_size]

            # Compute cosine similarities (dot products of normalized embeddings)
            cosine_similarities = torch.matmul(embeddings.T, embeddings)  # Shape: [vocab_size, vocab_size]

            # Exclude self-similarities by subtracting the diagonal
            vocab_size = embeddings.size(1)
            mask = ~torch.eye(vocab_size, dtype=torch.bool, device=embeddings.device)  # Mask for non-diagonal elements
            pairwise_cosines = cosine_similarities[mask]  # Extract only off-diagonal elements

            # Compute ANI
            ani = pairwise_cosines.mean().item()

            logger.log("train_collapse/ani_score", ani, on_step=True, prog_bar=False)
                

    

