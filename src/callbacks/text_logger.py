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


class TextLogger(Callback):
    def __init__(
        self,
        odeint_params: Dict[str, Any],
        root_dir: Optional[str],
        n_steps: int = 300,
        batch_size: int = 25,
        log_train_every_n_steps: Optional[int] = None,
        sample_seed: Optional[int] = 42,
        get_sde: Optional[bool] = True,
        get_ode: Optional[bool] = True,
    ):
        self._odeint_params = odeint_params
        self._n_steps = n_steps
        self.batch_size = batch_size
        self.log_train_every_n_steps = log_train_every_n_steps # to not log train set this to none
        self._sample_seed = sample_seed
        
        self.text_table = wandb.Table(columns=["epoch", "global_step", "text_ode", "text_sde"])
        self.root_dir = root_dir

        self.get_sde = get_sde
        self.get_ode = get_ode

    def idx_to_words(self, index):
        meta_url = Path(self.root_dir) / "data" / "text8" / "meta.pkl" #TODO: Change this to a more general path
        with open(meta_url, 'rb') as f:
            meta = pickle.load(f)

        itos = meta['itos']
        decoded_texts = []
        for sequence in index:
            words = [itos[idx.item()] for idx in sequence]
            decoded_texts.append(" ".join(words))  # Combine words into a single string
            #TODO: Improve punctuation handling

        return decoded_texts
    
    def sample_from_diffusion(self, module, batch_size):
        # sample batch size random z's, these must have the shape equal to our datapoints so that is [batch_size, block_size, n_embed]
        # I should be able to get the from the input_size and block_size of the transformer model
        block_size = module.model.pred.model.block_size
        hidden_size = module.model.pred.model.hidden_size

        #z = torch.randn(torch.Size(batch_size, block_size, hidden_size))
        z = torch.randn(batch_size, block_size, hidden_size)
        z = z.to(module.device)

        sde_indices = None
        ode_indices = None

        if self.get_sde:
            sde_solved, path = solve_de(z, 0, 1, self._n_steps, module, 'sde')
            #decode the sde solve back into words 
            sde_logits = module.model.decoder(sde_solved, module.model.encoder.embedding.weight)
            sde_indices = sde_logits.argmax(dim=-1).squeeze(-1)
        if self.get_ode:
            ode_solved, path = solve_de(z, 0, 1, self._n_steps, module, 'ode')
            #decode the ode solve back into words
            ode_logits = module.model.decoder(ode_solved, module.model.encoder.embedding.weight)
            ode_indices = ode_logits.argmax(dim=-1).squeeze(-1)
        return sde_indices, ode_indices

        

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.log_train_every_n_steps is not None:
            if not (trainer.global_step % self.log_train_every_n_steps == 0):
                return
        
        logger = get_wandb_logger(trainer)
        if logger is None:
            return
       

        words_sde, words_ode = self.sample_from_diffusion(pl_module, self.batch_size)

        if words_sde is None:
            w_sde = None
        else:
            w_sde = self.idx_to_words(words_sde)
        if words_ode is None:
            w_ode = None
        else:
            w_ode = self.idx_to_words(words_ode)

        epoch = trainer.current_epoch
        global_step = trainer.global_step

        self.text_table = wandb.Table(
            columns=self.text_table.columns, data=self.text_table.data
        )

        self.text_table.add_data(str(epoch), str(global_step), str(w_ode), str(w_sde))

        logger.experiment.log({"generated_samples": self.text_table})
        #delete the artifact made the previous step to save space

    @rank_zero_only
    def on_train_end(self, trainer, pl_module) -> None:
        if trainer.global_step < 20:
            return
        logger = get_wandb_logger(trainer)
        if logger is None:
            return

        #sample one final time 
        words_sde, words_ode = self.sample_from_diffusion(pl_module, self.batch_size)

        if words_sde is None:
            w_sde = None
        else:
            w_sde = self.idx_to_words(words_sde)
        if words_ode is None:
            w_ode = None
        else:
            w_ode = self.idx_to_words(words_ode)

        epoch = trainer.current_epoch
        global_step = trainer.global_step

        self.text_table = wandb.Table(
            columns=self.text_table.columns, data=self.text_table.data
        )

        self.text_table.add_data(str(epoch), str(global_step), str(w_ode), str(w_sde))

        

    

