from typing import Any, Optional, Dict, Sequence, Tuple

import torch
from torch import Tensor
import wandb
import pickle

from lightning import Trainer, Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from pathlib import Path
from sampling.sampling import sample_from_diffusion, idx_to_words, plot_gamma, visualize_path
from types import SimpleNamespace
import time

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
        modes: Optional[Sequence[str]] = ["star", "marginal"],
        vizualize: Optional[bool] = False,
        no_epoch_logging: Optional[bool] = False,
    ):
        super().__init__()
        self._odeint_params = odeint_params
        self._n_steps = n_steps
        self.batch_size = batch_size
        self.log_train_every_n_steps = log_train_every_n_steps # to not log train set this to none
        self._sample_seed = sample_seed
        
        self.text_table = wandb.Table(columns=["epoch", "global_step", modes[0], modes[1]])
        self.root_dir = root_dir

        self.modes = modes
        self.no_epoch_logging = no_epoch_logging

        

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.log_train_every_n_steps is not None:
            if not (trainer.global_step % self.log_train_every_n_steps == 0):
                return
        
        logger = get_wandb_logger(trainer)
        if logger is None:
            return
       
        self.sample_code(trainer, pl_module, logger)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if trainer.global_step < 20:
            return
        if self.no_epoch_logging:
            return

        logger = get_wandb_logger(trainer)
        if logger is None:
            return
       

        self.sample_code(trainer, pl_module, logger)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module) -> None:
        if trainer.global_step < 20:
            return
        logger = get_wandb_logger(trainer)
        if logger is None:
            return
       

        self.sample_code(trainer, pl_module, logger)

    def sample_code(self, trainer, pl_module, logger):
        if trainer.global_step < 20:
            return
        block_size = trainer.datamodule.data_train.block_size
        hidden_size = pl_module.ema.module.pred.model.in_channels

        model = pl_module.ema.module
        model.eval()

        run_name = logger.experiment.name

        if hasattr(model, 'gamma'):
            plot_gamma(model,out_dir="output",model_base_name=f"vdm_{model.gamma.__class__.__name__}_{model.transform.__class__.__name__}_{run_name}", 
                    batch_size=self.batch_size, block_size=block_size, hidden_size=hidden_size)
        
        tokenizer = trainer.datamodule.tokenizer
        
        w_list = []
        time_sampler_args = SimpleNamespace(
            uniform=True,
            use_default_nfdm=False,
            time_sampler = pl_module.time_sampler
        )

        for mode in self.modes:
            latent, words, path = sample_from_diffusion(model, self.batch_size, block_size, hidden_size, _n_steps=self._n_steps, clamping=False, do_top_k=False, k=10, do_top_p=False, p=0.9, temperature=1.0, sampling_mode=mode, time_sampler_args=time_sampler_args)
            visualize_path(model, path, tokenizer, mode=mode)
            words = idx_to_words(words, tokenizer)
            w_list.append(words)

        #try
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        self.text_table = wandb.Table(
            columns=self.text_table.columns, data=self.text_table.data
        )
        
        self.text_table.add_data(str(epoch), str(global_step), str(w_list[0]), str(w_list[1]))

        logger.experiment.log({"generated_samples": self.text_table})
    


    

