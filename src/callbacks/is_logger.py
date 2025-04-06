from typing import Any, Optional, Dict, Sequence, Tuple

import torch
from torch import Tensor
import wandb
import pickle

from lightning import Trainer, Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from pathlib import Path
import torch 
import numpy as np
import matplotlib.pyplot as plt
import os
from src.models.time_samplers.time_samplers import TimeSampler

def plot_timedist(self, sampler, out_dir, device, num_points=1000):
    with torch.no_grad():
        # Generate evenly spaced t values
        t_values = torch.linspace(0, 1, num_points).view(-1, 1).to(device)  # Shape: (num_points, 1)
        
        # Compute p(t) at these points
        p_values = sampler.prob(t_values).cpu().numpy().flatten()
        t_values = t_values.cpu().numpy().flatten()

        # Compute the integral using the trapezoidal rule
        dt = 1 / (num_points - 1)  # Step size
        integral = np.trapz(p_values, t_values)  # Numerical integration

        # Plot the learned PDF
        plt.plot(t_values, p_values, label=f"Learned PDF (∫p(t)dt ≈ {integral:.4f})", color="b", linewidth=2)
        plt.xlabel("t")
        plt.ylabel("p(t)")
        plt.title("Learned Probability Density Function")
        plt.legend()
        plt.grid(True)
        plt.show()




def plot_lossdist(self, model, batch, out_dir):
    with torch.no_grad():
        #dont always use full batch, need only like a hundred points
        length = batch.shape[0] if batch.shape[0] < 100 else 100
        #print(batch.shape)
        batch = batch[:length, :]

        losses = []
        t_points = torch.linspace(0, 1, 100).to(batch.device)
        for t in t_points:
            t = t.expand(batch.shape[0], 1, 1)
            #print(t.shape)
            diffusion_loss, _, _, _ = model.forward(t, batch, 
                    compute_diffusion_loss = True, 
                    compute_reconstruction_loss =False, 
                    compute_prior_loss= False, 
                    reconstruction_loss_type="collapse")
            avg_loss = diffusion_loss.mean()
            losses.append(avg_loss.item())

        loss = torch.tensor(losses)
        plt.plot(t_points.detach().cpu().numpy(), loss.detach().cpu().numpy())
        path = os.path.join(out_dir, "losses.png")
        os.makedirs(out_dir, exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(path)
        plt.close() 
    

def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    wandb_logger = None
    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            if wandb_logger is not None:
                raise ValueError('More than one WandbLogger was found in the list of loggers')

            wandb_logger = logger

    return wandb_logger


class IsLogger(Callback):
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

        if not isinstance(pl_module.time_sampler, TimeSampler):
            return 
        
        logger = get_wandb_logger(trainer)
        if logger is None:
            return
        model = pl_module
        device = model.device
        run_name = trainer.logger.experiment.name
        
        out_dir="output"
        model_base_name = f"vdm_{model.model.gamma.__class__.__name__}_{model.model.transform.__class__.__name__}_{run_name}"

        out_dir = os.path.join(out_dir, model_base_name)

        #first plot timedist
        plot_timedist(self, model.time_sampler, out_dir, device)

        #then plot loss dist
        plot_lossdist(self, model, batch, out_dir)






    

