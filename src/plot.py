from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from sklearn.metrics import mean_tweedie_deviance





rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.sampling.sampling import sample_code
from src.callbacks.is_logger import plot_timedist, plot_lossdist
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

from src.models.language_diff_module import DiffusionModule
from src.utils.checkpoint_loading import get_latest_checkpoint, get_latest_run_folder

log = RankedLogger(__name__, rank_zero_only=True)

from types import SimpleNamespace

import os, sys, glob
import json
from src.sampling.sampling import sample_code#improved_diffusion.sampling.sampling import sample_code

from src.metrics.ppl_under_ar import main as main_ppl
# full_lst = glob.glob('diff_models_synth128*')
# full_lst = glob.glob('diff_models_synth32*')
# full_lst = glob.glob('diff_models_synth32_3_rand16*')
# full_lst = glob.glob('diff_models_synth_rand_16_trans_lr_1e-5_long_Lsimple')
#create an args parser instead of using sys.argv
import argparse
from src.metrics.mauve import print_mauve
from src.metrics.utils import get_preprocessed_data, file_to_list, metric_to_std
from src.metrics.diversity import compute_diversity, compute_memorization
from src.metrics.hf_ppl_under_ar import compute_perplexity
from src.models.ndm.components.context import NoneContext
import numpy as np
import matplotlib.pyplot as plt
import torch 

import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_gamma_snr_with_colormap(
    model, z, T=300, outpath=".", device="cuda:0", cmap_name="viridis"
):
    with torch.no_grad():
        """
        model: your diffusion model
        z:     [N, D, hidden_size]  latent samples
        T:     number of t-steps
        device:"cuda:0" or "cpu"
        cmap_name: name of a matplotlib colormap (we're using "viridis")
        """
        os.makedirs(outpath, exist_ok=True)
        model.to(device)
        z = z.to(device)

        N, D, _ = z.shape
        # build t-grid
        t = torch.linspace(0, 1, T, device=device).view(T, 1, 1).expand(T, N, 1)
        t_flat = t.reshape(T * N, 1)

        # sample + repeat contexts
        if not isinstance(model.context, NoneContext):

            ctx = model.context.sample_context(z)                # [N, C]
            ctx_flat = ctx.unsqueeze(0).expand(T, N, -1).reshape(T * N, -1)

            # run gamma
            gmm_flat, _ = model.gamma(t_flat, ctx_flat)          # [T*N, D, 1]
        else:
            gmm_flat, _ = model.gamma(t_flat)                    # [T*N, D, 1]
            
        gmm_flat = gmm_flat.squeeze(-1)
        gmm = gmm_flat.view(T, N, D)                         # [T, N, D]

        # compute SNR
        a2   = model.gamma.alpha_2(gmm)
        s2   = model.gamma.sigma_2(gmm)
        snr  = a2 / s2                                       # [T, N, D]

        # stats over N
        γ_mean = gmm.mean(dim=1).cpu()                       # [T, D]
        γ_std  = gmm.std(dim=1).cpu()                        # [T, D]
        μ_snr  = snr.mean(dim=1).cpu()                       # [T, D]
        var_snr= snr.var(dim=1).cpu()                        # [T, D]

        ts = torch.linspace(0, 1, T).cpu().numpy()

        # prepare viridis colormap
        cmap   = plt.get_cmap(cmap_name)
        colors = [cmap(i/(D-1)) for i in range(D)]

        def _plot_and_save(y, title, fname, ylabel):
            plt.figure(figsize=(6,4))
            for d in range(D):
                plt.plot(ts, y[:, d], color=colors[d], alpha=0.8)
            #plt.title(title)
            plt.xlabel("t")
            plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(os.path.join(outpath, fname))
            plt.close()

        # 4 separate plots
        _plot_and_save(γ_mean,  "Mean γ(t) per latent dim",       "gamma_mean.png", "γ")
        _plot_and_save(γ_std,   "Std  γ(t) per latent dim",       "gamma_std.png",  "γ")
        _plot_and_save(μ_snr,   "Mean SNR(t) per latent dim",     "snr_mean.png",   "SNR")
        _plot_and_save(var_snr, "Var  SNR(t) per latent dim",     "snr_var.png",    "SNR")

        # gradient legend
        gradient = np.linspace(0, 1, D).reshape(1, D)
        plt.figure(figsize=(8,1))
        plt.imshow(gradient, aspect="auto", cmap=cmap)
        plt.xticks([0, D-1], [0, D-1])
        plt.yticks([])
        plt.xlabel("latent‐index →")
        plt.title("Color‐map: latent dim index 0 → 63")
        plt.tight_layout()
        plt.savefig(os.path.join(outpath, "colormap_legend.png"))
        plt.close()

        print(f"Saved 5 figures under {outpath}/:")
        print("  • gamma_mean.png       (mean γ)")
        print("  • gamma_std.png        (std  γ)")
        print("  • snr_mean.png         (mean SNR)")
        print("  • snr_var.png          (var  SNR)")
        print("  • colormap_legend.png  (index→color mapping)")


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    if not cfg.ckpt_path:
        log.info("Finding latest checkpoint...")

        run_folder = get_latest_run_folder(cfg.model_name)
        print(run_folder)
        print(cfg.model_name)
        cfg.ckpt_path = get_latest_checkpoint(run_folder)  # Use model_name from config

    assert cfg.ckpt_path, "Checkpoint path must be specified or found automatically!"

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model_lightning = DiffusionModule.load_from_checkpoint(cfg.ckpt_path)
    model = model_lightning.ema.module #EMA model is the one we want to sample from
    #model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.eval()
    if not hasattr(model, "context"):
        model.context = NoneContext(None)

    if not hasattr(model.gamma, "around_reference"):
        model.gamma.around_reference = False
    datamodule.setup(stage="fit")

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    #####################
    #do plotting stuff -> reproduce those plots from the mulan paper
    #####################

    # Create the directory if it doesn't exist
    out_dir = 'mulan_plots'   
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model_base_name = os.path.basename(os.path.split(cfg.ckpt_path)[0]) + f'.{os.path.split(cfg.ckpt_path)[1]}' + '.' + "roc" + f'.{cfg.model_name}'
    outpath = os.path.join(out_dir, f"{model_base_name}")
    os.makedirs(outpath, exist_ok=True)

    block_size = datamodule.data_train.block_size
    hidden_size = model.pred.model.in_channels

    #z = torch.randn(300, block_size, hidden_size) #this means each t has a different z, but I want to see all t for some z 
    N = 500
    T = 300

    z = torch.randn(N, block_size, hidden_size)
    #z = z.expand(300, -1, -1) 
    #although the fact that for a different t at each time we get a bunch of smooth lines seems like a really bad sign to me, means z has almost no effect?
   
    plot_gamma_snr_with_colormap(model, z, T=T, outpath=outpath, device="cuda")

       

    #it crashes at the end for some reason I have no idea why?
    return None, None

    


@hydra.main(version_base="1.3", config_path="../configs", config_name="plot.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
