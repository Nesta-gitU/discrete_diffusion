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
    z = torch.randn(1, block_size, hidden_size)
    #z = z.expand(300, -1, -1) 
    #although the fact that for a different t at each time we get a bunch of smooth lines seems like a really bad sign to me, means z has almost no effect?

    with torch.no_grad():
        model.to("cpu")
        t = torch.linspace(0, 1, 300)[:, None].to("cpu")
        t=t.unsqueeze(-1)

        context = model.context.sample_context(z) #slightly incorrect if using NN but with VAE its fine
        context = context.expand(300, -1, -1) 
        
        if context is None:
            gmm, _ = model.gamma(t)
        else:
            gmm, _ = model.gamma(t, context)
        
        gmm = gmm.squeeze(-1)

        #MULAN paper plots mean and variance of snr over time 

        #print the order of the sequence wise gamma. 
        print("the order gamma is: ")
        #shape of gmm in the case where an order is relevant is [300, 64, 1]
        #now it is not data dependent so we can ignore batch dim and just look at the order at 0.5 I guess
        gmm_for_order = gmm[150]
        sorted_gmm = torch.argsort(gmm_for_order, descending=True) #high to low so we know what is being maksed out first 
        print(sorted_gmm, "sorted gmm")
        # [64, 1] I want the order where I print (ggm_dim, )

        plt.plot(gmm)
        plt.savefig(f"{outpath}/gamma_plot.png")
        plt.close()

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
