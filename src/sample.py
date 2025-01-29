from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig




rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.sampling.sampling import sample_code
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

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = DiffusionModule.load_from_checkpoint(cfg.ckpt_path)
    #model: LightningModule = hydra.utils.instantiate(cfg.model)

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
    #trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
    
    get_sde = cfg.get("get_sde", True)
    get_ode = cfg.get("get_ode", True)
    n_steps = cfg.get("n_steps")
    batch_size = cfg.get("batch_size")

    debug = cfg.get("debug", False)
    clamp = cfg.get("clamp", False)
    top_k = cfg.get("top_k", False)
    k = cfg.get("k", 1)
    top_p = cfg.get("top_p", False)
    p = cfg.get("p", 0.9)
    temperature = cfg.get("temperature", 1.0)

    print("getting ode:", get_ode)
    print("getting sde:", get_sde)

    sample_code(model, datamodule, logger, get_sde, get_ode, n_steps, batch_size, debug, clamp, top_k, k, top_p, p, temperature)

    print("done sampling!")
    #it crashes at the end for some reason I have no idea why?

    


@hydra.main(version_base="1.3", config_path="../configs", config_name="sample.yaml")
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
