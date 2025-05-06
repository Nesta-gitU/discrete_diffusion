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



def generate_samples_mine(args, model, datamodule, batch_size, out_dir):
    model_base_name = args.model_base_name 
    #tokenizer = load_tokenizer(training_args['modality'], training_args['experiment'], os.path.split(tgt)[0])
    tokenizer = datamodule.tokenizer
    block_size = datamodule.data_train.block_size
    hidden_size = model.pred.model.in_channels
    #block_size = training_args['image_size'] ** 2
    #hidden_size = training_args['in_channel']
    #args2 = create_argparser().parse_args()
    #model, diffusion = create_model_and_diffusion(
    #    **training_args
    #)
    

    #run the sampling code in a loop and return entropy and other metric lists 
    entropy_list = []
    outpath_list = []
    for mode in args.sampling_mode:
        outpath = os.path.join(out_dir, f"{model_base_name}.samples_{mode}.json")
        outpath_list.append(outpath)
        output_cands = glob.glob(outpath)

        if len(output_cands) > 0 and not args.rerun:
            entropy = None
        else:
            entropy  = sample_code(model = model, 
                        tokenizer=tokenizer, 
                        batch_size=batch_size, 
                        block_size=block_size,
                        hidden_size=hidden_size, 
                        out_dir = out_dir,
                        model_base_name = model_base_name,
                        sampling_mode=mode,
                        n_steps=args.n_steps, 
                        clamping=args.clamping, 
                        do_top_k=args.do_top_k, 
                        k=args.top_k, 
                        do_top_p=args.do_top_p, 
                        p=args.top_p, 
                        temperature=args.temperature, 
                        compute_ani=args.compute_ani, 
                        debug=True, 
                        num_samples=args.num_samples)

        entropy_list.append(entropy)

    return entropy_list, outpath_list

class DotDict(dict):
    """Dictionary with dot notation access to attributes."""
    def __getattr__(self, key):
        try:
            value = self[key]
            # Optionally, recursively convert inner dictionaries.
            if isinstance(value, dict):
                value = DotDict(value)
            return value
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")



def generate_samples_test_set(args, datamodule, tokenizer, num_samples, out_dir):
    decoded_texts = get_preprocessed_data('test', datamodule, tokenizer, num_samples)
    
    outpath = os.path.join(out_dir, f"test_samples.json")

    with open(outpath, 'w') as f:
        for sample in decoded_texts:
            print(json.dumps(sample), file=f)


    return outpath



def sample_here(args, model, modality, datamodule):
    out_dir = 'generation_outputs'   

    model_base_name = args.model_base_name 
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.setting == 'reference_mode':
        print("generating in reference mode")
        entropy = None #entorpy is a function of model uncertainty 
        outpath = generate_samples_test_set(args, datamodule, datamodule.tokenizer, args.num_samples, out_dir)
        out_path_list = [outpath]
        entropy_list = [entropy]
    else:
        print("generating with nfdm")
        entropy_list, out_path_list = generate_samples_mine(args, model, datamodule, args.batch_size, out_dir)
    
    #get the perplexity of the generated samples
    if modality == 'e2e': 
        print('using e2e-tgt model gpt2-large')
        model_name_path = "openai-community/gpt2-large" #TODO: change this to the correct model
    elif modality == 'roc':
        model_name_path = "openai-community/gpt2-large" #openai-community/gpt2, msintaha/gpt2-finetuned-rocstories
    else:
        raise ValueError('modality not supported')

    if args.setting == 'reference_mode':
        args.sampling_mode = ["reference_mode"]

    
    for i in range(len(out_path_list)):
        name = args.sampling_mode[i]
        out_path2 = out_path_list[i]
        entropy = entropy_list[i]
        print("running eval loop!")

        #old perplexity computation
        custom_args = {
                "model":model,
                "model_name_or_path": model_name_path,
                "input_text": out_path2,
                "mode": "eval",
                "modality": modality,
                "experiment": "random",
                "std_split": args.std_split # support this in the main_ppl function, 
                # the idea is that we generate 5k samples and then split them into 5 splits of 1k samples each and report the std of the 5 splits
            }
        
        custom_args = DotDict(custom_args)
        print("starting ppl")
        perplexity_mean, perplexity_std = main_ppl(custom_args)
        print("ending old ppl with mean and std", perplexity_mean, perplexity_std)

        # new metric computation
        # first get the generated samples to a list 
        #file_to_list(text_path, datamodule, tokenizer, setting):
        all_texts_list, human_references, human_references_train = file_to_list(out_path2, datamodule, datamodule.tokenizer, args.setting)
        # compute metrics using metric_to_std
        mean_mauve, std_mauve = metric_to_std(all_texts_list, human_references, print_mauve, args.std_split, args.num_samples)
        mean_diversity, std_diversity = metric_to_std(all_texts_list, human_references_train, compute_diversity, args.std_split, args.num_samples)
        mean_memorization, std_memorization = metric_to_std(all_texts_list, human_references_train, compute_memorization, args.std_split, args.num_samples)
        mean_ppl, std_ppl = metric_to_std(all_texts_list, human_references, compute_perplexity, args.std_split, args.num_samples)

        #store a json file in the output directory with the results of entropy and perplexity
        results = {
            "entropy": entropy,
            "perplexity_mean": perplexity_mean,
            "perplexity_std": perplexity_std,
            "mauve_mean": mean_mauve,
            "mauve_std": std_mauve,	
            "diversity_mean": mean_diversity,
            "diversity_std": std_diversity,
            "memorization_mean": mean_memorization,
            "memorization_std": std_memorization,
            "new_ppl_mean": mean_ppl,
            "new_ppl_std": std_ppl,
            "num_samples": args.num_samples,
            "split": args.std_split,
            "clamp": args.clamping,
            "n_steps": args.n_steps,
            "sampling_mode": name,
        }

        

        with open(os.path.join(out_dir, f"{model_base_name}.sample_results_{name}.json"), 'w') as f:
            print("writing results to ", os.path.join(out_dir, f"{model_base_name}.sample_results_{name}.json"))
            json.dump(results, f)
    


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

    # For sampling_mode, the config may be provided as a comma‐separated string.
    # If you want multiple modes, pass them as a comma‐separated string like "ode,sde"
    sampling_mode_cfg = cfg.get("sampling_mode", ["marginal", "star"])
    if isinstance(sampling_mode_cfg, str):
        sampling_mode = [x.strip() for x in sampling_mode_cfg.split(",")]
    else:
        sampling_mode = sampling_mode_cfg  # assuming it's already a list

    args = SimpleNamespace()

    args.sampling_mode = sampling_mode
    args.n_steps = cfg.get("n_steps", 2000)
    args.clamping = cfg.get("clamping", False)
    args.do_top_k = cfg.get("do_top_k", False)
    args.top_k = cfg.get("top_k", 40)
    args.do_top_p = cfg.get("do_top_p", False)
    args.top_p = cfg.get("top_p", -1)
    args.temperature = cfg.get("temperature", 1.0)
    args.pattern_ = cfg.get("pattern_", "ema")
    args.get_mauve = cfg.get("get_mauve", True)
    args.num_samples = cfg.get("num_samples", 128)
    args.batch_size = cfg.get("batch_size", 41)
    args.compute_ani = cfg.get("compute_ani", False)
    args.rerun = cfg.get("rerun", True)
    args.setting = cfg.get("setting", "test_mode")
    print("setting is ", args.setting)
    args.modality = cfg.get("dataset", "roc")
    args.plot_time_and_loss = cfg.get("plot_time_and_loss", False)

    #get model name by checkpoint so that it includes the epoch at which it was taken 
    args.model_base_name = os.path.basename(os.path.split(cfg.ckpt_path)[0]) + f'.{os.path.split(cfg.ckpt_path)[1]}' + '.' + args.modality + f'.{cfg.model_name}'
    print("decoding for ", args.model_base_name)

    #set the arguments for the different mode's
    if args.setting == 'test_mode':
        args.std_split = 1
        args.num_samples = 64
        args.batch_size = 64
    elif args.setting == 'full_mode':
        args.std_split = 5
        args.num_samples = 5000
        args.batch_size = 1000
    elif args.setting == 'reference_mode':
        args.std_split = 5
        args.num_samples = 5000
        args.batch_size = 2500
        #but this on will be used later to generate by sampling from the dataset instead of the model
        args.decode_theirs = False

    #if args.plot_time_and_loss:
    if False:
        out_dir="output"
        model_base_name = args.model_base_name
        out_dir = os.path.join(out_dir, model_base_name)

        time_sampler = model_lightning.time_sampler.to(model_lightning.device)
        print(time_sampler.device)
        print(model_lightning.device)
        time_sampler.device = model_lightning.device

        plot_timedist(time_sampler, out_dir, model_lightning.device)
        

        #then plot loss dist
        batch = next(iter(datamodule.train_dataloader()))
        print(batch.shape)
        print(batch.device)
        batch = batch.to(model_lightning.device)
        plot_lossdist(model_lightning, batch, out_dir)

    sample_here(args, model, args.modality, datamodule)



    #then here function call to the method I have in the other code base! that should be sampling done so tmr test it on the one I have working now,
    #also tmr make model name a command line argument on snellius so I can easily test different models without multiple scripts. 

    #what was the problem with this? ah yeah I need it to call the full batch_decode file instead, also make sure we arent in inference mode that would be bad!
    #also dont fucking compile your model, it ruins checkpoint loading!!!

    #the other problem is file paths for checkpoint paths, currently it just all goes in the logs folder but instead I want it to be similar to the other one,
    #where we have a diffusion_checkpoints folder that has folders with checkpoints for timed runs of each model, you should be able to restart from a checkpoint with just a model name or config,
    #and if a folder for that config exists and you choose restart_from_checkpoint == true it should start from a checkpoint in that folder automatically, and it should then continue logging to that same 
    #checkpoint. it should add the step to the checkpoint name. and then logging file here should just have a folder path as input and it just takes the most recent timestammeped folder for checkpoints and the most 
    #recent checkpoint in there. That should maximize my manual labour timesavings. 

    print("done sampling!")
    #it crashes at the end for some reason I have no idea why?
    return None, None

    


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
