import os
import glob
import time

CHECKPOINT_BASE_DIR = "logs/diffusion_logs"

def get_model_checkpoint_dir(model_name):
    """Returns the base directory for a model's checkpoints."""
    return os.path.join(CHECKPOINT_BASE_DIR, model_name)

def get_latest_run_folder(model_name):
    """Finds the most recent timestamped folder for the given model."""
    model_dir = get_model_checkpoint_dir(model_name)
    run_folders = sorted(glob.glob(f"{model_dir}/*/"), key=os.path.getmtime)
    print(run_folders)
    return run_folders[-1] if run_folders else None

def get_latest_checkpoint(run_folder):
    """Finds the most recent checkpoint in the given run folder."""
    if not run_folder:
        print("No run folder found!")
        return None
    #checkpoints = sorted(glob.glob(f"{run_folder}/*.ckpt"))
    #this breaks if we save last, since it will go last, last-v1 etc so last will always be first in sorted (letter before numbers)
    #checkpoints = sorted(glob.glob(f"{run_folder}/*.ckpt"), key=os.path.getmtime)
    checkpoints = sorted(glob.glob(f"{run_folder}/*.ckpt"), key=os.path.getmtime)

    if not checkpoints:
        print("No checkpoints found!")
    else:
        print(f"the latests checkpoint is {checkpoints[-1]}")

    return checkpoints[-1] if checkpoints else None

def get_checkpoint_path(cfg):
    """
    Determines the correct checkpoint path and logging directory.
    - If `restart_from_checkpoint=True`, it finds the latest run and checkpoint.
    - If `specific_run` is provided, it resumes from that folder.
    - Otherwise, creates a new timestamped run folder.
    """
    model_name = cfg.get("model_name")
    restart = cfg.get("restart_from_checkpoint", False)
    specific_run = cfg.get("specific_run", None)

    model_dir = get_model_checkpoint_dir(model_name)

    # Determine which folder to use
    if restart:
        run_folder = os.path.join(model_dir, specific_run) if specific_run else get_latest_run_folder(model_name)
        if not run_folder:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_folder = os.path.join(model_dir, timestamp)
            os.makedirs(run_folder, exist_ok=True)
            print(f"Creating new run folder: {run_folder}")    
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_folder = os.path.join(model_dir, timestamp)
        os.makedirs(run_folder, exist_ok=True)

    # Find latest checkpoint in that folder
    checkpoint_path = get_latest_checkpoint(run_folder) if restart else None

    return checkpoint_path, run_folder