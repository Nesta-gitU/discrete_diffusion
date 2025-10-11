# Create Your Own Trajectory
## towards latent diffusion suitable for text

This repository contains the official implementation of the work titled **Create Your Own Trajectory**.

## Setup 

Install the neccecary packages:

```sh
!pip install -r requirements.txt
```

## Local Usage

Reproduce the training results from a given experiment (NFDM in this case):

```sh
python src/train.py experiment=roc_nfdm_new ++restart_from_checkpoint=False
```

A folder will be created with the model name specificied in the experiment file. This folder stores the checkpoints and is used to restart from checkpoint. 

## SLURM Usage

To use on slurm, clone the project on the slurm server. Find the job file corresponding to the model you want to train in discrete_diffusion/job_files. Then batch the job there. 

## Acknowledgments

This codebase is built on top of opensource code from the following repository:

- https://github.com/XiangLi1999/Diffusion-LM

