# Create Your Own Trajectory
## towards latent diffusion suitable for text

This repository contains the official implementation of my Masters Thesis titled **Create Your Own Trajectory**.

The code in this repository allows the user to train latent space diffusion models for language generation. Below is a demo of the model NFDM-G-AdaLN, which uses a learned forward process:

[![Watch the demo](https://img.youtube.com/vi/g1cgj2s_idM/maxresdefault.jpg)](https://youtu.be/g1cgj2s_idM?si=WSHn4Q_dii1AKO-4)


There are 3 other demo's here:

- [Diffusion-LM](https://youtu.be/R6lApViDZ0o)
- [MuLN-Rescaled](https://youtu.be/R6lApViDZ0o)
- [MuLAN-Rescaled](https://youtu.be/lWZqdKA9D48)


## Setup 

Install the neccecary packages:

```sh
!pip install -r requirements.txt
```

## Local Usage

Reproduce the training results from a given experiment (NFDM-G-AdaLN in this case):

```sh
python src/train.py experiment=roc_nfdm_new ++restart_from_checkpoint=False
```

A folder will be created with the model name specificied in the experiment file. This folder stores the checkpoints and is used to restart from checkpoint. 

## SLURM Usage

To use on slurm, clone the project on the slurm server. Find the job file corresponding to the model you want to train in discrete_diffusion/job_files. Then batch the job there. 

## Acknowledgments

This codebase is built on top of opensource code from the following two repositories:

- https://github.com/XiangLi1999/Diffusion-LM
- https://github.com/GrigoryBartosh/neural_diffusion

