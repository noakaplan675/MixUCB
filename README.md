# MixUCB (RLC 2025)
This repository contains code for the following paper:  "MixUCB: Enhancing Safe Exploration in Contextual Bandits with Human Oversight", by Jinyan Su, Wen Sun, Sarah Dean, Rohan Banerjee, Jiankai Sun, which has been accepted to the Reinforcement Learning Conference (RLC) 2025.

## Installation

Create a conda environment using the provided `requirements.txt` file as follows:

```bash
conda create -n mixucb python=3.10
conda activate mixucb
pip install -r requirements.txt
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```

## Main experimental script

The following script reproduces the experiments in the paper for the four datasets: synthetic, SPANet, heart disease, and MedNIST. It consists of (1) data generation, (2) running the algorithms, (3) generating plots.

```bash
bash run_all.sh
```