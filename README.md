# MixUCB (RLC 2025)
This repository contains code for the following paper:  "MixUCB: Enhancing Safe Exploration in Contextual Bandits with Human Oversight", by Jinyan Su, Wen Sun, Sarah Dean, Rohan Banerjee, Jiankai Sun, which has been accepted to the Reinforcement Learning Conference (RLC) 2025.

## Installation

Create a conda environment using the provided `env.yml` file:

```bash
conda env create -f env.yml
```

Or create a virtualenv using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Main experimental script

The following script reproduces the experiments in the paper for the four datasets: synthetic, SPANet, heart disease, and MedNIST. It consists of (1) data generation, (2) running the algorithms, (3) generating plots.

```bash
bash run_all.sh
```