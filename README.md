# Conditional Monge Gap

[![CI](https://github.com/AI4SCR/conditional-monge/actions/workflows/ci.yml/badge.svg)](https://github.com/AI4SCR/conditional-monge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Contents
- [Overview](#overview)
- [Requirements](#systems-and-software-requirements)
- [Installation](#installation-from-pypi)
- [Development installation](#development-setup--installation)
- [Data](#data)
- [Example](#example-usage)
- [Own data instructions](#instructions-for-running-on-your-own-data)
- [Legacy checkpoint loading](#older-checkpoints-loading)
- [Citation](#citation)

## Overview

![](assets/overview.jpg)

An extension of the [Monge Gap](https://proceedings.mlr.press/v202/uscidda23a.html), an approach to estimate transport maps conditionally on arbitrary context vectors. It is based on a two-step training procedure combining an encoder-decoder architecture with an OT estimator. The model is applied to [4i](https://pubmed.ncbi.nlm.nih.gov/30072512/) and [scRNA-seq](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7289078/) datasets.

## Systems and software requirements
Software package requirements and version information can be found in `requirements.txt` and/or the `pyproject.toml`. This package has been tested on Python versions 3.10 and 3.11. 
Hardware requirements enough memory (RAM) to process the data and batches. GPU is not needed but does accelerate computation. This software has been tested on HPCs and local machines (iOS). 

## Installation from PyPI

You can install this package as follows
```sh
pip install cmonge
```
which should take about two minutes on a laptop.

## Development setup & installation
The package environment is managed by [poetry](https://python-poetry.org/docs/managing-environments/). 
```sh
pip install poetry
git clone git@github.com:AI4SCR/conditional-monge.git
cd cmonge
poetry install -v
```

If the installation was successful you can run the tests using pytest
```sh
poetry shell # activate env
pytest
```

## Data

The preprocessed version of the Sciplex3 and 4i datasets can be downloaded [here](https://www.research-collection.ethz.ch/handle/20.500.11850/609681).


## Example usage

You can find a demo config in `configs/demo_config.yml`.
To train an autoencoder and CMonge you can use the following script (also provided in `scripts/demo_train.py`), make sure that the paths in the `scripts/demo_train.py`, `configs/demo_condig.yml`, and `configs/autoencoder-demo.yml` point to the correct data reading and saving locations and well as model checkpoint locations.
```py
from loguru import logger

from cmonge.datasets.conditional_loader import ConditionalDataModule
from cmonge.trainers.ae_trainer import AETrainerModule
from cmonge.trainers.conditional_monge_trainer import ConditionalMongeTrainer
from cmonge.utils import load_config


logger_path = "logs/demo_logs.yml"
config_path = "configs/demo_config.yml"

config = load_config(config_path)
logger.info(f"Experiment: Training model on {config.condition.conditions}")


# Train an AE model to reduce data dimension
config.data.ae = True
config.data.reduction = None
datamodule = ConditionalDataModule(config.data, config.condition)
ae_trainer = AETrainerModule(config.ae)
ae_trainer.train(datamodule)

# Train conditional monge model
config.data.ae = False
config.ae.model.act_fn = "gelu"
config.data.reduction = "ae"
datamodule = ConditionalDataModule(config.data, config.condition, ae_config=config.ae)
trainer = ConditionalMongeTrainer(
    jobid=1, logger_path=logger_path, config=config.model, datamodule=datamodule
)
trainer.train(datamodule)
trainer.evaluate(datamodule)

```
This demo trains a CMonge model using in-distribution data split. First, an autoencoder is trained to reduce the dimensionality of the data, which can be found in `data/dummy_data.h5ad`. This autoencoder model is checkpointed and the resulting checkpoint can also be found in `models/demo`. This example uses the RDkit fingerprint embedding, which uses the SMILES information provided in the `data` directory to compute a numerical embedding, which is saved in  `models/embed/rdkit`. Next, the conditional monge model is trained and evaluated, the results of training and evaluation are saved in the logger file which is defined by the `logger_path` variable. For this example, the logs can thus be found in `logs/demo.yml`. On a 2025 MacBook Air with M4 chip, this demo only takes a few minutes.

## Instructions for running on your own data
For running CMonge on your own data, you probably need to adapt the dataloader to ensure the correct handling of your data. At least you need to implement your own single data loader, of which examples can be found in `cmonge/datasets/single_loader.py`. This single loader needs to be passed to the conditional dataloader, as can be found in `cmonge/datasets/conditional_loader.py`, where some adaptions might be necessary. The conditional dataloader interacts with the CMonge model (and if needed the autoencoder). The hyperparameters of the CMonge neural network and the autoencoder can be defined in the config files.

## Older checkpoints loading
If you want to load model weights of older checkpoints (cmonge-{moa, rdkit}-ood or cmonge-{moa, rdkit}-homogeneous), make sure you are on the tag `cmonge_checkpoint_loading`.

```sh
git checkout cmonge_checkpoint_loading
```

## Citation
If you use the package, please cite:
```bib
@article{driessen2025towards,
  title={Towards generalizable single-cell perturbation modeling via the Conditional Monge Gap},
  author={Driessen, Alice and Harsanyi, Benedek and Rapsomaniki, Marianna and Born, Jannis},
  journal={arXiv preprint arXiv:2504.08328},
  note={Preliminary version at ICLR 2024 Workshop on Machine Learning for Genomics Explorations}
  year={2025}
}
```
