# Learning Drug Perturbations via Conditional Map Estimators

![](assets/overview.jpg)

An extension of the [Monge Gap](https://arxiv.org/abs/2302.04953), an approach to estimate transport maps conditionally on arbitrary context vectors. It is based on a two-step training procedure combining an encoder-decoder architecture with an OT estimator. The model is applied to [4i](https://pubmed.ncbi.nlm.nih.gov/30072512/) and [scRNA-seq](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7289078/) datasets.

## Environment setup

The environemnt is managed with [poetry](https://python-poetry.org/docs/managing-environments/). We recommend setting up a virtual environment. The code was tested in Python 3.10.
```sh
pip install poetry
poetry install -v
```

If the installation was successful you can run the tests using pytest
```sh
poetry shell # activate env
pytest
```

## Data

The preprocessed version of the Sciplex3 and 4i datasets can be downloaded [here](https://polybox.ethz.ch/index.php/s/RAykIMfDl0qCJaM).


## Example usage

You can find example config in `configs/conditional-monge-sciplex.yml`.
To train an autoencoder model:
```py
from cmonge.datasets.conditional_loader import ConditionalDataModule
from cmonge.trainers.ae_trainer import AETrainerModule
from cmonge.utils import load_config


config_path = Path("configs/conditional-monge-sciplex.yml")
config = load_config(config_path)
config.data.ae = True

datamodule = ConditionalDataModule(config.data, config.condition)
ae_trainer = AETrainerModule(config.ae)

ae_trainer.train(datamodule)
ae_trainer.evaluate(datamodule)
```

To train a conditional monge model:

```py
from cmonge.datasets.conditional_loader import ConditionalDataModule
from cmonge.trainers.conditional_monge_trainer import ConditionalMongeTrainer
from cmonge.utils import load_config

config_path = Path("configs/conditional-monge-sciplex.yml")
logger_path = Path("logs")
config = load_config(config_path)

datamodule = ConditionalDataModule(config.data, config.condition)
trainer = ConditionalMongeTrainer(jobid=1, logger_path=logger_path, config=config.model, datamodule=datamodule)

trainer.train(datamodule)
trainer.evaluate(datamodule)
```

## Citation
If you use the package, please cite:
```bib
@inproceedings{
  harsanyi2024learning,
  title={Learning Drug Perturbations via Conditional Map Estimators},
  author={Benedek Harsanyi and Marianna Rapsomaniki and Jannis Born},
  booktitle={ICLR 2024 Workshop on Machine Learning for Genomics Explorations},
  year={2024},
  url={https://openreview.net/forum?id=FE7lRuwmfI}
}
```
