from typing import Dict, Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import scanpy as sc
from dotmap import DotMap
from loguru import logger

from cmonge.datasets.single_loader import AbstractDataModule, DataModuleFactory
from cmonge.trainers.ae_trainer import AETrainerModule


class ConditionalDataModule:
    """Class for conditional dataloaders.

    Available modes:
    random - Conditions are randomly split into train and test, train on all data from some conditions and test on all data from others.
    homogeneous - Train on all conditions but keep out some data from every condition for testing.
    """

    def __init__(
        self,
        data_config: DotMap,
        condition_config: DotMap,
        ae_config: Optional[DotMap] = None,
    ):
        self.data_config = data_config
        self.name = f"cond-{data_config.name}"
        assert condition_config.mode in ["extrapolate", "homogeneous", "custom"]
        self.condition_config = condition_config
        self.ae_config = ae_config
        if self.condition_config.mode == "homogeneous":
            self.drug_condition = "homogeneous"
        else:
            self.drug_condition = data_config.drug_condition
        self.batch_size = self.data_config.batch_size
        self.key = jax.random.PRNGKey(data_config.seed)
        self.meta_loader = DataModuleFactory[data_config.name]
        self.loaders: Dict[str, AbstractDataModule] = {}

        # Instead of loading the AnnData object from disk for every loader
        # we just pass it by reference from the "parent" class
        if self.data_config.file_path:
            self.adata = sc.read_h5ad(self.data_config.file_path)
            self.data_config.parent = self.adata
        if self.data_config.reduction:
            self.reducer()
        self.splitter()

    def sample_condition(self, split_type: str):
        """Return a random condition based on the train/valid/test split."""
        assert split_type in ["train", "valid", "test"]
        if split_type == "train":
            conditions = self.train_conditions
        elif split_type == "valid":
            conditions = self.valid_conditions
        elif split_type == "test":
            conditions = self.test_conditions

        idxs = jnp.arange(len(conditions))
        k1, self.key = jax.random.split(self.key, 2)
        idx = jax.random.choice(k1, idxs)
        return conditions[idx]

    def set_conditions(self):
        self.conditions = list(set(self.condition_config.conditions))

        self.train_conditions = []
        self.valid_conditions = []
        self.test_conditions = []
        for cond, loader in self.loaders.items():
            if len(loader.target_train_cells) > 0:
                self.train_conditions.append(cond)
            if len(loader.target_valid_cells) > 0:
                self.valid_conditions.append(cond)
            if len(loader.target_test_cells) > 0:
                self.test_conditions.append(cond)

    def setup_single_loader(self, condition: str):
        """Iitializes a loader instance for a single condition."""
        logger.info(f"Setting up datamodules for {condition}")
        self.data_config.drug_condition = condition
        loader_instance = self.meta_loader(self.data_config)
        self.loaders[condition] = loader_instance

    def decoder(self, x):
        loader = list(self.loaders.values())[0]
        return loader.decoder(x)

    @property
    def marker_idx(self):
        loader = list(self.loaders.values())[0]
        return loader.marker_idx

    def reducer(self):
        trainer = AETrainerModule(self.ae_config)
        print(self.name, self.drug_condition)
        trainer.load_model(self.name, self.drug_condition)
        self.data_config.parent_reducer = trainer

    def splitter(self):
        """Performs trian/test/valid splits based on the mode variable."""
        if self.condition_config.mode == "extrapolate":
            for condition in self.condition_config.conditions:
                if condition in self.condition_config.ood:
                    self.data_config.split = self.condition_config.ood_split or [0, 1]
                else:
                    self.data_config.split = self.condition_config.split
                self.setup_single_loader(condition)

        elif self.condition_config.mode == "homogeneous":
            self.drug_condition = "homogeneous"
            self.data_config.split = self.condition_config.split
            for condition in self.condition_config.conditions:
                self.setup_single_loader(condition)

        elif self.condition_config.mode == "custom":
            for condition, split in self.condition_config.conditions.items():
                self.data_config.split = split
                self.setup_single_loader(condition)

        self.set_conditions()

    def get_loaders_by_type(self, split_type: str):
        assert split_type in ["train", "valid", "test"]
        type_to_conditions = {
            "train": self.train_conditions,
            "valid": self.valid_conditions,
            "test": self.test_conditions,
        }

        cond_to_loaders = {
            cond: loader.get_loaders_by_type(split_type)
            for cond, loader in self.loaders.items()
            if cond in type_to_conditions[split_type]
        }
        if self.data_config.ae:

            def collapser_iter(split_type: str) -> Iterator[jnp.ndarray]:
                while len(cond_to_loaders) > 0:
                    condition = self.sample_condition(split_type)
                    loader = cond_to_loaders.get(condition, None)
                    try:
                        yield next(loader)
                    except StopIteration:
                        del cond_to_loaders[condition]
                    except TypeError:
                        pass

            loader = collapser_iter(split_type)
            return loader
        else:
            return cond_to_loaders

    def train_dataloaders(
        self,
    ) -> Dict[
        str,
        Iterator[jnp.ndarray] | Tuple[Iterator[jnp.ndarray]] | Iterator[jnp.ndarray],
    ]:
        logger.info("Setting up train dataloaders.")
        return self.get_loaders_by_type("train")

    def valid_dataloaders(
        self,
    ) -> Dict[
        str,
        Iterator[jnp.ndarray] | Tuple[Iterator[jnp.ndarray]] | Iterator[jnp.ndarray],
    ]:
        logger.info("Setting up valid dataloaders.")
        return self.get_loaders_by_type("valid")

    def test_dataloaders(
        self,
    ) -> Dict[
        str,
        Iterator[jnp.ndarray] | Tuple[Iterator[jnp.ndarray]] | Iterator[jnp.ndarray],
    ]:
        logger.info("Setting up test dataloaders.")
        return self.get_loaders_by_type("test")
