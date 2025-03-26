import abc
from itertools import cycle
from pathlib import Path
from typing import Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scanpy as sc
from anndata import AnnData
from dotmap import DotMap
from jaxtyping import PRNGKeyArray
from loguru import logger
from sklearn.model_selection import train_test_split

from cmonge.trainers.ae_trainer import AETrainerModule
from cmonge.utils import load_config


class AbstractDataModule:
    """Abstract class for handling data loading, processing and splitting."""

    def __init__(self) -> None:
        self.adata = None
        self.encoder = lambda x: x
        self.decoder = lambda x: x
        self.reduction = None
        self.marker_idx = None
        self.ae = False

    @abc.abstractmethod
    def setup(self, *args, **kwargs) -> None:
        """Abstract method for loading, setting up the DataGenerator."""

    @abc.abstractmethod
    def train_dataloaders(self) -> Tuple[Iterator[jnp.ndarray], Iterator[jnp.ndarray]]:
        """Abstract method for creating and returning an interator from the training data."""

    @abc.abstractmethod
    def valid_dataloaders(self) -> Tuple[Iterator[jnp.ndarray], Iterator[jnp.ndarray]]:
        """Abstract method for creating and returning an iterator from the validation data."""

    @abc.abstractmethod
    def test_dataloaders(self) -> Tuple[Iterator[jnp.ndarray], Iterator[jnp.ndarray]]:
        """Abstract method for creating and returning an iterator from the test data."""

    @staticmethod
    def sampler_iter(
        array: jnp.ndarray, batch_size: int, key: PRNGKeyArray
    ) -> Iterator[jnp.ndarray]:
        """Creates an inifinite dataloader with random sampling out of a jax array."""
        while True:
            k1, key = jax.random.split(key, 2)
            yield jax.random.choice(key=k1, a=array, shape=(batch_size,))

    @staticmethod
    def batcher(array: jnp.ndarray, batch_size: int) -> jnp.ndarray:
        """Groups the data into batches of size batch size."""
        data_size = array.shape[0]
        num_batches = data_size // batch_size
        trimmed = array[: num_batches * batch_size]
        batched = trimmed.reshape((num_batches, batch_size) + array.shape[1:])
        return batched

    def batcher_iter(
        self, array: jnp.ndarray, batch_size: int
    ) -> Iterator[jnp.ndarray]:
        """Groups the data into batches and returns an iterator obejct."""
        batched = self.batcher(array, batch_size)
        for batch in batched:
            yield batch

    def cyclic_iter(self, array: jnp.ndarray, batch_size: int) -> Iterator[jnp.ndarray]:
        """Groups the data into batches and creates a cyclic iterator object."""
        batched = self.batcher(array, batch_size)
        return cycle(batched)

    def splitter(self):
        """Splits the dataset into source-target and train/valid/test buckets."""
        logger.info("Splitting dataset started.")
        source_idx = self.adata.obs[
            self.adata.obs[self.drug_col] == self.control_condition
        ].index
        self.control_adata = self.adata[source_idx, :]
        target_idx = self.adata.obs[
            self.adata.obs[self.drug_col] == self.drug_condition
        ].index
        self.target_adata = self.adata[target_idx, :]
        assert sum(self.split) == 1
        key, self.key = jax.random.split(self.key, 2)
        random_states = jax.random.randint(key, (4,), 0, 1000).tolist()
        if len(self.split) == 2:
            self.split = self.split + [0]
        if len(self.split) == 3:
            (
                self.control_train_cells,
                self.control_valid_cells,
                self.control_test_cells,
            ) = get_train_valid_test_split(
                self.control_adata.obs.index, self.split, random_states[:2]
            )
            self.target_train_cells, self.target_valid_cells, self.target_test_cells = (
                get_train_valid_test_split(
                    self.target_adata.obs.index, self.split, random_states[2:]
                )
            )
        else:
            raise ValueError("Invalid split.")
        cs1, cs2, cs3 = (
            self.control_train_cells.shape,
            self.control_valid_cells.shape,
            self.control_test_cells.shape,
        )

        ts1, ts2, ts3 = (
            self.target_train_cells.shape,
            self.target_valid_cells.shape,
            self.target_test_cells.shape,
        )

        logger.info(
            f"Target dataset number of cells - train: {ts1}, valid: {ts2}, test: {ts3}."
        )
        logger.info(
            f"Control dataset number of cells - train: {cs1}, valid: {cs2}, test: {cs3}."
        )
        logger.info("Splitting finished.")

    def get_loaders_by_type(
        self, split_type: str, batch_size: Optional[int] = None
    ) -> Tuple[Iterator[jnp.ndarray], Iterator[jnp.ndarray]]:
        """Convert adata object into control and target iterators,
        subset based on the split type (train/valid/test)."""
        if split_type == "train":
            control_cells = self.control_train_cells
            target_cells = self.target_train_cells
        elif split_type == "valid":
            control_cells = self.control_valid_cells
            target_cells = self.target_valid_cells
        elif split_type == "test":
            control_cells = self.control_test_cells
            target_cells = self.target_test_cells
        else:
            raise ValueError("Invalid split, split_type should be train/valid/test.")

        control = self.control_adata[control_cells, :].X
        target = self.target_adata[target_cells, :].X

        if self.ae:
            # Loader for AutoEncoder model
            loaders = self.get_ae_iter(control, target)

        else:
            # Loader for OT models
            if self.reduction:
                control = self.encoder(control)
                target = self.encoder(target)

            k1, self.key = jax.random.split(self.key, 2)
            k2, self.key = jax.random.split(self.key, 2)

            if batch_size is None:
                batch_size = self.batch_size
            loaders = self.sampler_iter(control, batch_size, k1), self.sampler_iter(
                target, batch_size, k2
            )
        return loaders

    def get_ae_iter(
        self, control: jnp.ndarray, target: jnp.ndarray
    ) -> Iterator[jnp.ndarray]:
        x = np.vstack((control, target))
        k1, self.key = jax.random.split(self.key, 2)
        x = jax.random.permutation(k1, x)
        loaders = self.batcher_iter(x, self.batch_size)
        return loaders


def get_train_valid_test_split(x: jnp.ndarray, split: list[int], seeds=[0, 1]):
    assert len(split) == 3
    assert sum(split) == 1
    train, valid, test = split
    empty = jnp.empty(0)
    if train == 0:
        x_train = empty
        x_valid_test = x
    elif train == 1:
        x_train = x
        x_valid_test = empty
    else:
        x_train, x_valid_test = train_test_split(
            x, random_state=seeds[0], test_size=(1 - train), shuffle=True
        )

    if valid == 0:
        x_valid = empty
        x_test = x_valid_test
    elif train + valid == 1:
        x_valid = x_valid_test
        x_test = empty
    else:
        x_valid, x_test = train_test_split(
            x_valid_test,
            random_state=seeds[1],
            test_size=(test / (1 - train)),
            shuffle=True,
        )
    return x_train, x_valid, x_test


class SciPlexModule(AbstractDataModule):
    def __init__(self, config: DotMap):
        super().__init__()
        self.setup(**config)

    def setup(
        self,
        name: str,
        file_path: Path,
        batch_size: int,
        split: list[float],
        drug_col: str,
        drug_condition: str,
        control_condition: str,
        seed: str,
        ae: bool = False,
        ae_config_path: Optional[Path] = None,
        reduction: Optional[str] = None,
        parent: Optional[AnnData] = None,
        parent_reducer: Optional[str] = None,
    ):
        self.name = name
        self.file_path = file_path
        self.batch_size = batch_size
        self.split = split
        self.drug_col = drug_col
        self.drug_condition = drug_condition
        self.control_condition = control_condition
        self.ae = ae
        self.ae_config = load_config(ae_config_path) if ae_config_path else None
        self.reduction = reduction
        self.parent = parent
        self.parent_reducer = parent_reducer
        self.seed = seed
        self.drug = self.drug_condition.split("-")[0]
        self.key = jax.random.PRNGKey(self.seed)

        self.loader()
        self.preprocesser()
        self.splitter()
        self.reducer()

    def loader(self) -> None:
        if self.parent:
            self.adata = self.parent
        else:
            self.adata = sc.read_h5ad(self.file_path)
        # save marker genes for evaluation
        try:
            genes = self.adata.uns["rank_genes_groups"]["names"][self.drug]
            self.marker_genes = [
                gene for gene in genes if gene in self.adata.var.index
            ][:50]
            self.gene_idx_to_enum = {
                idx: enum for (enum, idx) in enumerate(self.adata.var.index)
            }
            self.marker_idx = [
                self.gene_idx_to_enum[gene] for gene in self.marker_genes
            ]
            logger.info(
                f"{len(self.marker_idx)} marker genes are saved for evaluation."
            )
        except ValueError:
            print("Make sure the h5ad file_path points to the sciplex dataset.")

    def preprocesser(self) -> None:
        """Normalizes and log transofrms the data."""
        # sc.pp.log1p(self.adata)
        # sc.pp.normalize_total(self.adata)
        if not isinstance(self.adata.X, np.ndarray):
            self.adata.X = jnp.asarray(self.adata.X.todense())

    def reducer(self):
        """Sets up dimensionality reduction, either with PCA, AE or identity."""
        if self.reduction == "pca":
            self.pca_means = self.adata.X.mean(axis=0)
            self.encoder = lambda x: (x - self.pca_means) @ self.adata.varm["PCs"]
            self.decoder = lambda x: x @ self.adata.varm["PCs"].T + self.pca_means
        elif self.reduction == "ae":
            if self.parent_reducer:
                trainer = self.parent_reducer
            else:
                trainer = AETrainerModule(self.ae_config)
                trainer.load_model(self.name, self.drug_condition)
            model = trainer.model.bind({"params": trainer.state.params})
            self.encoder = lambda x: model.encoder(x)
            self.decoder = lambda x: model.decoder(x)
        else:
            self.encoder = lambda x: x
            self.decoder = lambda x: x

    def train_dataloaders(self) -> Tuple[Iterator[jnp.ndarray], Iterator[jnp.ndarray]]:
        """Convert training dataset into infinite iterators."""
        train_loaders = self.get_loaders_by_type("train")
        return train_loaders

    def valid_dataloaders(self) -> Tuple[Iterator[jnp.ndarray], Iterator[jnp.ndarray]]:
        """Convert test dataset into infinite iterators."""
        valid_loaders = self.get_loaders_by_type("valid")
        return valid_loaders

    def test_dataloaders(self) -> Tuple[Iterator[jnp.ndarray], Iterator[jnp.ndarray]]:
        """Convert test dataset into a batch iterator."""
        test_loaders = self.get_loaders_by_type("test")
        return test_loaders


class FourIModule(AbstractDataModule):
    def __init__(self, config: DotMap) -> None:
        super().__init__()
        self.setup(**config)

    def loader(self) -> None:
        if self.parent:
            self.adata = self.parent
        else:
            self.adata = sc.read_h5ad(self.file_path)
        with open(self.features_path) as f:
            features = f.readlines()
        self.features = [feature.rstrip() for feature in features]

        with open(self.drugs_path) as f:
            drugs = f.readlines()
        self.drugs = [drug.rstrip() for drug in drugs]

    def preprocesser(self) -> None:
        self.adata = self.adata[:, self.features]
        self.adata.X = jnp.asarray(self.adata.X)

    def setup(
        self,
        name: str,
        file_path: Path,
        drugs_path: Path,
        features: Path,
        split: list[float],
        batch_size: int,
        drug_col: str,
        drug_condition: str,
        control_condition: str,
        ae: bool,
        seed: int,
        split_dose: bool = False,
        parent: Optional[AnnData] = None,
        reduction: Optional[str] = None,  # For compatability
        parent_reducer: Optional[str] = None,  # For compatability
    ) -> None:
        self.name = name
        self.file_path = file_path
        self.split = split
        self.batch_size = batch_size
        self.features_path = features
        self.drugs_path = drugs_path
        self.drug_col = drug_col
        self.drug_condition = drug_condition
        self.control_condition = control_condition
        self.ae = ae
        self.key = jax.random.PRNGKey(seed)
        self.parent = parent
        self.reduction = reduction
        self.parent_reducer = parent_reducer
        self.loader()
        self.preprocesser()
        self.splitter()

    def train_dataloaders(self) -> Tuple[Iterator[jnp.ndarray], Iterator[jnp.ndarray]]:
        train_loaders = self.get_loaders_by_type("train")
        return train_loaders

    def valid_dataloaders(self) -> Tuple[Iterator[jnp.ndarray], Iterator[jnp.ndarray]]:
        valid_loaders = self.get_loaders_by_type("valid")
        return valid_loaders

    def test_dataloaders(
        self, batch_size: Optional[int] = None
    ) -> Tuple[Iterator[jnp.ndarray], Iterator[jnp.ndarray]]:
        test_loaders = self.get_loaders_by_type("test", batch_size)
        return test_loaders

    # For compatability with the ConditionalDataloader
    def reducer(self):
        """Sets up dimensionality reduction, either with PCA, AE or identity."""
        if self.reduction == "pca":
            self.pca_means = self.adata.X.mean(axis=0)
            self.encoder = lambda x: (x - self.pca_means) @ self.adata.varm["PCs"]
            self.decoder = lambda x: x @ self.adata.varm["PCs"].T + self.pca_means
        elif self.reduction == "ae":
            if self.parent_reducer:
                trainer = self.parent_reducer
            else:
                trainer = AETrainerModule(self.ae_config)
                trainer.load_model(self.name, self.drug_condition)
            model = trainer.model.bind({"params": trainer.state.params})
            self.encoder = lambda x: model.encoder(x)
            self.decoder = lambda x: model.decoder(x)
        else:
            self.encoder = lambda x: x
            self.decoder = lambda x: x
