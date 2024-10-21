from pathlib import Path
from typing import Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# import scanpy as sc
from anndata import AnnData
from dotmap import DotMap
from jaxtyping import PRNGKeyArray
from loguru import logger

from cmonge.datasets.single_loader import SciPlexModule
from cmonge.utils import load_config


class SciPlexCPAModule(SciPlexModule):
    def __init__(self, config: DotMap):
        super().__init__(config)
        self.setup(**config)

    def setup(
        self,
        name: str,
        file_path: Path,
        batch_size: int,
        split: list[float],
        drug_col: str,
        celltype_col: str,
        drug_condition: str,
        control_condition: str,
        seed: str,
        ae: bool = False,
        ae_config_path: Optional[Path] = None,
        reduction: Optional[str] = None,
        parent: Optional[AnnData] = None,
        parent_reducer: Optional[str] = None,
        parent_celltype_to_idx: Optional[dict] = None,
        parent_drug_to_idx: Optional[dict] = None,
    ):
        self.name = name
        self.file_path = file_path
        self.batch_size = batch_size
        self.split = split
        self.drug_col = drug_col
        self.celltype_col = celltype_col
        self.drug_condition = drug_condition
        self.control_condition = control_condition
        self.ae = ae
        self.ae_config = load_config(ae_config_path) if ae_config_path else None
        self.reduction = reduction
        self.parent = parent
        self.parent_reducer = parent_reducer
        self.parent_celltype_to_idx = parent_celltype_to_idx
        self.parent_drug_to_idx = parent_drug_to_idx
        self.seed = seed
        self.key = jax.random.PRNGKey(self.seed)

        cond_split = self.drug_condition.split("-")
        self.drug = "-".join(cond_split[:-1])

        self.loader()
        self.preprocesser()
        self.splitter()
        self.reducer()

    def preprocesser(self) -> None:
        """Normalizes and log transofrms the data."""
        if not isinstance(self.adata.X, np.ndarray):
            self.adata.X = jnp.asarray(self.adata.X.todense())

        if self.parent_celltype_to_idx:
            self.celltype_to_idx = self.parent_celltype_to_idx
        else:
            cell_lines = sorted(self.adata.obs[self.celltype_col].astype(str).unique())
            self.data_config.celltype_to_idx = {c: i for i, c in enumerate(cell_lines)}

        if not self.parent_drug_to_idx:
            logger.warning(
                "Parent drug to idx not provided, drug index will always be 0"
            )
            self.drug_idx = 0
        else:
            # Getting drug based on condition in case condition also contains dose
            drug = [
                k for k in self.parent_drug_to_idx.keys() if k in self.drug_condition
            ][0]
            self.drug_idx = self.parent_drug_to_idx[drug]

        self.degs_bool = [g in self.marker_genes for g in self.adata.var_names]

    def get_model_inputs_from_adata(self, adata, cell_idx, condition):
        X = adata[cell_idx, :].X
        if self.parent_celltype_to_idx:
            cell_type = jnp.asarray(
                [
                    self.celltype_to_idx[c]
                    for c in adata.obs.loc[cell_idx, self.celltype_col]
                ],
                dtype="int32",
            )
        else:
            cell_type = jnp.asarray([0] * len(cell_idx), dtype="int32")

        drug_idx = jnp.asarray([self.drug_idx] * len(cell_idx), dtype="int32")

        degs = jnp.asarray(
            [self.degs_bool for i in range(len(cell_idx))], dtype="float32"
        )

        return X, cell_type, drug_idx, degs

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

        control, control_celltypes, control_drug_idx, degs = (
            self.get_model_inputs_from_adata(
                self.control_adata, control_cells, self.control_condition
            )
        )
        target, target_celltypes, target_drug_idx, degs = (
            self.get_model_inputs_from_adata(
                self.target_adata, target_cells, self.drug_condition
            )
        )

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
            loaders = (
                self.sampler_iter(
                    [control, control_celltypes, control_drug_idx, degs], batch_size, k1
                ),
                self.sampler_iter(
                    [target, target_celltypes, target_drug_idx, degs], batch_size, k2
                ),
            )
        return loaders

    @staticmethod
    def sampler_iter(
        arrays: [jnp.ndarray], batch_size: int, key: PRNGKeyArray
    ) -> Iterator[jnp.ndarray]:
        """Creates an inifinite dataloader with random sampling out of a jax array."""
        keys = [key]
        while True:
            keys = jax.random.split(keys[0], len(arrays))
            outs = [
                jax.random.choice(key=keys[i], a=arrays[i], shape=(batch_size,))
                for i in range(len(arrays))
            ]
            yield outs
