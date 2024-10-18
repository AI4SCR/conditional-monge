from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scanpy as sc
from anndata import AnnData
from dotmap import DotMap
from jaxtyping import PRNGKeyArray
from loguru import logger

from cmonge.datasets.conditional_loader import ConditionalDataModule
from cmonge.datasets.single_loader import (
    AbstractDataModule,
    DataModuleFactory,
    SciPlexModule,
)
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
        self.drug = self.drug_condition.split("-")[0]
        self.key = jax.random.PRNGKey(self.seed)

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
        if not self.parent_drug_to_idx:
            drug_idx = jnp.asarray([0] * len(cell_idx), dtype="int32")
        else:
            # Getting drug based on condition in case condition also contains dose
            drug = [k for k in self.parent_drug_to_idx.keys() if k in condition][0]
            drug_idx = jnp.asarray(
                [self.parent_drug_to_idx[drug]] * len(cell_idx), dtype="int32"
            )

        return X, cell_type, drug_idx

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

        control, control_celltypes, control_drug_idx = self.get_model_inputs_from_adata(
            self.control_adata, control_cells, self.control_condition
        )
        target, target_celltypes, target_drug_idx = self.get_model_inputs_from_adata(
            self.target_adata, target_cells, self.drug_condition
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
                    [control, control_celltypes, control_drug_idx], batch_size, k1
                ),
                self.sampler_iter(
                    [target, target_celltypes, target_drug_idx], batch_size, k2
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


# from typing import Dict, Iterator, Optional, Tuple

# import jax
# import jax.numpy as jnp
# import scanpy as sc
# from dotmap import DotMap
# from loguru import logger

# from cmonge.datasets.single_loader import AbstractDataModule, DataModuleFactory
# from cmonge.trainers.ae_trainer import AETrainerModule
# from cmonge.datasets.conditional_loader import ConditionalDataModule

# DataModuleFactory = {"cpa_sciplex": SciPlexCPAModule}


class ConditionalCPADataModule(ConditionalDataModule):
    def __init__(
        self,
        data_config: DotMap,
        condition_config: DotMap,
        ood_condition_config: Optional[DotMap] = None,
        ae_config: Optional[DotMap] = None,
        split_dose: bool = True,
    ):
        self.data_config = data_config
        self.name = f"cond-{data_config.name}"
        assert condition_config.mode in ["extrapolate", "homogeneous", "custom"]
        self.condition_config = condition_config
        self.ood_condition_config = ood_condition_config
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

            if split_dose:
                control = [self.data_config.control_condition.split("-")[0]]
                drugs = list(
                    set([c.split("-")[0] for c in self.condition_config.conditions])
                )
                ood_drugs = (
                    list(
                        set(
                            [
                                c.split("-")[0]
                                for c in self.ood_condition_config.conditions
                            ]
                        )
                    )
                    if self.ood_condition_config
                    else []
                )
            else:
                # Assumes single drugs - wouldn't work with 4i combi therapies
                control = [self.data_config.control_condition]
                drugs = self.condition_config.conditions
                ood_drugs = (
                    self.ood_condition_config.conditions
                    if self.ood_condition_config
                    else []
                )
            drugs = sorted(drugs + ood_drugs + control)
            self.data_config.parent_drug_to_idx = {d: i for i, d in enumerate(drugs)}

            cell_lines = sorted(self.adata.obs["cell_type"].astype(str).unique())
            self.data_config.parent_celltype_to_idx = {
                c: i for i, c in enumerate(cell_lines)
            }

            self.data_config.parent = self.adata

        if self.data_config.reduction:
            self.reducer()
        self.splitter()
