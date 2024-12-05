from typing import Dict, Optional

import jax
import scanpy as sc
from dotmap import DotMap
from loguru import logger

from cmonge.datasets.conditional_loader import ConditionalDataModule
from cmonge.datasets.single_loader import AbstractDataModule
from cmonge.datasets import DataModuleFactory


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
            drugs = sorted(drugs + ood_drugs) + control
            self.data_config.parent_drug_to_idx = {d: i for i, d in enumerate(drugs)}

            logger.info(f"drug to idx mapping {self.data_config.parent_drug_to_idx}")

            cell_lines = sorted(self.adata.obs["cell_type"].astype(str).unique())
            self.data_config.parent_celltype_to_idx = {
                c: i for i, c in enumerate(cell_lines)
            }

            self.data_config.parent = self.adata

        if self.data_config.reduction:
            self.reducer()
        self.splitter()
