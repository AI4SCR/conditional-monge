import random

import pandas as pd

from cmonge.datasets.conditional_loader import ConditionalDataModule
from cmonge.datasets.single_loader import AbstractDataModule
from cmonge.trainers.conditional_monge_trainer import ConditionalMongeTrainer
from cmonge.trainers.ot_trainer import MongeMapTrainer


def monge_get_source_target_transport(
    trainer: MongeMapTrainer,
    datamodule: AbstractDataModule,
    n_samples=1,
    target=True,
    source=True,
    transport=True,
    batch_size: int = None,
):
    if batch_size is None:
        batch_size = datamodule.batch_size

    if datamodule.split[2] > 0:
        print("Evaluating on test set")
        split_type = "test"
    elif datamodule.split[1] > 0:
        print("Evaluating on validation set")
        split_type = "valid"
    else:
        print("Evaluating on training set")
        split_type = "train"
    all_expr = []
    all_meta = []
    for i in range(n_samples):
        if split_type == "valid":
            sel_target_cells = random.sample(
                datamodule.target_valid_cells.tolist(),
                batch_size,
            )
            sel_control_cells = random.sample(
                datamodule.control_valid_cells.tolist(),
                batch_size,
            )

        elif split_type == "test":
            sel_target_cells = random.sample(
                datamodule.target_test_cells.tolist(), batch_size
            )
            sel_control_cells = random.sample(
                datamodule.control_test_cells.tolist(),
                batch_size,
            )
        elif split_type == "train":
            sel_target_cells = random.sample(
                datamodule.target_train_cells.tolist(),
                batch_size,
            )
            sel_control_cells = random.sample(
                datamodule.control_train_cells.tolist(),
                batch_size,
            )

        cond_expr = datamodule.adata[sel_target_cells].X
        cond_meta = datamodule.adata.obs.loc[sel_target_cells, :]

        source_expr = datamodule.adata[sel_control_cells].X
        source_meta = datamodule.adata.obs.loc[sel_control_cells, :]

        if target:
            cond_meta["sample_n"] = i
            cond_meta["dtype"] = "target"
            all_expr.append(pd.DataFrame(cond_expr, columns=datamodule.adata.var_names))
            all_meta.append(cond_meta)

        if source:
            source_meta["dtype"] = "source"
            source_meta["sample_n"] = i

            all_meta.append(source_meta)
            all_expr.append(
                pd.DataFrame(source_expr, columns=datamodule.adata.var_names)
            )

        if transport:
            trans = trainer.transport(source_expr, num_contexts=2)
            trans = datamodule.decoder(trans)
            trans_meta = cond_meta.copy()
            trans_meta["dtype"] = "transport"
            trans_meta["sample_n"] = i

            all_expr.append(pd.DataFrame(trans, columns=datamodule.adata.var_names))
            all_meta.append(trans_meta)

    all_expr = pd.concat(all_expr).reset_index(drop=True)
    all_meta = pd.concat(all_meta).reset_index(drop=True)

    return all_expr, all_meta


def get_source_target_transport(
    trainer: ConditionalMongeTrainer,
    datamodule: ConditionalDataModule,
    conditions,
    target=True,
    source=True,
    transport=True,
):
    if datamodule.data_config.split[2] > 0:
        print("Evaluating on test set")
        split_type = "test"

    elif datamodule.data_config.split[1] > 0:
        print("Evaluating on validation set")
        split_type = "valid"

    else:
        print("Evaluating on training set")
        split_type = "train"

    all_expr = []
    for condition in conditions:
        print(condition)
        cond_embeddings, num_contexts = trainer.embedding_module(condition)

        if split_type == "valid":
            loader_source, loader_target = datamodule.valid_dataloaders()[condition]
        elif split_type == "test":
            loader_source, loader_target = datamodule.test_dataloaders()[condition]
        elif split_type == "train":
            loader_source, loader_target = datamodule.train_dataloaders()[condition]

        target_expr = next(iter(loader_target))
        source_expr = next(iter(loader_source))

        if target:
            res_target = datamodule.decoder(target_expr)
            res_target = pd.DataFrame(res_target)
            res_target["dtype"] = "target"
            res_target["condition"] = condition
            all_expr.append(res_target)

        if source:
            res_source = datamodule.decoder(source_expr)
            res_source = pd.DataFrame(res_source)
            res_source["dtype"] = "source"
            res_source["condition"] = condition
            all_expr.append(res_source)

        if transport:
            trans = trainer.transport(source_expr, cond_embeddings, num_contexts)
            trans = datamodule.decoder(trans)
            trans = pd.DataFrame(trans)
            trans["dtype"] = "trans"
            trans["condition"] = condition
            all_expr.append(trans)

    return pd.concat(all_expr)
