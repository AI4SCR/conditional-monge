import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import jax.numpy as jnp
import pandas as pd
import optax
import yaml
from dotmap import DotMap
from flax import linen as nn


def get_source_target_transport(
    trainer,
    datamodule,
    conditions,
    target=True,
    source=True,
    transport=True,
):
    if datamodule.data_config.split[2] > 0:
        print("Evaluating on test set")
        split_type = "test"
        cond_to_loaders = datamodule.test_dataloaders()

    elif datamodule.data_config.split[1] > 0:
        print("Evaluating on validation set")
        split_type = "valid"

    else:
        print("Evaluating on training set")
        split_type = "train"
        cond_to_loaders = datamodule.train_dataloaders()

    all_expr = []
    for condition in conditions:
        print(condition)
        cond_embeddings = trainer.embedding_module(condition)

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
            trans = trainer.transport(source_expr, cond_embeddings)
            trans = datamodule.decoder(trans)
            trans = pd.DataFrame(trans)
            trans["dtype"] = "trans"
            trans["condition"] = condition
            all_expr.append(trans)

    return pd.concat(all_expr)


def jax_serializer(obj):
    if isinstance(obj, jnp.ndarray):
        return obj.tolist()
    raise TypeError("Type not serializable")


def create_or_update_logfile(file_path: Path, item: Dict[str, Any] = {}):
    """Updates the logfile containing the model experimetns."""
    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            json.dump({"experiments": []}, file)

    with open(file_path, "r") as file:
        data = json.load(file)

    data["experiments"].append(item)
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4, default=jax_serializer)


def load_config(path: Path) -> DotMap[str, Any]:
    """Load model and data configs from yaml file."""

    with open(path, "r") as file:
        yaml_data = yaml.safe_load(file)
    yaml_data = DotMap(yaml_data)
    return yaml_data


optim_factory = {"adamw": optax.adamw, "adam": optax.adam, "sgd": optax.sgd}
activation_factory = {"gelu": nn.gelu, "relu": nn.relu, "leakyrelu": nn.leaky_relu}
