import json
import os
import optax
import yaml

import jax.numpy as jnp

from dotmap import DotMap
from flax import linen as nn
from pathlib import Path
from typing import Any, Dict


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
