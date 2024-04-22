import json
import os
from pathlib import Path
from typing import Any, Dict

import jax.numpy as jnp
import optax
import yaml
from dotmap import DotMap
from flax import linen as nn
from jax._src.typing import Array


def get_environ_var(env_var_name, fail_gracefully=True):
    try:
        assert (
            env_var_name in os.environ
        ), f"Environment variable ${env_var_name} not set, are you on a CCC job?"
        var = os.environ[env_var_name]
    except AssertionError:
        if not fail_gracefully:
            raise
        else:
            var = None

    return var


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


def flatten(dictionary, parent_key="", separator="."):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, DotMap):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def flatten_metrics(dictionary, parent_key="", separator="."):
    items = []
    for key, value in dictionary.items():
        if isinstance(value, Array):
            value = float(value)
        elif isinstance(value, list):
            if isinstance(value[0], Array):
                value = [float(i) for i in Array]

        new_key = parent_key + separator + key if parent_key else key

        if isinstance(value, dict):
            items.extend(flatten_metrics(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


optim_factory = {"adamw": optax.adamw, "adam": optax.adam, "sgd": optax.sgd}
activation_factory = {"gelu": nn.gelu, "relu": nn.relu, "leakyrelu": nn.leaky_relu}
