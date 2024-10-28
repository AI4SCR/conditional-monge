from typing import Any, Dict

import jax.numpy as jnp
from loguru import logger

from cmonge.metrics import (average_r2, compute_scalar_mmd, drug_signature,
                            eucledian_monge_gap, sinkhorn_div,
                            wasserstein_distance)


def init_logger_dict(metrics: Dict[str, Any], drug) -> Dict[str, Any]:
    metrics["point_clouds"] = {"source": [], "target": [], "transport": []}
    metrics["drug"] = drug
    metrics["wasserstein"] = []
    metrics["mmd"] = []
    metrics["sinkhorn_div"] = []
    metrics["monge_gap"] = []
    metrics["drug_signature"] = []
    metrics["r2"] = []
    metrics["mean_statistics"] = {}
    return metrics


def log_metrics(metrics, target, transport):
    w_dist = wasserstein_distance(target, transport)
    mmd_dist = compute_scalar_mmd(target, transport)
    sh_div = sinkhorn_div(target, transport)
    monge_gap = eucledian_monge_gap(target, transport)
    ds = drug_signature(target, transport)
    r2 = average_r2(target, transport)

    metrics["wasserstein"].append(w_dist)
    metrics["mmd"].append(mmd_dist)
    metrics["sinkhorn_div"].append(sh_div)
    metrics["monge_gap"].append(monge_gap)
    metrics["drug_signature"].append(ds)
    metrics["r2"].append(r2)


def log_point_clouds(metrics, source, target, transport):
    metrics["point_clouds"]["source"].append(source)
    metrics["point_clouds"]["target"].append(target)
    metrics["point_clouds"]["transport"].append(transport)


def log_mean_metrics(metrics):
    metrics["mean_statistics"]["mean_wasserstein"] = float(
        sum(metrics["wasserstein"]) / len(metrics["wasserstein"])
    )
    metrics["mean_statistics"]["mean_mmd"] = float(
        sum(metrics["mmd"]) / len(metrics["mmd"])
    )
    metrics["mean_statistics"]["mean_sinkhorn div"] = float(
        sum(metrics["sinkhorn_div"]) / len(metrics["sinkhorn_div"])
    )
    metrics["mean_statistics"]["mean_monge_gap"] = float(
        sum(metrics["monge_gap"]) / len(metrics["monge_gap"])
    )
    metrics["mean_statistics"]["mean_drug_signature"] = float(
        sum(metrics["drug_signature"]) / len(metrics["drug_signature"])
    )
    metrics["mean_statistics"]["mean_r2"] = float(
        sum(metrics["r2"]) / len(metrics["r2"])
    )
    logger.info(metrics["mean_statistics"])


def get_single_loaders_for_eval(datamodule, split):
    if split == "valid":
        loader_source, loader_target = datamodule.valid_dataloaders()
    else:
        loader_source, loader_target = datamodule.test_dataloaders()
    return loader_source, loader_target


def get_conditional_loaders_for_eval(datamodule):
    condition = datamodule.sample_condition("valid")
    valid_condition_to_loaders = datamodule.valid_dataloaders()
    loader_source, loader_target = valid_condition_to_loaders[condition]
    condition = datamodule.embeddings[condition]
    cond = jnp.asarray([[condition] for _ in range(datamodule.batch_size)])
    return loader_source, loader_target, condition, cond
