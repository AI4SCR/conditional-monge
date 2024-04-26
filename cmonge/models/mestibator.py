# Based on ott.tools.map_estimator

import collections
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Iterable
from dotmap import DotMap
import mlflow

import jax
import jax.numpy as jnp
from orbax.checkpoint import CheckpointManager
from orbax.checkpoint.args import StandardSave, StandardRestore
import optax
from flax.training import train_state

from ott.solvers.nn import models
from ott.tools.map_estimator import MapEstimator
from cmonge.utils import flatten_metrics


class EarlyStopMapEstimator(MapEstimator):
    def __init__(
        self,
        dim_data: int,
        model: models.ModelBase,
        optimizer: Optional[optax.OptState] = None,
        fitting_loss: Optional[Callable[[jnp.ndarray, jnp.ndarray], float]] = None,
        regularizer: Optional[Callable[[jnp.ndarray, jnp.ndarray], float]] = None,
        regularizer_strength: float = 1.0,
        num_train_iters: int = 10_000,
        logging: bool = False,
        valid_freq: int = 500,
        rng: Optional[jax.random.PRNGKey] = None,
        checkpoint_manager: CheckpointManager = None,
        extra_cm_save_args: Optional[DotMap] = None,
        mlflow_logging: bool = False,
    ):
        super().__init__(
            dim_data=dim_data,
            model=model,
            optimizer=optimizer,
            fitting_loss=fitting_loss,
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            num_train_iters=num_train_iters,
            logging=logging,
            valid_freq=valid_freq,
            rng=rng,
        )

        self.checkpoint_manager = checkpoint_manager
        self.mlflow_logging = mlflow_logging

    def train_map_estimator(
        self,
        trainloader_source: Iterator[jnp.ndarray],
        trainloader_target: Iterator[jnp.ndarray],
        validloader_source: Iterator[jnp.ndarray],
        validloader_target: Iterator[jnp.ndarray],
    ) -> Tuple[train_state.TrainState, Dict[str, Any]]:
        """Training loop."""
        # define logs
        logs = collections.defaultdict(lambda: collections.defaultdict(list))

        # try to display training progress with tqdm
        try:
            from tqdm import trange

            tbar = trange(self.num_train_iters, leave=True)
        except ImportError:
            tbar = range(self.num_train_iters)

        for step in tbar:
            #  update step
            is_logging_step = True
            train_batch = self._generate_batch(
                loader_source=trainloader_source,
                loader_target=trainloader_target,
            )
            valid_batch = self._generate_batch(
                loader_source=validloader_source,
                loader_target=validloader_target,
            )

            self.state_neural_net, current_logs = self.step_fn(
                self.state_neural_net,
                train_batch,
                valid_batch,
                is_logging_step=is_logging_step,
            )
            if self.mlflow_logging:
                mlflow.log_metrics(flatten_metrics(current_logs))

            # store and print metrics if logging step
            for log_key in current_logs:
                for metric_key in current_logs[log_key]:
                    logs[log_key][metric_key].append(current_logs[log_key][metric_key])

            # update the tqdm bar if tqdm is available
            if not isinstance(tbar, range):
                reg_msg = (
                    "not computed"
                    if current_logs["eval"]["regularizer"] == 0.0
                    else f"{current_logs['eval']['regularizer']:.4f}"
                )
                postfix_str = (
                    f"fitting_loss: {current_logs['eval']['fitting_loss']:.4f}, "
                    f"regularizer: {reg_msg}."
                )
                tbar.set_postfix_str(postfix_str)

            ckpt = self.state_neural_net
            self.checkpoint_manager.save(
                step,
                args=StandardSave(ckpt),
                metrics={
                    "sinkhorn_div": float(current_logs["eval"]["fitting_loss"]),
                    "monge_gap": float(current_logs["eval"]["regularizer"]),
                    "total_loss": float(current_logs["eval"]["regularizer"]),
                },
            )
        self.checkpoint_manager.wait_until_finished()
        return self.state_neural_net, logs

    @classmethod
    def load_from_model_state(
        cls,
        model,
        num_genes: int,
        dim_hidden: Iterable,
        fitting_loss: Optional[Callable[[jnp.ndarray, jnp.ndarray], float]] = None,
        regularizer: Optional[Callable[[jnp.ndarray, jnp.ndarray], float]] = None,
        regularizer_strength: float = 1.0,
        num_train_iters: int = 10_000,
        logging: bool = False,
        valid_freq: int = 500,
        rng: Optional[jax.random.PRNGKey] = None,
        checkpoint_manager: CheckpointManager = None,
        mlflow_logging: bool = False,
        step: int = None,
        **extras,
    ):

        out_class = cls(
            model=model,
            dim_data=num_genes,
            fitting_loss=fitting_loss,
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            num_train_iters=num_train_iters,
            logging=logging,
            valid_freq=valid_freq,
            rng=rng,
            checkpoint_manager=checkpoint_manager,
            mlflow_logging=mlflow_logging,
        )

        if step is None:
            # Only checks steps with metrics available
            step = checkpoint_manager.best_step()
        out_class.checkpoint_manager.restore(step, args=StandardRestore())

        return out_class
