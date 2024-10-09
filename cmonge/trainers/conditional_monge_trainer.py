import collections
import functools
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from dotmap import DotMap
from flax.core import frozen_dict
from flax.training import train_state
from flax.training.orbax_utils import save_args_from_target
from jax.tree_util import tree_map
from loguru import logger
from orbax.checkpoint import PyTreeCheckpointer

from cmonge.datasets.conditional_loader import ConditionalDataModule
from cmonge.evaluate import init_logger_dict, log_mean_metrics, log_metrics
from cmonge.models.embedding import BaseEmbedding, EmbeddingFactory
from cmonge.models.nn import ConditionalPerturbationNetwork
from cmonge.trainers.ot_trainer import (
    AbstractTrainer,
    loss_factory,
    regularizer_factory,
)
from cmonge.utils import create_or_update_logfile, optim_factory


class ConditionalMongeTrainer(AbstractTrainer):
    embedding_factory: Dict[str, BaseEmbedding] = EmbeddingFactory

    def __init__(
        self,
        jobid: int,
        logger_path: Path,
        config: DotMap,
        datamodule: ConditionalDataModule,
    ) -> None:
        super().__init__(jobid, logger_path)
        self.config = config
        self.datamodule = datamodule

        self.key = jax.random.PRNGKey(self.config.seed)
        self.regularizer_strength = 1
        self.num_train_iters = self.config.num_train_iters
        self.grad_acc_steps = self.config.optim.get("grad_acc_steps", 1)
        self.setup(datamodule=datamodule)

    def setup(self, datamodule: ConditionalDataModule):
        """
        Setup function has to be overriden since it is abstract in base.

        Args:
            datamodule (ConditionalDataModule): The datamodule to be trained on.
        """
        # setup loss function and regularizer
        fitting_loss_fn = loss_factory[self.config.fitting_loss.name]
        regularizer_fn = regularizer_factory[self.config.regularizer.name]
        self.fitting_loss = partial(fitting_loss_fn, **self.config.fitting_loss.kwargs)
        self.regularizer = partial(regularizer_fn, **self.config.regularizer.kwargs)

        # setup optimizer and scheduler
        opt_fn = optim_factory[self.config.optim.name]
        lr_scheduler = self.config.get("lr_scheduler", DotMap({"name": "cosine"}))
        if lr_scheduler.name.lower() == "cosine":
            lr_scheduler = optax.cosine_decay_schedule(
                init_value=self.config.optim.lr,
                decay_steps=self.num_train_iters,
                alpha=1e-2,
            )
        elif lr_scheduler.name.lower() == "linear":
            lr_scheduler = optax.linear_onecycle_schedule(
                transition_steps=self.num_train_iters,
                peak_value=self.config.optim.lr,
                **lr_scheduler.kwargs,
            )
        optimizer = opt_fn(learning_rate=lr_scheduler, **self.config.optim.kwargs)

        self.neural_net = ConditionalPerturbationNetwork(
            **self.config.mlp
        )  # TODO: create embedding and model factory

        embed_module = self.embedding_factory[self.config.embedding.name]
        self.embedding_module = embed_module(
            datamodule=datamodule, **self.config.embedding
        )

        self.step_fn = self._get_step_fn()
        self.key, rng = jax.random.split(self.key, 2)
        self.state_neural_net = self.neural_net.create_train_state(rng, optimizer)

    def generate_batch(
        self,
        datamodule: ConditionalDataModule,
        split_type: str,
    ) -> Dict[str, jnp.ndarray]:
        """Generate a batch of condition and samples."""
        condition_to_loaders = datamodule.get_loaders_by_type(split_type)
        condition = datamodule.sample_condition(split_type)
        loader_source, loader_target = condition_to_loaders[condition]
        embeddings, n_contexts = self.embedding_module(condition=condition)
        return (
            {
                "source": next(loader_source),
                "target": next(loader_target),
                "condition": embeddings,
                "num_contexts": n_contexts,
            },
            condition,
        )

    def update_logs(self, current_logs, logs, tbar):
        # store and print metrics if logging step

        for log_key in current_logs:
            for metric_key in current_logs[log_key]:
                logs[log_key][metric_key].append(current_logs[log_key][metric_key])

        # update the tqdm bar if tqdm is available
        if not isinstance(tbar, range):
            reg_value = current_logs["eval"].get("regularizer", "NA")
            postfix_str = (
                f"fitting_loss: {current_logs['eval']['fitting_loss']:.4f}, "
                f"regularizer: {reg_value:.4f},"
                f"total: {current_logs['eval']['total_loss']:.4f}"
            )
            tbar.set_postfix_str(postfix_str)

    def train(self, datamodule: ConditionalDataModule):
        logs = collections.defaultdict(lambda: collections.defaultdict(list))
        try:
            from tqdm import trange

            tbar = trange(self.num_train_iters, leave=True)
        except ImportError:
            tbar = range(self.num_train_iters)

        train_conditions = []
        grads = tree_map(jnp.zeros_like, self.state_neural_net.params)
        for step in tbar:
            is_logging_step = step % 100 == 0
            is_gradient_acc_step = (step + 1) % self.grad_acc_steps == 0
            train_batch, condition = self.generate_batch(datamodule, "train")

            valid_batch, _ = (
                ({"num_contexts": None}, None)
                if not is_logging_step
                else self.generate_batch(datamodule, "valid")
            )

            self.state_neural_net, grads, current_logs = self.step_fn(
                self.state_neural_net,
                grads=grads,
                train_batch=train_batch,
                valid_batch=valid_batch,
                is_logging_step=is_logging_step,
                is_gradient_acc_step=is_gradient_acc_step,
                n_train_contexts=train_batch["num_contexts"],
                n_valid_contexts=valid_batch["num_contexts"],
            )
            train_conditions.append(condition)

            if is_logging_step:
                self.update_logs(current_logs, logs, tbar)
        self.metrics["ott-logs"] = logs
        self.metrics["train_conditions"] = train_conditions

        return self.state_neural_net, logs

    def _get_step_fn(self) -> Callable:
        """Create a one step training and evaluation function."""

        def loss_fn(
            params: frozen_dict.FrozenDict,
            apply_fn: Callable,
            batch: Dict[str, jnp.ndarray],
            n_contexts: int,
        ) -> Tuple[float, Dict[str, float]]:
            """Loss function."""
            # map samples with the fitted map
            mapped_samples = apply_fn(
                {"params": params},
                batch["source"],
                batch["condition"],
                n_contexts,
            )

            # compute the loss
            val_fitting_loss = self.fitting_loss(batch["target"], mapped_samples)
            val_regularizer = self.regularizer(batch["source"], mapped_samples)
            val_tot_loss = val_fitting_loss + val_regularizer

            # store training logs
            loss_logs = {
                "total_loss": val_tot_loss,
                "fitting_loss": val_fitting_loss,
                "regularizer": val_regularizer,
            }

            return val_tot_loss, loss_logs

        @functools.partial(jax.jit, static_argnums=[4, 5, 6, 7])
        def step_fn(
            state_neural_net: train_state.TrainState,
            grads: frozen_dict.FrozenDict,
            train_batch: Dict[str, jnp.ndarray],
            valid_batch: Optional[Dict[str, jnp.ndarray]] = None,
            is_logging_step: bool = False,
            is_gradient_acc_step: bool = False,
            n_train_contexts: int = 2,
            n_valid_contexts: int = 2,
        ) -> Tuple[train_state.TrainState, frozen_dict.FrozenDict, Dict[str, float]]:
            """Step function."""
            # compute loss and gradients
            grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
            (_, current_train_logs), step_grads = grad_fn(
                state_neural_net.params,
                state_neural_net.apply_fn,
                train_batch,
                n_train_contexts,
            )
            # Accumulate gradients
            grads = tree_map(lambda g, step_g: g + step_g, grads, step_grads)

            # logging step
            current_logs = {"train": current_train_logs, "eval": {}}
            if is_logging_step:
                _, current_eval_logs = loss_fn(
                    params=state_neural_net.params,
                    apply_fn=state_neural_net.apply_fn,
                    batch=valid_batch,
                    n_contexts=n_valid_contexts,
                )
                current_logs["eval"] = current_eval_logs

            # update state
            if is_gradient_acc_step:
                state_neural_net = state_neural_net.apply_gradients(
                    grads=tree_map(lambda g: g / self.grad_acc_steps, grads)
                )
                # Reset gradients
                grads = tree_map(jnp.zeros_like, grads)

            return state_neural_net, grads, current_logs

        return step_fn

    def transport(self, x, c, num_contexts):
        return self.state_neural_net.apply_fn(
            {"params": self.state_neural_net.params}, x, c, num_contexts
        )

    def evaluate(
        self,
        datamodule: ConditionalDataModule,
        identity: bool = False,
        n_samples: int = 9,
    ) -> None:
        """Evaluate a trained model on a validation set and save the metrics to a json file."""

        def evaluate_condition(
            loader_source: Iterator[jnp.ndarray],
            loader_target: Iterator[jnp.ndarray],
            cond_embeddings: jnp.ndarray,
            metrics: str,
            n_contexts,
        ):
            for enum, (source, target) in enumerate(zip(loader_source, loader_target)):
                if not identity:
                    transport = self.transport(source, cond_embeddings, n_contexts)
                else:
                    transport = source

                transport = datamodule.decoder(transport)
                target = datamodule.decoder(target)

                if datamodule.marker_idx:
                    target = target[:, datamodule.marker_idx]
                    transport = transport[:, datamodule.marker_idx]

                log_metrics(metrics, target, transport)
                if enum > n_samples:
                    break

        def evaluate_split(
            cond_to_loaders: Dict[
                str, Tuple[Iterator[jnp.ndarray], Iterator[jnp.ndarray]]
            ],
            split_type: str,
        ):
            self.metrics[split_type] = {}
            for cond, loader in cond_to_loaders.items():
                logger.info(f"Evaluation started on {cond} {split_type}.")
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
                cond_embedding, n_contexts = self.embedding_module(cond)
=======
                cond_embedding = self.embedding_module(cond, self.datamodule.data_config.split_dose)
>>>>>>> c24d78e (Adapt cmonge for conditional 4i experiments:)
=======
                cond_embedding = self.embedding_module(
                    cond, self.datamodule.data_config.split_dose
=======
                cond_embedding, n_contexts = self.embedding_module(
                    cond, self.split_dose
>>>>>>> 5c1b8a0 (Perturbation MLP two settings, sciplex and equal context embedding. Equal context embedding taken num_contexts per batch)
                )
>>>>>>> 2d35a30 (chore: blackening)
                loader_source, loader_target = loader

                self.metrics[split_type][cond] = {}
                init_logger_dict(
                    self.metrics[split_type][cond], datamodule.drug_condition
                )
                evaluate_condition(
                    loader_source,
                    loader_target,
                    cond_embedding,
                    self.metrics[split_type][cond],
                    n_contexts,
                )
                log_mean_metrics(self.metrics[split_type][cond])

        # Log in test set if present, otherwise valid otherwise train
        if self.datamodule.data_config.split[2] > 0:
            logger.info("Evaluating on test set")
            cond_to_loaders = datamodule.test_dataloaders()
            evaluate_split(
                cond_to_loaders=cond_to_loaders,
                split_type="test-set",
            )
        elif self.datamodule.data_config.split[1] > 0:
            logger.info("Evaluating on validation set")
            cond_to_loaders = datamodule.valid_dataloaders()
            evaluate_split(
                cond_to_loaders=cond_to_loaders,
                split_type="valid-set",
            )
        else:
            logger.info("Evaluating on train set")
            cond_to_loaders = datamodule.train_dataloaders()
            evaluate_split(
                cond_to_loaders=cond_to_loaders,
                split_type="train-set",
            )

        create_or_update_logfile(self.logger_path, self.metrics)

    @property
    def model(self) -> nn.Module:
        return self.state_neural_net

    @model.setter
    def model(self, value: nn.Module):
        """Setter for the model to be checkpointed."""
        self.state_neural_net = value
