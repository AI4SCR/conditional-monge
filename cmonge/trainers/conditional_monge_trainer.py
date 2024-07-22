import collections
import functools
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from cmonge.datasets.conditional_loader import ConditionalDataModule
from cmonge.evaluate import init_logger_dict, log_mean_metrics, log_metrics
from cmonge.models.embedding import embed_factory
from cmonge.models.nn import ConditionalPerturbationNetwork
from cmonge.trainers.ot_trainer import (
    AbstractTrainer,
    loss_factory,
    regularizer_factory,
)
from cmonge.utils import create_or_update_logfile, optim_factory
from dotmap import DotMap
from flax.core import frozen_dict
from flax.training import train_state
from loguru import logger

from orbax.checkpoint import PyTreeCheckpointer
from flax.training.orbax_utils import save_args_from_target


class ConditionalMongeTrainer(AbstractTrainer):
    def __init__(self, jobid: int, logger_path: Path, config: DotMap, datamodule: ConditionalDataModule) -> None:
        super().__init__(jobid, logger_path)
        self.config = config
        self.datamodule = datamodule

        self.key = jax.random.PRNGKey(self.config.seed)
        self.regularizer_strength = 1
        self.num_train_iters = self.config.num_train_iters

        self.init_model(datamodule=datamodule)

    def init_model(self, datamodule: ConditionalDataModule):
        # setup loss function and regularizer
        fitting_loss_fn = loss_factory[self.config.fitting_loss.name]
        regularizer_fn = regularizer_factory[self.config.regularizer.name]
        self.fitting_loss = partial(fitting_loss_fn, **self.config.fitting_loss.kwargs)
        self.regularizer = partial(regularizer_fn, **self.config.regularizer.kwargs)

        # setup optimizer and scheduler
        opt_fn = optim_factory[self.config.optim.name]
        lr_scheduler = optax.cosine_decay_schedule(
            init_value=self.config.optim.lr, decay_steps=self.num_train_iters, alpha=1e-2
        )
        optimizer = opt_fn(learning_rate=lr_scheduler, **self.config.optim.kwargs)

        self.neural_net = ConditionalPerturbationNetwork(**self.config.mlp)  # TODO: create embedding and model factory

        embed_module = embed_factory[self.config.embedding.name]
        self.embedding_module = embed_module(datamodule=datamodule ,**self.config.embedding)

        self.step_fn = self._get_step_fn()
        self.key, rng = jax.random.split(self.key, 2)
        self.state_neural_net = self.neural_net.create_train_state(rng, optimizer)

    def generate_batch(self, datamodule: ConditionalDataModule, split_type: str) -> Dict[str, jnp.ndarray]:
        """Generate a batch of condition and samples."""
        condition_to_loaders = datamodule.get_loaders_by_type(split_type)
        condition = datamodule.sample_condition(split_type)
        loader_source, loader_target = condition_to_loaders[condition]
        embeddings = self.embedding_module(condition)
        return {"source": next(loader_source), "target": next(loader_target), "condition": embeddings}

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

        for step in tbar:
            is_logging_step = step % 100 == 0
            train_batch = self.generate_batch(datamodule, "train")
            valid_batch = None if not is_logging_step else self.generate_batch(datamodule, "valid")

            self.state_neural_net, current_logs = self.step_fn(
                self.state_neural_net, train_batch, valid_batch, is_logging_step
            )

            if is_logging_step:
                self.update_logs(current_logs, logs, tbar)
        self.metrics["ott-logs"] = logs

        return self.state_neural_net, logs

    def _get_step_fn(self) -> Callable:
        """Create a one step training and evaluation function."""

        def loss_fn(
            params: frozen_dict.FrozenDict,
            apply_fn: Callable,
            batch: Dict[str, jnp.ndarray],
        ) -> Tuple[float, Dict[str, float]]:
            """Loss function."""
            # map samples with the fitted map
            mapped_samples = apply_fn({"params": params}, batch["source"], batch["condition"])

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

        @functools.partial(jax.jit, static_argnums=3)
        def step_fn(
            state_neural_net: train_state.TrainState,
            train_batch: Dict[str, jnp.ndarray],
            valid_batch: Optional[Dict[str, jnp.ndarray]] = None,
            is_logging_step: bool = False,
        ) -> Tuple[train_state.TrainState, Dict[str, float]]:
            """Step function."""
            # compute loss and gradients
            grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
            (_, current_train_logs), grads = grad_fn(state_neural_net.params, state_neural_net.apply_fn, train_batch)

            # logging step
            current_logs = {"train": current_train_logs, "eval": {}}
            if is_logging_step:
                _, current_eval_logs = loss_fn(
                    params=state_neural_net.params, apply_fn=state_neural_net.apply_fn, batch=valid_batch
                )
                current_logs["eval"] = current_eval_logs

            # update state
            return state_neural_net.apply_gradients(grads=grads), current_logs

        return step_fn

    def transport(self, x, c):
        return self.state_neural_net.apply_fn({"params": self.state_neural_net.params}, x, c)

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
        ):
            for enum, (source, target) in enumerate(zip(loader_source, loader_target)):
                if not identity:
                    transport = self.transport(source, cond_embeddings)
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
            cond_to_loaders: Dict[str, Tuple[Iterator[jnp.ndarray], Iterator[jnp.ndarray]]], split_type: str
        ):
            self.metrics[split_type] = {}
            for cond, loader in cond_to_loaders.items():
                logger.info(f"Evaluation started on {cond} {split_type}.")
                cond_embedding = self.embedding_module(cond)
                loader_source, loader_target = loader

                self.metrics[split_type][cond] = {}
                init_logger_dict(self.metrics[split_type][cond], datamodule.drug_condition)
                evaluate_condition(loader_source, loader_target, cond_embedding, self.metrics[split_type][cond])
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

    def save_checkpoint(
        self,path: Path
    ) -> None:
        ckpt = self.state_neural_net
        checkpointer=PyTreeCheckpointer()
        save_args = save_args_from_target(ckpt)
        checkpointer.save(
                path,
                ckpt, 
                save_args=save_args,
                force=True
            )


    @classmethod
    def load_checkpoint(
        cls,
        jobid: int,
        logger_path: Path,
        config: DotMap,
        datamodule: ConditionalDataModule,
        ckpt_path: Path,
    ) -> None:
        out_class = cls(
            jobid=jobid,
            logger_path=logger_path,
            config=config,
            datamodule=datamodule,
        )
        checkpointer=PyTreeCheckpointer()
        out_class.state_neural_net = checkpointer.restore(
            ckpt_path, item=out_class.state_neural_net
        )
        logger.info("Loaded ConditionalMongeTrainer from checkpoint")
        return out_class