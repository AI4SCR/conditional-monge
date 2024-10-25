import abc
from functools import partial
from pathlib import Path
from typing import Any, Dict, Union

import jax.numpy as jnp
import optax
from dotmap import DotMap
from flax import linen as nn
from flax.training.orbax_utils import save_args_from_target
from jax.lib import xla_bridge
from loguru import logger
from orbax.checkpoint import PyTreeCheckpointer
from ott.solvers.nn import models, neuraldual
from ott.tools import map_estimator

from cmonge.datasets.single_loader import AbstractDataModule
from cmonge.evaluate import (
    init_logger_dict,
    log_mean_metrics,
    log_metrics,
    log_point_clouds,
)
from cmonge.metrics import fitting_loss, regularizer
from cmonge.utils import create_or_update_logfile, optim_factory


class AbstractTrainer:
    """Abstract class for neural OT traininig."""

    def __init__(self, jobid: int, logger_path: Path) -> None:
        self.jobid = jobid
        self.logger_path = logger_path
        self.metrics: Dict[str, Any] = {}
        self.metrics["jobid"] = jobid
        device = xla_bridge.get_backend().platform
        logger.info(f"JAX uses {device} for trianing.")

    @abc.abstractmethod
    def setup(
        self,
        *args,
        **kwargs,
    ):
        """Abstract method for setting up the model."""
        pass

    @abc.abstractmethod
    def train(self, datamodule: AbstractDataModule) -> None:
        """Abstract method for training loop."""
        pass

    @abc.abstractmethod
    def transport(self, source: jnp.ndarray) -> jnp.ndarray:
        """Abstract method for transporting a batch of data."""
        pass

    @abc.abstractmethod
    def save_checkpoint(self, path: Path) -> None:
        """Abstract method for saving model parameters to a pickle file."""
        pass

    @abc.abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """Abstract method for loading model parameters from pickle file."""
        pass

    def evaluate(
        self,
        datamodule: AbstractDataModule,
        identity: bool = False,
        valid: bool = False,
        n_samples: int = 9,
        log_transport: bool = False,
    ) -> None:
        """Evaluate a trained model on a validation set and save the metrics to a json file."""
        logger.info(f"Evaluation started on {datamodule.drug_condition}.")
        init_logger_dict(self.metrics, datamodule.drug_condition)
        if valid:
            loader_source, loader_target = datamodule.valid_dataloaders()
            self.metrics["eval"] = "valid"
        else:
            loader_source, loader_target = datamodule.test_dataloaders()
            self.metrics["eval"] = "test"
        for enum, (source, target) in enumerate(zip(loader_source, loader_target)):
            if not identity:
                transport = self.transport(source)

            else:
                transport = source

            transport = datamodule.decoder(transport)
            target = datamodule.decoder(target)
            if log_transport:
                log_point_clouds(self.metrics, source, target, transport)
            if datamodule.marker_idx:
                target = target[:, datamodule.marker_idx]
                transport = transport[:, datamodule.marker_idx]

            log_metrics(self.metrics, target, transport)
            if enum > n_samples:
                break

        log_mean_metrics(self.metrics)
        create_or_update_logfile(self.logger_path, self.metrics)


class MongeMapTrainer(AbstractTrainer):
    """Wrapper class for Monge Gap training."""

    def __init__(self, jobid: int, logger_path: Path, config: DotMap) -> None:
        super().__init__(jobid, logger_path)
        self.setup(**config)

    def setup(
        self,
        method: str,
        dim_hidden: list[int],
        num_genes: int,
        num_train_iters: int,
        fitting_loss: Dict[str, Any],
        regularizer: Dict[str, Any],
        optim: Dict[str, Any],
    ) -> None:
        """Initializes models and optimizers."""
        self.metrics["params"] = {
            "num_genes": num_genes,
            "num_train_iters": num_train_iters,
            "fitting_loss": fitting_loss.name,
            "regularizer": regularizer.name,
            "dim_hidden": dim_hidden,
            "lr": optim.lr,
            "method": method,
        }

        # setup loss function and regularizer
        fitting_loss_fn = loss_factory[fitting_loss.name]
        regularizer_fn = regularizer_factory[regularizer.name]
        fitting_loss = partial(fitting_loss_fn, **fitting_loss.kwargs)
        regularizer = partial(regularizer_fn, **regularizer.kwargs)

        # setup neural network model
        model = models.MLP(dim_hidden=dim_hidden, is_potential=False, act_fn=nn.gelu)

        # setup optimizer and scheduler
        opt_fn = optim_factory[optim.name]
        lr_scheduler = optax.cosine_decay_schedule(
            init_value=optim.lr, decay_steps=num_train_iters, alpha=1e-2
        )
        optimizer = opt_fn(learning_rate=lr_scheduler, **optim.kwargs)

        # setup ott-jax solver
        self.solver = map_estimator.MapEstimator(
            num_genes,
            fitting_loss=fitting_loss,
            regularizer=regularizer,
            model=model,
            optimizer=optimizer,
            regularizer_strength=1,
            num_train_iters=num_train_iters,
            logging=True,
            valid_freq=100,
        )

    def train(self, datamodule: AbstractDataModule) -> None:
        """Trains a Monge Map estimator."""
        logger.info("Training started")
        train_loader_source, train_loader_target = datamodule.train_dataloaders()
        valid_loader_source, valid_loader_target = datamodule.valid_dataloaders()
        state, logs = self.solver.train_map_estimator(
            trainloader_source=train_loader_source,
            trainloader_target=train_loader_target,
            validloader_source=valid_loader_source,
            validloader_target=valid_loader_target,
        )
        self.state = state
        self.metrics["ottlogs"] = logs
        logger.info("Training finished.")

    def transport(self, source: jnp.ndarray) -> jnp.ndarray:
        """Transports a batch of data using the learned model."""
        return self.state.apply_fn({"params": self.state.params}, source)

    def save_checkpoint(self, path: Path = None, config: DotMap = None) -> None:
        if path is None and config is None:
            logger.error(
                """Please provide a checkpoint save path
            either directly or through the config, checkpoint was NOT saved."""
            )
        elif path is None:
            path = config.checkpointing_path

        ckpt = self.solver.state_neural_net
        checkpointer = PyTreeCheckpointer()
        save_args = save_args_from_target(ckpt)
        checkpointer.save(path, ckpt, save_args=save_args, force=True)

    @classmethod
    def load_checkpoint(
        cls,
        jobid: int,
        logger_path: Path,
        config: DotMap,
        ckpt_path: Path = None,
    ) -> Union[AbstractTrainer, None]:

        try:
            out_class = cls(
                jobid=jobid,
                logger_path=logger_path,
                config=config,
            )

            if ckpt_path is None:
                if len(config.checkpointing_path) > 0:
                    ckpt_path = config.checkpointing_path
                else:
                    logger.error(
                        "Provide checkpointing path either directly or through the model config"
                    )
            checkpointer = PyTreeCheckpointer()
            out_class.solver.state_neural_net = checkpointer.restore(
                ckpt_path, item=out_class.solver.state_neural_net
            )
            out_class.state = out_class.solver.state_neural_net
            logger.info("Loaded MongeMapTrainer from checkpoint")
            return out_class
        except Exception:
            logger.error(
                "Failed to load checkpoint, are you sure checkpoint was saved and correct path is provided?"
            )
            return


class NeuralDualTrainer(AbstractTrainer):
    """Wrapper class for neural dual training."""

    def __init__(self, jobid: int, logger_path: Path, config: DotMap) -> None:
        super().__init__(jobid, logger_path)
        self.setup(**config)

    def setup(
        self,
        dim_hidden: list[int],
        lr: float,
        num_genes: int,
        num_train_iters: int,
        num_inner_iters: int,
        samples_source: jnp.ndarray,
        samples_target: jnp.ndarray,
        **kwargs,
    ) -> None:
        """Initializes models and optimizers."""
        self.metrics["params"] = {
            "num_genes": num_genes,
            "num_training_iters": num_train_iters,
            "num_inner_iters": num_inner_iters,
            "dim_hidden": dim_hidden,
            "lr": lr,
            "method": "dual",
        }
        neural_f = models.ICNN(
            dim_data=num_genes,
            dim_hidden=dim_hidden,
            gaussian_map_samples=(samples_source, samples_target),
        )
        neural_g = models.MLP(dim_hidden=dim_hidden)

        lr_schedule = optax.cosine_decay_schedule(
            init_value=lr, decay_steps=num_train_iters, alpha=1e-2
        )
        optimizer_f = optax.adamw(learning_rate=lr_schedule, b1=0.5, b2=0.5)
        optimizer_g = optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.999)

        self.neural_dual_solver = neuraldual.W2NeuralDual(
            num_genes,
            neural_f,
            neural_g,
            optimizer_f,
            optimizer_g,
            num_train_iters=num_train_iters,
            logging=True,
            num_inner_iters=num_inner_iters,
            valid_freq=100,
            log_freq=100,
            amortization_loss="objective",
        )

    def train(self, datamodule: AbstractDataModule) -> None:
        """Trains Dual Estimator."""
        logger.info("Training started")
        train_loader_source, train_loader_target = datamodule.train_dataloaders()
        valid_loader_source, valid_loader_target = datamodule.valid_dataloaders()
        result = self.neural_dual_solver(
            train_loader_source,
            train_loader_target,
            valid_loader_source,
            valid_loader_target,
        )
        if isinstance(result, tuple):
            pot, logs = result
            self.potentials = pot
            self.metrics["ottlogs"] = logs
            logger.info("Training finished")

    def transport(self, source: jnp.ndarray) -> jnp.ndarray:
        """Transports a batch of data using the Brenier formula."""
        return self.potentials.transport(source)

    def save_checkpoint(self, path: Path) -> None:
        raise NotImplementedError

    def load_checkpoint(self, path: Path) -> None:
        raise NotImplementedError


loss_factory = {"sinkhorn": fitting_loss}
regularizer_factory = {"monge": regularizer}
