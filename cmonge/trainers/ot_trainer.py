import abc
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import jax.numpy as jnp
import optax
from dotmap import DotMap
from flax import linen as nn
from flax.training.orbax_utils import save_args_from_target
from jax.lib import xla_bridge
from loguru import logger
from orbax.checkpoint import PyTreeCheckpointer
from ott.neural.methods.monge_gap import MongeGapEstimator
from ott.neural.methods.neuraldual import W2NeuralDual
from ott.neural.networks.icnn import ICNN
from ott.neural.networks.potentials import PotentialMLP

from cmonge.datasets.single_loader import AbstractDataModule
from cmonge.evaluate import (
    init_logger_dict,
    log_mean_metrics,
    log_metrics,
    log_point_clouds,
)
from cmonge.metrics import fitting_loss, regularizer
from cmonge.utils import create_or_update_logfile, optim_factory

T = TypeVar("T", bound="AbstractTrainer")


class AbstractTrainer(abc.ABC):
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

    @property
    @abc.abstractmethod
    def model(self) -> nn.Module:
        """Abstract method that returns the attribute to be checkpointed."""
        pass

    @model.setter
    @abc.abstractmethod
    def model(self, value: nn.Module):
        """Abstract property to set the checkpointed attribute."""
        pass

    def save_checkpoint(
        self, path: Optional[Path] = None, config: Optional[DotMap] = None
    ) -> None:
        """Abstract method for saving model parameters to a pickle file.

        Args:
            path: Path where the checkpoint should be saved. Defaults to None in which case
                it is retrieved from config.
            config: The model training configuration with a `checkpointing_path` field.
                Defaults to None.
                NOTE: If `config` and `path` are both not None, `path` takes preference.
        """
        if path is None and config is None:
            raise ValueError(
                "Checkpoint cannot be saved. Provide a checkpoint save path either directly or through the config."
            )
        elif path is None:
            path = config.checkpointing_path
        try:
            checkpointer = PyTreeCheckpointer()
            save_args = save_args_from_target(self.model)
            checkpointer.save(path, self.model, save_args=save_args, force=True)
        except Exception as e:
            raise Exception(f"Error in saving checkpoint to {path}: {e}")

    @classmethod
    def load_checkpoint(
        cls: Type[T],
        jobid: int,
        logger_path: Path,
        config: DotMap,
        ckpt_path: Path = None,
        *args,
        **kwargs,
    ) -> T:
        """
        Loading a model from a checkpoint

        Args:
            cls: Class object to be created.
            jobid: ID of the job.
            logger_path: Path where the logging files are stored.
            config: Model training configuration.
            ckpt_path: Optional path from where checkpoint is restored.
                Defaults to None, in that case inferred from config.

        Returns:
            Class object with restored weights.
        """
        try:
            out_class = cls(
                jobid=jobid, logger_path=logger_path, config=config, *args, **kwargs
            )

            if ckpt_path is None:
                if len(config.checkpointing_path) > 0:
                    ckpt_path = config.checkpointing_path
                else:
                    logger.error(
                        "Provide checkpointing path either directly or through the model config"
                    )
            checkpointer = PyTreeCheckpointer()
            out_class.model = checkpointer.restore(ckpt_path, item=out_class.model)

            logger.info("Loaded model from checkpoint")
            return out_class
        except Exception as e:
            raise Exception(
                f"Failed to load checkpoin from {ckpt_path}: {e}\nAre you sure checkpoint was saved and correct path is provided?"
            )

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


class MongeGapTrainer(AbstractTrainer):
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
        checkpointing_path: Optional[str] = None,  # For compatibility with base class
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
        model = PotentialMLP(dim_hidden=dim_hidden, is_potential=False, act_fn=nn.gelu)

        # setup optimizer and scheduler
        opt_fn = optim_factory[optim.name]
        lr_scheduler = optax.cosine_decay_schedule(
            init_value=optim.lr, decay_steps=num_train_iters, alpha=1e-2
        )
        optimizer = opt_fn(learning_rate=lr_scheduler, **optim.kwargs)

        # setup ott-jax solver
        self.solver = MongeGapEstimator(
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
        """Trains a Monge Gap estimator."""
        logger.info("Training started")
        train_loader_source, train_loader_target = datamodule.train_dataloaders()
        valid_loader_source, valid_loader_target = datamodule.valid_dataloaders()
        self.solver.state_neural_net, logs = self.solver.train_map_estimator(
            trainloader_source=train_loader_source,
            trainloader_target=train_loader_target,
            validloader_source=valid_loader_source,
            validloader_target=valid_loader_target,
        )
        self.metrics["ottlogs"] = logs
        logger.info("Training finished.")

    def transport(self, source: jnp.ndarray) -> jnp.ndarray:
        """Transports a batch of data using the learned model."""
        return self.solver.state_neural_net.apply_fn(
            {"params": self.solver.state_neural_net.params}, source
        )

    @property
    def model(self) -> nn.Module:
        return self.solver.state_neural_net

    @model.setter
    def model(self, value: nn.Module):
        """Setter for the model to be checkpointed."""
        self.solver.state_neural_net = value


class MongeMapTrainer(MongeGapTrainer):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "MongeMapTrainer is deprecated and will be removed in a future release. "
            "Please use MongeGapTrainer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


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
        neural_f = ICNN(
            dim_data=num_genes,
            dim_hidden=dim_hidden,
            gaussian_map_samples=(samples_source, samples_target),
        )
        neural_g = PotentioalMLP(dim_hidden=dim_hidden)

        lr_schedule = optax.cosine_decay_schedule(
            init_value=lr, decay_steps=num_train_iters, alpha=1e-2
        )
        optimizer_f = optax.adamw(learning_rate=lr_schedule, b1=0.5, b2=0.5)
        optimizer_g = optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.999)

        self.neural_dual_solver = W2NeuralDual(
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

    @property
    def model(self) -> nn.Module:
        return self.neural_dual_solver

    @model.setter
    def model(self, value: nn.Module):
        """Setter for the model to be checkpointed."""
        self.neural_dual_solver = value


loss_factory = {"sinkhorn": fitting_loss}
regularizer_factory = {"monge": regularizer}
