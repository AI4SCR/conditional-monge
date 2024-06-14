from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from cmonge.utils import activation_factory, optim_factory
from dotmap import DotMap
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.training import checkpoints
from flax.training.train_state import TrainState
from loguru import logger

if TYPE_CHECKING:
    from cmonge.datasets.single_loader import AbstractDataModule


class Encoder(nn.Module):
    """Flax Encoder module."""

    hidden_dims: Sequence[int]
    latent_dim: int
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu

    @nn.compact
    def __call__(self, x):
        for n_hidden in self.hidden_dims:
            W = nn.Dense(n_hidden, use_bias=True)
            x = self.act_fn(W(x))
        x = nn.Dense(features=self.latent_dim)(x)
        return x


class Decoder(nn.Module):
    """Flax Decoder module."""

    hidden_dims: Sequence[int]
    data_dim: int
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu

    @nn.compact
    def __call__(self, x):
        for n_hidden in self.hidden_dims:
            W = nn.Dense(n_hidden, use_bias=True)
            x = self.act_fn(W(x))
        x = nn.Dense(features=self.data_dim)(x)
        return x


class AutoEncoder(nn.Module):
    """Flax AutoEncoder module."""

    hidden_dims: Sequence[int]
    latent_dim: int
    data_dim: int
    seed: int
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu

    def setup(self):
        self.encoder = Encoder(self.hidden_dims, self.latent_dim, self.act_fn)
        self.decoder = Decoder(self.hidden_dims, self.data_dim, self.act_fn)

    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


def mse_reconstruction_loss(model: AutoEncoder, params: FrozenDict, batch: jnp.ndarray):
    """l2 reconstruction loss for the autoencoder."""
    pred = model.apply({"params": params}, batch)
    return 2 * optax.l2_loss(pred, batch).mean()


class AETrainerModule:
    """Class for training, evaluating and saving an AutoEncoder model."""

    def __init__(self, config: DotMap,):
        self.config = config
        self.model_dir = Path(self.config.training.model_dir)
        self.create_functions()
        self.init_model()

    def init_model(self):
        """Initialize optimizer and model parameters."""
        self.config.model.act_fn = activation_factory[self.config.model.act_fn]
        self.model = AutoEncoder(**self.config.model)

        rng = jax.random.PRNGKey(self.config.model.seed)
        rng, init_rng = jax.random.split(rng)
        empty_array = jnp.empty((1, self.config.model.data_dim))
        self.latent_shift = empty_array
        params = self.model.init(init_rng, empty_array)["params"]

        lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.optim.lr,
            warmup_steps=100,
            decay_steps=self.config.training.n_epochs * 100,
            end_value=1e-5,
        )
        opt_fn = optim_factory[self.config.optim.optimizer]
        optimizer = opt_fn(learning_rate=lr_scheduler, **self.config.optim.kwargs)

        self.state = TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optimizer
        )

    def create_functions(self):
        """Define jitted train and eval step on a batch of input."""

        def train_step(state: TrainState, batch: jnp.ndarray):
            loss_fn = lambda params: mse_reconstruction_loss( # noqa: E731
                self.model, params, batch
            )  
            loss, grads = jax.value_and_grad(loss_fn)(
                state.params
            )  # Get loss and gradients for loss
            state = state.apply_gradients(grads=grads)  # Optimizer update step
            return state, loss

        def eval_step(state: TrainState, batch: jnp.ndarray):
            return mse_reconstruction_loss(self.model, state.params, batch)

        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def train(self, datamodule: AbstractDataModule):
        """Train model for n_epochs, save best model after each epoch."""
        logger.info("Training started.")
        best_eval = 1e6
        for epoch in range(self.config.training.n_epochs):
            # Training step
            losses = []
            for batch in datamodule.train_dataloaders():
                self.state, loss = self.train_step(state=self.state, batch=batch)
                losses.append(loss)
            losses_np = np.stack(jax.device_get(losses))
            avg_loss = losses_np.mean()
            logger.info(f"train/loss - epoch {epoch}: {avg_loss}")

            # Validation step
            if self.config.training.valid:
                batch_sizes = []
                losses = []
                for batch in datamodule.valid_dataloaders():
                    loss = self.eval_step(self.state, batch)
                    losses.append(loss)
                    batch_sizes.append(batch[0].shape[0])
                losses_np = np.stack(jax.device_get(losses))
                batch_sizes_np = np.stack(batch_sizes)
                eval_loss = (losses_np * batch_sizes_np).sum() / batch_sizes_np.sum()
                logger.info(f"valid/loss - epoch {epoch}: {eval_loss}")

                # Saving checkpoint if eval metric improved
                if eval_loss < best_eval and self.config.training.cpkt:
                    best_eval = eval_loss
                    self.compute_latent_shift(datamodule)
                    self.save_model(
                        dataset_name=datamodule.name,
                        drug_condition=datamodule.drug_condition,
                        step=epoch,
                    )

        # save the final model if no checkpointing
        if not self.config.training.cpkt:
            self.save_model(
                dataset_name=datamodule.name,
                drug_condition=datamodule.drug_condition,
                step=epoch,
            )
        logger.info("Training finished.")

    def save_model(self, dataset_name: str, drug_condition: str, step: int = 0):
        """Save current model at certain training iteration."""
        model_dir = self.model_dir / dataset_name
        model_dir.mkdir(parents=True, exist_ok=True)
        cpkt = {"params": self.state.params, "latent_shift": self.latent_shift}
        prefix = f"autoencoder_{self.model.latent_dim}_{dataset_name}_{drug_condition}_"
        logger.info(f"Saving model checkpoint to {prefix}")

        checkpoints.save_checkpoint(
            ckpt_dir=model_dir,
            target=cpkt,
            prefix=prefix,
            step=step,
            overwrite=True,
        )
        logger.info("Model checkpoint saved.")

    def load_model(self, dataset_name: str, drug_condition: str):
        """Load model from checkpoint."""
        model_dir = self.model_dir / dataset_name
        target = {"params": self.state.params, "latent_shift": self.latent_shift}
        prefix = f"autoencoder_{self.model.latent_dim}_{dataset_name}_{drug_condition}_"
        logger.info(f"Loading AE model checkpoint from {prefix}")
        cpkt = checkpoints.restore_checkpoint(
            ckpt_dir=model_dir,
            prefix=prefix,
            target=target,
        )
        if cpkt["params"] is self.state.params:
            logger.info("Failed to load AE model checkpoint.")
        else:
            logger.info("AE Model checkpoint loaded.")
            logger.info(f"AE Model: {prefix}")

        self.state = TrainState.create(
            apply_fn=self.model.apply, params=cpkt["params"], tx=self.state.tx
        )
        self.latent_shift = cpkt["latent_shift"]

    def compute_latent_shift(self, datamodule: AbstractDataModule):
        logger.info(f"Computing latent shift for drug {datamodule.drug_condition}.")
        source = datamodule.adata[datamodule.control_train_cells, :].X
        target = datamodule.adata[datamodule.target_train_cells, :].X
        self.model_bd = self.model.bind({"params": self.state.params})
        mean_encoded_source = self.model_bd.encoder(source).mean(0)
        mean_encoded_target = self.model_bd.encoder(target).mean(0)
        self.latent_shift = mean_encoded_target - mean_encoded_source
        logger.info("Latent shift computed.")

    def transport(self, source: jnp.ndarray) -> jnp.ndarray:
        self.model_bd = self.model.bind({"params": self.state.params})
        z = self.model_bd.encoder(source)
        z_hat = z + self.latent_shift
        transport = self.model_bd.decoder(z_hat)
        return transport
