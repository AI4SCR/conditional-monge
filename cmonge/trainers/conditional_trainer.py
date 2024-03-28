from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import optax
from cmonge.datasets.conditional_loader import ConditionalDataModule
from cmonge.evaluate import init_logger_dict, log_mean_metrics, log_metrics
from cmonge.models.nn import PICNN
from cmonge.trainers.ot_trainer import AbstractTrainer
from cmonge.utils import create_or_update_logfile
from dotmap import DotMap
from flax.core.scope import FrozenVariableDict
from loguru import logger
from ott.solvers.nn.conjugate_solvers import DEFAULT_CONJUGATE_SOLVER
from ott.solvers.nn.models import ICNN


class ConditionalTrainer(AbstractTrainer):
    """Wrapper class for conditional neural dual training."""

    def __init__(self, jobid: int, logger_path: Path, config: DotMap, datamodule) -> None:
        super().__init__(jobid, logger_path)
        self.init_model(datamodule=datamodule, **config)
        self.create_functions()

    def init_model(
        self,
        num_train_iters: int,
        num_inner_iters: int,
        lr: float,
        seed: int,
        num_genes: int,
        dim_hidden: list[int],
        cond_dim: int,
        embedding: str,
        method: str,
        pos_weights: bool,
        datamodule,
    ):
        self.num_inner_iters = num_inner_iters
        self.num_train_iters = num_train_iters
        self.lr = lr
        self.seed = seed
        self.num_genes = num_genes
        self.dim_hidden = dim_hidden
        self.num_genes = self.input_dim = num_genes
        self.cond_dim = cond_dim
        self.embedding = embedding
        self.method = method
        self.pos_weights = pos_weights
        self.beta = 1
        loaders = datamodule.train_dataloaders()
        self.generate_embeddings(datamodule)
        w_0 = self.init_modulator_weights(loaders)
        factors, means = self.compute_gaussian_map_params(loaders)

        self.key = jax.random.PRNGKey(self.seed)
        self.key, key_f, key_g = jax.random.split(self.key, 3)

        self.neural_f = PICNN(
            dim_hidden=self.dim_hidden,
            dim_data=self.num_genes,
            cond_dim=self.cond_dim,
            conditions=w_0,
            factors=factors,
            means=means,
        )
        self.neural_g = PICNN(
            dim_hidden=self.dim_hidden,
            dim_data=self.num_genes,
            cond_dim=self.cond_dim,
            conditions=w_0,
            factors=factors,
            means=means,
        )

        self.conjugate_solver = DEFAULT_CONJUGATE_SOLVER

        lr_schedule = optax.cosine_decay_schedule(init_value=self.lr, decay_steps=self.num_inner_iters, alpha=1e-2)
        optimizer_f = optax.adamw(learning_rate=lr_schedule)
        optimizer_g = optax.adamw(learning_rate=lr_schedule)

        self.state_f = self.neural_f.create_train_state(
            key_f,
            optimizer_f,
            self.num_genes,
        )
        self.state_g = self.neural_g.create_train_state(
            key_g,
            optimizer_g,
            self.num_genes,
        )

    def compute_gaussian_map_params(self, train_loaders):
        factors = []
        means = []
        for cond, loaders in train_loaders.items():
            samples = next(loaders[0]), next(loaders[1])
            factor, mean = ICNN._compute_gaussian_map_params(samples)

            factors.append(factor)
            means.append(mean)
        return jnp.vstack(factors), jnp.vstack(means)

    def init_modulator_weights(self, train_loaders):
        conditions = []
        for cond, loaders in train_loaders.items():
            embedding = self.embeddings[cond]
            conditions.append(jnp.asarray(embedding))
        return jnp.stack(conditions).T

    def _penalize_weights_icnn(self, params: FrozenVariableDict) -> float:
        """Penalize weights of ICNN."""
        penalty = 0
        for key in params:
            if key.startswith("wz_"):
                penalty += jnp.linalg.norm(jax.nn.relu(-params[key]["kernel"]))
        return penalty

    def _clip_weights_icnn(self, params: FrozenVariableDict) -> FrozenVariableDict:
        """Clip weights of ICNN."""
        for key in params:
            if key.startswith("wz_"):
                params[key]["kernel"] = jnp.clip(params[key]["kernel"], a_min=0)

        return params

    def get_step_fn(self, train: bool, to_optimize: Literal["f", "g"]):
        """Create a parallel training and evaluation function."""

        def loss_fn(params_f, params_g, f_value, g_value, g_gradient, batch):
            """Loss function for both potentials."""
            source, target, condition = batch["source"], batch["target"], batch["condition"]

            init_source_hat = g_gradient(params_g)(target, condition)
            batch_dot = jax.vmap(jnp.dot)

            f_source = f_value(params_f)(source, condition)
            f_star_target = batch_dot(init_source_hat, target) - f_value(params_f)(init_source_hat, condition)
            dual_source = f_source.mean()
            dual_target = f_star_target.mean()
            loss_f = dual_source + dual_target

            f_value_parameters_detached = f_value(jax.lax.stop_gradient(params_f))
            loss_g = (
                f_value_parameters_detached(init_source_hat, condition) - batch_dot(init_source_hat, target)
            ).mean()

            # compute Wasserstein-2 distance
            C = jnp.mean(jnp.sum(source**2, axis=-1)) + jnp.mean(jnp.sum(target**2, axis=-1))
            W2_dist = C - 2.0 * (f_source.mean() + f_star_target.mean())

            if not self.pos_weights:
                penalty = self.beta * self._penalize_weights_icnn(params_f) + self.beta * self._penalize_weights_icnn(
                    params_g
                )
                loss_f += penalty
                loss_g += penalty
            else:
                penalty = 0

            if to_optimize == "f":
                return loss_f, (W2_dist, loss_f, loss_g, penalty)
            if to_optimize == "g":
                return loss_g, (W2_dist, loss_f, loss_g, penalty)

        @jax.jit
        def step_fn(state_f, state_g, batch):
            """Step function of either training or validation."""
            grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 1], has_aux=True)
            if train:
                # compute loss and gradients
                (_, (W2_dist, loss_f, loss_g, penalty)), (grads_f, grads_g) = grad_fn(
                    state_f.params,
                    state_g.params,
                    state_f.potential_value_fn,
                    state_g.potential_value_fn,
                    state_f.potential_gradient_fn,
                    batch,
                )

                if to_optimize == "f":
                    return state_f.apply_gradients(grads=grads_f), W2_dist, loss_f, loss_g, penalty
                if to_optimize == "g":
                    return state_g.apply_gradients(grads=grads_g), W2_dist, loss_f, loss_g, penalty
                raise ValueError("Optimization target has been misspecified.")

            # compute loss and gradients
            (_, (W2_dist, loss_f, loss_g, penalty)), _ = grad_fn(
                state_f.params,
                state_g.params,
                state_f.potential_value_fn,
                state_g.potential_value_fn,
                state_f.potential_gradient_fn,
                batch,
            )

            return W2_dist, loss_f, loss_g, penalty

        return step_fn

    def create_functions(self):
        self.train_step_f = self.get_step_fn(train=True, to_optimize="f")
        self.train_step_g = self.get_step_fn(train=True, to_optimize="g")
        self.valid_step_f = self.get_step_fn(train=False, to_optimize="f")
        self.valid_step_g = self.get_step_fn(train=False, to_optimize="g")

    def train(self, datamodule: ConditionalDataModule, valid: bool = False):
        logger.info("Training started.")
        condition_to_loaders = datamodule.train_dataloaders()

        self.generate_embeddings(datamodule)

        # self.pretrain_identity(datamodule)

        batch_g, batch_f, valid_batch = {}, {}, {}
        valid_logs = {"loss_f": [], "loss_g": [], "w_dist": []}

        #     return pretrain_logs
        train_logs = {"loss_f": [], "loss_g": [], "w_dist": []}
        for step in range(self.num_train_iters):
            condition = datamodule.sample_condition("train")
            trainloader_source, trainloader_target = condition_to_loaders[condition]
            condition = self.embeddings[condition]
            condition_batch = jnp.asarray([condition for _ in range(datamodule.batch_size)])

            for _ in range(self.num_inner_iters):
                batch_g["source"] = jnp.asarray(next(trainloader_source))
                batch_g["target"] = jnp.asarray(next(trainloader_target))
                batch_g["condition"] = condition_batch

                self.state_g, w_dist, loss_f, loss_g, penalty = self.train_step_g(self.state_f, self.state_g, batch_g)

            batch_f["source"] = jnp.asarray(next(trainloader_source))
            batch_f["target"] = jnp.asarray(next(trainloader_target))
            batch_f["condition"] = condition_batch

            self.state_f, w_dist, loss_f, loss_g, penalty = self.train_step_f(self.state_f, self.state_g, batch_f)

            if valid and step % 100 == 0 and step > 0:
                condition = datamodule.sample_condition("valid")
                valid_condition_to_loaders = datamodule.valid_dataloaders()
                validloader_source, validloader_target = valid_condition_to_loaders[condition]
                condition = self.embeddings[condition]
                condition_batch = jnp.asarray([condition for _ in range(datamodule.batch_size)])

                valid_batch["source"] = jnp.asarray(next(validloader_source))
                valid_batch["target"] = jnp.asarray(next(validloader_target))
                valid_batch["condition"] = condition_batch

                valid_w_dist, valid_loss_f, valid_loss_g, penalty = self.valid_step_g(
                    self.state_f, self.state_g, valid_batch
                )

                self.update_logging(train_logs, loss_f, loss_g, w_dist, "train", step)
                self.update_logging(valid_logs, valid_loss_f, valid_loss_g, valid_w_dist, "valid", step)

        self.metrics["ottlogs"] = {"train_logs": train_logs, "valid_logs": valid_logs}
        logger.info("Training finished.")

    def update_logging(self, log_dict, loss_f, loss_g, w_dist, kind, step):
        log_dict["loss_f"].append(loss_f)
        log_dict["loss_g"].append(loss_g)
        log_dict["w_dist"].append(w_dist)
        loss_f_format = "{:.2f}".format(loss_f)
        loss_g_format = "{:.2f}".format(loss_g)
        w_dist_fomrat = "{:.2f}".format(w_dist)
        logger.info(
            f"{step} n_iters - {kind} - loss_f: {loss_f_format}, loss_g: {loss_g_format}, w_dist: {w_dist_fomrat}"
        )
        return log_dict

    def generate_embeddings(self, datamodule: ConditionalDataModule):
        assert self.embedding in ["dosage", "dense"]
        if self.embedding == "dosage":
            embeddings = {cond: [int(cond.split("-")[1]) / 10000] for cond in datamodule.conditions}
            self.embeddings = embeddings
            datamodule.embeddings = embeddings

    def transport(self, x: jnp.ndarray, condition: jnp.ndarray) -> jnp.ndarray:
        """Conditionally transport according to Brenier formula."""

        def f(x, c) -> float:
            return self.state_g.apply_fn({"params": self.state_g.params}, x, c)

        return jax.vmap(jax.grad(f, argnums=0))(x, condition)

    def evaluate(
        self,
        datamodule: ConditionalDataModule,
        identity: bool = False,
        n_samples: int = 9,
    ) -> None:
        """Evaluate a trained model on a validation set and save the metrics to a json file."""

        def evaluate_condition(loader_source, loader_target, cond_embeddings, metrics):
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

        cond_to_loaders = datamodule.train_dataloaders()
        for cond, loader in cond_to_loaders.items():
            logger.info(f"Evaluation started on {cond} (in-sample).")
            cond_embedding = self.embeddings[cond]
            cond_embedding = jnp.asarray([[cond_embedding] for _ in range(datamodule.batch_size)])
            loader_source, loader_target = loader

            self.metrics[f"in-sample-{cond}"] = {}
            init_logger_dict(self.metrics[f"in-sample-{cond}"], datamodule.drug_condition)
            evaluate_condition(loader_source, loader_target, cond_embedding, self.metrics[f"in-sample-{cond}"])
            log_mean_metrics(self.metrics[f"in-sample-{cond}"])

        cond_to_loaders = datamodule.valid_dataloaders()
        for cond, loader in cond_to_loaders.items():
            logger.info(f"Evaluation started on {cond} (out-sample).")
            cond_embedding = self.embeddings[cond]
            cond_embedding = jnp.asarray([[cond_embedding] for _ in range(datamodule.batch_size)])
            loader_source, loader_target = loader

            self.metrics[f"out-sample-{cond}"] = {}
            init_logger_dict(self.metrics[f"out-sample-{cond}"], datamodule.drug_condition)
            evaluate_condition(loader_source, loader_target, cond_embedding, self.metrics[f"out-sample-{cond}"])
            log_mean_metrics(self.metrics[f"out-sample-{cond}"])

        create_or_update_logfile(self.logger_path, self.metrics)
