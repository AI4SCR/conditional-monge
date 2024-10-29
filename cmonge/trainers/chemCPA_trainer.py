import collections
import functools
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Tuple

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
from cmonge.models.chemCPA import AdversarialCPAModule, AutoEncoderchemCPA
from cmonge.models.embedding import embed_factory
from cmonge.trainers.ot_trainer import AbstractTrainer, loss_factory
from cmonge.utils import create_or_update_logfile, optim_factory
from cmonge.metrics import average_r2


class ComPertTrainer(AbstractTrainer):
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
        self.num_train_iters = self.config.num_train_iters
        self.split_dose = self.config.embedding.get("split_dose", True)
        self.init_model(datamodule=datamodule)

    def init_model(self, datamodule: ConditionalDataModule):
        # Initial drug embedding
        embed_module = embed_factory[self.config.embedding.name]
        self.embedding_module = embed_module(
            datamodule=datamodule, **self.config.embedding
        )

        # setup loss functions
        reconstruction_loss_fn = loss_factory[self.config.reconstruction_loss.name]
        self.reconstruction_loss = partial(
            reconstruction_loss_fn, **self.config.reconstruction_loss.kwargs
        )

        loss_adversary_drugs_fn = loss_factory[self.config.loss_adversary_drugs.name]
        self.loss_adversary_drugs = partial(
            loss_adversary_drugs_fn, **self.config.loss_adversary_drugs.kwargs
        )

        loss_adversary_covariates_fn = loss_factory[
            self.config.loss_adversary_covariates.name
        ]
        self.loss_adversary_covariates = partial(
            loss_adversary_covariates_fn, **self.config.loss_adversary_covariates.kwargs
        )

        # setup optimizer and scheduler
        opt_fn = optim_factory[self.config.ae_optim.name]
        ae_lr_scheduler = optax.piecewise_constant_schedule(
            init_value=self.config.ae_optim.lr,
            boundaries_and_scales={
                0 + i * self.config.ae_optim.step_size: i * self.config.ae_optim.gamma
                for i in range(self.num_train_iters)
                if i * self.config.ae_optim.step_size < self.num_train_iters
            },
        )
        ae_optimizer = opt_fn(ae_lr_scheduler, **self.config.ae_optim.kwargs)

        opt_fn = optim_factory[self.config.adversary_optim.name]
        adv_lr_scheduler = optax.piecewise_constant_schedule(
            init_value=self.config.adversary_optim.lr,
            boundaries_and_scales={
                0
                + i
                * self.config.adversary_optim.step_size: i
                * self.config.adversary_optim.gamma
                for i in range(self.num_train_iters)
                if i * self.config.adversary_optim.step_size < self.num_train_iters
            },
        )
        adv_drugs_optimizer = opt_fn(
            adv_lr_scheduler, **self.config.adversary_optim.kwargs
        )

        # setup models
        self.autoencoder = AutoEncoderchemCPA(
            **self.config.ae,
            cov_embed_enc_dims=[
                self.config.adversary.cov_hidden_dims[-1],
                self.config.adversary.cov_hidden_dims[0],
            ],
        )
        self.adv_clfs = AdversarialCPAModule(**self.config.adversary)

        # setup training states and step functions
        # self.step_fn = self._get_step_fn()
        self.key, rng1, rng2 = jax.random.split(self.key, 3)
        self.state_autoencoder = self.autoencoder.create_train_state(rng1, ae_optimizer)
        self.state_adv_clfs = self.adv_clfs.create_train_state(
            rng2, adv_drugs_optimizer, self.config.ae.encoder_hidden_dims[-1]
        )

        # autoencoder, doser, drug embedding encoder and coveriate embedders all included in autoencoder optimization
        # Then adversary covariates and adversary drugs are one optimizer

        self.step_fn = self._get_step_fn()

    def train(self, datamodule: ConditionalDataModule):
        logs = collections.defaultdict(lambda: collections.defaultdict(list))
        try:
            from tqdm import trange

            tbar = trange(self.num_train_iters, leave=True)
        except ImportError:
            tbar = range(self.num_train_iters)

        train_conditions = []
        ae_grads = tree_map(jnp.zeros_like, self.state_autoencoder.params)
        adv_grads = tree_map(jnp.zeros_like, self.state_adv_clfs.params)
        n_adv_steps = 0
        n_ae_steps = 0
        for step in tbar:
            is_adv_step = (step + 1) % self.config.adv_step_interval == 0
            is_logging_step = step % 1 == 0
            train_batch, condition = self.generate_batch(datamodule, "train")

            valid_batch, _ = (
                ({"num_contexts": None}, None)
                if not is_logging_step
                else self.generate_batch(datamodule, "valid")
            )

            if is_adv_step:
                self.state_adv_clfs, adv_grads, current_logs = self.step_fn(
                    self.state_adv_clfs,
                    grads=adv_grads,
                    train_batch=train_batch,
                    valid_batch=valid_batch,
                    step=step,
                    n_specific_steps=n_adv_steps,
                    n_train_contexts=train_batch["num_contexts"],
                    n_valid_contexts=valid_batch["num_contexts"],
                )
                n_adv_steps += 1
            else:
                self.state_autoencoder, ae_grads, current_logs = self.step_fn(
                    self.state_autoencoder,
                    grads=ae_grads,
                    train_batch=train_batch,
                    valid_batch=valid_batch,
                    step=step,
                    n_specific_steps=n_ae_steps,
                    n_train_contexts=train_batch["num_contexts"],
                    n_valid_contexts=valid_batch["num_contexts"],
                )
                n_ae_steps += 1

            train_conditions.append(condition)

            if is_logging_step:
                self.update_logs(current_logs, logs, tbar, is_adv_step)
        self.metrics["ott-logs"] = logs
        self.metrics["train_conditions"] = train_conditions

        return self.state_autoencoder, self.adv_clfs, logs

    def _get_step_fn(self) -> Callable:
        """Create a one step training and evaluation function."""

        def ae_loss_fn(
            params: frozen_dict.FrozenDict,
            batch_stats,
            apply_fn: Callable,
            batch: Dict[str, jnp.ndarray],
            n_contexts: int,
            train: bool = True,
            latent_basal=None,  # for compatability
            aux_grad=None,  # for compatability
        ) -> Tuple[float, Dict[str, float]]:
            """Loss function."""
            # Predictions
            outs = apply_fn(
                {"params": params, "batch_stats": batch_stats},
                x=batch["target"],
                c=batch["condition"],
                covs=batch["target_ct"],
                n_contexts=n_contexts,
                train=train,
                mutable=["batch_stats"] if train else False,
            )

            (x_hat, cell_drug_embedding, latent_basal), batch_stats = (
                outs if train else (outs, None)
            )
            outs = self.adv_clfs.apply(
                {
                    "params": self.state_adv_clfs.params,
                    "batch_stats": self.state_adv_clfs.batch_stats,
                },
                latent_basal,
                train=train,
                mutable=["batch_stats"] if train else False,
            )

            (cov_pred, drug_pred), batch_stats = outs if train else (outs, None)
            # get mean and var for predicted gex
            dim = x_hat.shape[1] // 2
            mean = x_hat[:, :dim]
            var = x_hat[:, dim:]

            # compute the loss
            reconstruction_loss = self.reconstruction_loss(
                pred=mean, target=batch["target"], var=var
            )
            adv_drug_loss = self.loss_adversary_drugs(
                labels=batch["target_didx"], probs=drug_pred
            )
            adv_cov_loss = self.loss_adversary_covariates(
                labels=batch["target_ct"], probs=cov_pred
            )

            tot_loss = (
                reconstruction_loss
                - self.config.reg_adversary_drug * adv_drug_loss
                - self.config.reg_adversary_cov * adv_cov_loss
            )

            # store training logs
            loss_logs = {
                "total_ae_loss": tot_loss,
                "reconstruction_loss": reconstruction_loss,
                "adv_drug_loss": adv_drug_loss,
                "adv_cov_loss": adv_cov_loss,
            }

            return tot_loss, (loss_logs, batch_stats)

        def adv_loss_fn(
            params: frozen_dict.FrozenDict,
            batch_stats,
            apply_fn: Callable,
            batch: Dict[str, jnp.ndarray],
            latent_basal: jnp.ndarray,
            adv_cov_grad_penalty: jnp.ndarray,
            adv_drugs_grad_penalty: jnp.ndarray,
            train: bool = True,
            n_contexts=None,  # for compatability
        ) -> Tuple[float, Dict[str, float]]:
            """Loss function."""

            outs = apply_fn(
                {"params": params, "batch_stats": batch_stats},
                latent_basal,
                train=train,
                mutable=["batch_stats"] if train else False,
            )
            (cov_pred, drug_pred), batch_stats = outs if train else (outs, None)

            # compute the loss
            adv_drug_loss = self.loss_adversary_drugs(batch["target_didx"], drug_pred)
            adv_cov_loss = self.loss_adversary_covariates(batch["target_ct"], cov_pred)

            adv_drugs_grad_penalty = jnp.square(adv_drugs_grad_penalty).mean()
            adv_cov_grad_penalty = jnp.square(adv_cov_grad_penalty).mean()

            tot_loss = (
                adv_drug_loss
                + adv_cov_loss
                + self.config.penalty_adversary * adv_drugs_grad_penalty
                + self.config.penalty_adversary * adv_cov_grad_penalty
            )

            # store training logs
            loss_logs = {
                "total_adv_loss": tot_loss,
                "adv_drug_loss": adv_drug_loss,
                "adv_cov_loss": adv_cov_loss,
            }

            return tot_loss, (loss_logs, batch_stats)

        def aux_adv_loss_cov_fn(
            params: frozen_dict.FrozenDict,
            apply_fn: Callable,
            latent_basal: jnp.ndarray,
            train: bool = True,
            batch=None,  # for compatability
            n_contexts=None,  # for compatability
        ) -> Tuple[float, Dict[str, float]]:
            """Loss function."""

            (cov_pred, drug_pred), batch_stats = apply_fn(
                {"params": params},
                latent_basal,
                train=train,
                mutable=["batch_stats"] if train else False,
            )

            return cov_pred.sum(), batch_stats

        def aux_adv_loss_drugs_fn(
            params: frozen_dict.FrozenDict,
            apply_fn: Callable,
            latent_basal: jnp.ndarray,
            train: bool = True,
            batch=None,  # for compatability
            n_contexts=None,  # for compatability
        ) -> Tuple[float, Dict[str, float]]:
            """Loss function."""

            (cov_pred, drug_pred), batch_stats = apply_fn(
                {"params": params},
                latent_basal,
                train=train,
                mutable=["batch_stats"] if train else False,
            )

            return drug_pred.sum(), batch_stats

        @functools.partial(jax.jit, static_argnums=[4, 5, 6, 7])
        def step_fn(
            state_neural_net: train_state.TrainState,
            grads: frozen_dict.FrozenDict,
            train_batch: Dict[str, jnp.ndarray],
            valid_batch: Optional[Dict[str, jnp.ndarray]],
            step: int,
            n_specific_steps: int,
            n_train_contexts: int = 2,
            n_valid_contexts: int = 2,
        ) -> Tuple[train_state.TrainState, frozen_dict.FrozenDict, Dict[str, float]]:
            """Step function."""

            # Step functions booleans
            is_logging_step = step % 1 == 0
            is_adv_step = (step + 1) % self.config.adv_step_interval == 0
            is_gradient_acc_step = (
                n_specific_steps + 1
            ) % self.config.grad_acc_steps == 0

            # compute loss and gradients
            if not is_adv_step:
                grad_fn = jax.value_and_grad(ae_loss_fn, argnums=0, has_aux=True)
                (_, (current_train_logs, batch_stats)), step_grads = grad_fn(
                    state_neural_net.params,
                    state_neural_net.batch_stats,
                    state_neural_net.apply_fn,
                    train_batch,
                    n_train_contexts,
                    train=True,
                )
                # logging step
                current_logs = {"train": current_train_logs, "eval": {}}
                if is_logging_step:
                    _, (current_eval_logs, _) = ae_loss_fn(
                        params=state_neural_net.params,
                        batch_stats=state_neural_net.batch_stats,
                        apply_fn=state_neural_net.apply_fn,
                        batch=valid_batch,
                        n_contexts=n_valid_contexts,
                        train=False,
                    )
            else:
                # map samples with the fitted map
                (x_hat, cell_drug_embedding, latent_basal), batch_stats = (
                    self.autoencoder.apply(
                        {"params": self.state_autoencoder.params},
                        x=train_batch["target"],
                        c=train_batch["condition"],
                        covs=train_batch["target_ct"],
                        train=True,
                        mutable=["batch_stats"],
                    )
                )
                aux_grad_fn = jax.value_and_grad(
                    aux_adv_loss_cov_fn, argnums=2, has_aux=True
                )
                batch_stats, aux_grad_cov = aux_grad_fn(
                    state_neural_net.params,
                    state_neural_net.apply_fn,
                    latent_basal,
                    train=True,
                )
                aux_grad_fn = jax.value_and_grad(
                    aux_adv_loss_drugs_fn, argnums=2, has_aux=True
                )
                batch_stats, aux_grad_drugs = aux_grad_fn(
                    state_neural_net.params,
                    state_neural_net.apply_fn,
                    latent_basal,
                    train=True,
                )
                grad_fn = jax.value_and_grad(adv_loss_fn, argnums=0, has_aux=True)

                (_, (current_train_logs, batch_stats)), step_grads = grad_fn(
                    state_neural_net.params,
                    state_neural_net.batch_stats,
                    state_neural_net.apply_fn,
                    train_batch,
                    latent_basal,
                    aux_grad_cov,
                    aux_grad_drugs,
                    train=True,
                )

                # logging step
                current_logs = {"train": current_train_logs, "eval": {}}
                if is_logging_step:
                    x_hat, cell_drug_embedding, latent_basal = self.autoencoder.apply(
                        {
                            "params": self.state_autoencoder.params,
                            "batch_stats": self.state_autoencoder.batch_stats,
                        },
                        x=valid_batch["target"],
                        c=valid_batch["condition"],
                        covs=valid_batch["target_ct"],
                        train=False,
                        mutable=False,
                    )
                    _, (current_eval_logs, _) = adv_loss_fn(
                        params=state_neural_net.params,
                        batch_stats=state_neural_net.batch_stats,
                        apply_fn=state_neural_net.apply_fn,
                        batch=valid_batch,
                        latent_basal=latent_basal,
                        adv_cov_grad_penalty=jnp.array([0]),
                        adv_drugs_grad_penalty=jnp.array([0]),
                        train=False,
                    )
            current_logs["eval"] = current_eval_logs

            # Accumulate gradients
            grads = tree_map(lambda g, step_g: g + step_g, grads, step_grads)

            # update state
            if is_gradient_acc_step:
                grads = tree_map(lambda g: g / self.config.grad_acc_steps, grads)
                state_neural_net = state_neural_net.apply_gradients(grads=grads)
                # Reset gradients
                grads = tree_map(jnp.zeros_like, grads)

            return state_neural_net, grads, current_logs

        return step_fn

    def generate_batch(
        self,
        datamodule: ConditionalDataModule,
        split_type: str,
    ) -> Dict[str, jnp.ndarray]:
        """Generate a batch of condition and samples."""
        condition_to_loaders = datamodule.get_loaders_by_type(split_type)
        condition = datamodule.sample_condition(split_type)
        loader_source, loader_target = condition_to_loaders[condition]
        embeddings, n_contexts = self.embedding_module(
            condition=condition, dose_split=self.split_dose
        )
        source, source_cell_type, source_drug_idx = next(loader_source)
        target, target_cell_type, target_drug_idx = next(loader_target)
        return (
            {
                "source": source,
                "source_ct": source_cell_type,
                "source_didx": source_drug_idx,
                "target": target,
                "target_ct": target_cell_type,
                "target_didx": target_drug_idx,
                "condition": embeddings,
                "num_contexts": n_contexts,
            },
            condition,
        )

    def update_logs(self, current_logs, logs, tbar, is_adv_step):
        # store and print metrics if logging step
        for log_key in current_logs:
            for metric_key in current_logs[log_key]:
                logs[log_key][metric_key].append(current_logs[log_key][metric_key])

        # update the tqdm bar if tqdm is available
        if not isinstance(tbar, range):
            if not is_adv_step:
                postfix_str = (
                    f"reconstruction_loss: {current_logs['eval']['reconstruction_loss']:.4f}, "
                    f"total_ae_loss: {current_logs['eval']['total_ae_loss']:.4f}, "
                    f"adv_drug_loss: {current_logs['eval']['adv_drug_loss']:.4f}, "
                    f"adv_cov_loss: {current_logs['eval']['adv_cov_loss']:.4f}"
                )
            else:
                postfix_str = (
                    f"adv_drug_loss: {current_logs['eval']['adv_drug_loss']:.4f}, "
                    f"adv_cov_loss: {current_logs['eval']['adv_cov_loss']:.4f}, "
                    f"total_adv_loss: {current_logs['eval']['total_adv_loss']:.4f}"
                )
            tbar.set_postfix_str(postfix_str)

    def predict(
        self, source, target, cond_embeddings, num_contexts, train: bool = False
    ):
        source, source_celltype, source_drugs = source
        target, target_celltype, target_drugs = target

        x_hat, cell_drug_embedding, latent_basal = self.autoencoder.apply(
            {
                "params": self.state_autoencoder.params,
                "batch_stats": self.state_autoencoder.batch_stats,
            },
            x=source,
            c=cond_embeddings,
            covs=target_celltype,
            n_contexts=num_contexts,
            train=train,
            mutable=["batch_stats"] if train else False,
        )

        return x_hat

    def evaluate(
        self,
        datamodule: ConditionalDataModule,
        identity: bool = False,
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
                    pred_m_v = self.predict(
                        source, target, cond_embeddings, n_contexts, train=False
                    )
                    dim = int(pred_m_v.shape[1] / 2)
                    pred_m = pred_m_v[:, :dim]
                    pred_v = pred_m_v[:, dim:]
                else:
                    pred = source[0]
                    pred_m = source.mean(axis=0)
                    pred_v = source.var(axis=0)
                target_m = target.mean(axis=0)
                target_v = target.var(axis=0)
                if datamodule.marker_idx:
                    target = target[0][:, datamodule.marker_idx]
                    pred_m_v = pred[:, datamodule.marker_idx]

                metrics["r2_mean"].append(average_r2(target_m, pred_m))
                metrics["r2_var"].append(average_r2(target_v, pred_v))

        def evaluate_split(
            cond_to_loaders: Dict[
                str, Tuple[Iterator[jnp.ndarray], Iterator[jnp.ndarray]]
            ],
            split_type: str,
        ):
            self.metrics[split_type] = {}
            for cond, loader in cond_to_loaders.items():
                logger.info(f"Evaluation started on {cond} {split_type}.")
                cond_embedding, n_contexts = self.embedding_module(
                    cond, self.split_dose
                )
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
                self.metrics["mean_statistics"]["mean_r2_mean"] = float(
                    sum(self.metrics["r2_mean"]) / len(self.metrics["r2_mean"])
                )
                self.metrics["mean_statistics"]["mean_r2_var"] = float(
                    sum(self.metrics["r2_var"]) / len(self.metrics["r2_var"])
                )

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

    def save_checkpoint(self, path: Path) -> None:
        ae_ckpt = self.state_autoencoder
        adv_ckpt = self.state_adv_clfs

        checkpointer = PyTreeCheckpointer()
        save_args = save_args_from_target(ae_ckpt)
        checkpointer.save(
            Path(path) / "autoencoder", ae_ckpt, save_args=save_args, force=True
        )

        checkpointer = PyTreeCheckpointer()
        save_args = save_args_from_target(adv_ckpt)
        checkpointer.save(
            Path(path) / "adv_clfs", adv_ckpt, save_args=save_args, force=True
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
        checkpointer = PyTreeCheckpointer()
        out_class.state_autoencoder = checkpointer.restore(
            Path(ckpt_path) / "autoencoder", item=out_class.state_autoencoder
        )
        checkpointer = PyTreeCheckpointer()
        out_class.state_adv_clfs = checkpointer.restore(
            Path(ckpt_path) / "adv_clfs", item=out_class.state_adv_clfs
        )
        logger.info("Loaded ConditionalMongeTrainer from checkpoint")
        return out_class
