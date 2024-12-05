from typing import Any, Callable, Iterable, Sequence, Tuple, Union
import flax.linen as nn
import jax.numpy as jnp
import optax
from flax.training import train_state


class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    batch_stats: Any


class MLPchemCPA(nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.

    Careful: if activation is set to ReLU, ReLU is only applied to the first half of NN outputs!
    """

    hidden_dims: Sequence[
        int
    ]  # in chemCPA in and out dims are given, I think first hidden dim is actually data_dim.
    # We need one hidden dim less here since input dim is inferred from data
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    batch_norm: bool = True

    @nn.compact
    def __call__(self, x, train: bool = True):

        for i, n_hidden in enumerate(self.hidden_dims):
            w = nn.Dense(n_hidden, use_bias=True)
            x = w(x)
            if (
                i < len(self.hidden_dims) - 1
            ):  # Don't add activation and batch on last layer
                if self.batch_norm:
                    bn = nn.BatchNorm()
                    x = bn(x, use_running_average=not train)
                x = self.act_fn(x)

        return x

    def create_train_state(
        self,
        rng: jnp.ndarray,
        optimizer: optax.OptState,
        input_shape: Union[int, Tuple[int, ...]],
        **kwargs: Any,
    ) -> TrainState:

        variables = self.init(rng, jnp.ones((1, input_shape)))

        model_train_state = TrainState.create(
            apply_fn=self.apply,
            params=variables["params"],
            tx=optimizer,
            batch_stats=variables["batch_stats"],
        )

        return model_train_state


class AutoEncoderchemCPA(nn.Module):
    """Flax AutoEncoder module based on chemCPA"""

    encoder_hidden_dims: Sequence[int]
    decoder_hidden_dims: Sequence[int]
    drug_embed_enc_dims: Sequence[int]
    cov_embed_enc_dims: Sequence[int]
    doser_hidden_dims: Sequence[int]
    seed: int
    context_entity_bonds: Iterable[Tuple[int, int]]
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    batch_norm: bool = True

    def setup(self):
        self.encoder = MLPchemCPA(
            hidden_dims=self.encoder_hidden_dims,
            act_fn=self.act_fn,
            batch_norm=self.batch_norm,
        )
        self.decoder = MLPchemCPA(
            hidden_dims=self.decoder_hidden_dims,
            act_fn=self.act_fn,
            batch_norm=self.batch_norm,
        )
        self.drug_embed_enc = MLPchemCPA(self.drug_embed_enc_dims)
        self.cov_embed_enc = nn.Embed(
            self.cov_embed_enc_dims[0], self.cov_embed_enc_dims[1]
        )
        self.doser = MLPchemCPA(self.doser_hidden_dims)
        self.degs_predictor = MLPchemCPA(
            [self.encoder_hidden_dims[-1], int(self.decoder_hidden_dims[-1] / 2)],
            batch_norm=True,
        )

    @nn.compact
    def __call__(self, x, c, covs, n_contexts: int = None, train: bool = True):
        """
        Args:
            x (jnp.ndarray): The input data of shape bs x dim_data
            c (jnp.ndarray): The context of shape bs x dim_cond with possibly different modalities
                concatenated, as can be specified via context_entity_bonds.
            n_contexts (int): Number of contexts in c, contexts bounds should be defined on init in `context_entity_bonds`
            covs: covariate index to be encoded and added to the latent vector
        Returns:
            jnp.ndarray: _description_
        """

        # Chunk the inputs
        drugs = c[:, :-1]
        log_dose = c[:, -1]
        dose = jnp.exp(log_dose)  # RDkit embedding takes log of dose
        dose = round(dose / 10000, 5)  # chemCPA uses smaller dose values
        c = jnp.concatenate([drugs, jnp.atleast_2d(dose).T], axis=1)
        latent_drugs = self.drug_embed_enc(drugs, train)
        latent_dosages = self.doser(c, train).reshape(-1)
        drug_embedding = jnp.einsum("i,ij->ij", latent_dosages, latent_drugs)

        cov_embedding = self.cov_embed_enc(covs)

        latent_basal = self.encoder(x, train)
        latent_treated = latent_basal + drug_embedding + cov_embedding

        x_hat = self.decoder(latent_treated, train)
        dim = x_hat.shape[1] // 2
        mean = x_hat[:, :dim]
        var = nn.softplus(x_hat[:, dim:])
        x_hat = jnp.concatenate([mean, var], axis=1)

        cell_drug_embedding = jnp.concatenate([cov_embedding, drug_embedding], axis=1)

        degs_pred = self.degs_predictor(cell_drug_embedding, train)

        return x_hat, cell_drug_embedding, latent_basal, degs_pred

    def create_train_state(
        self,
        rng: jnp.ndarray,
        optimizer: optax.OptState,
        **kwargs: Any,
    ) -> TrainState:

        variables = self.init(
            rng,
            x=jnp.ones((1, int(self.decoder_hidden_dims[-1] / 2))),
            c=jnp.ones((1, self.context_entity_bonds[-1][-1])),
            covs=jnp.ones((1,), dtype="int32"),
        )

        model_train_state = TrainState.create(
            apply_fn=self.apply,
            params=variables["params"],
            tx=optimizer,
            batch_stats=variables["batch_stats"],
        )

        return model_train_state


class AdversarialCPAModule(nn.Module):
    """Flax AutoEncoder module based on chemCPA"""

    drugs_hidden_dims: Sequence[int]
    cov_hidden_dims: Sequence[int]
    seed: int
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    batch_norm: bool = True

    def setup(self):
        self.adv_drugs_clsf = MLPchemCPA(hidden_dims=self.drugs_hidden_dims)
        self.adv_cov_clsf = MLPchemCPA(hidden_dims=self.cov_hidden_dims)

    @nn.compact
    def __call__(self, z, train: bool = True):
        """
        Args:
            z (jnp.ndarray): basal cell latent space

        Returns:
            jnp.ndarray: _description_
        """

        cov_pred = self.adv_cov_clsf(z, train=train)
        drug_pred = self.adv_drugs_clsf(z, train=train)

        return cov_pred, drug_pred

    def create_train_state(
        self,
        rng: jnp.ndarray,
        optimizer: optax.OptState,
        dim: int,
        **kwargs: Any,
    ) -> TrainState:

        variables = self.init(rng, z=jnp.ones((1, dim)))

        model_train_state = TrainState.create(
            apply_fn=self.apply,
            params=variables["params"],
            tx=optimizer,
            batch_stats=variables["batch_stats"],
        )

        return model_train_state
