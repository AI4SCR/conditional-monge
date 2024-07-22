from typing import Any, Callable, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core import frozen_dict
from jax.nn import initializers
from ott.solvers.nn.layers import PosDefPotentials
from ott.solvers.nn.models import (
    ICNN,
    ModelBase,
    NeuralTrainState,
    PotentialGradientFn_t,
    PotentialValueFn_t,
)


class PICNN(ICNN):
    """partial Input convex neural network (PICNN) architecture."""

    dim_data: int = None
    dim_hidden: Sequence[int] = None
    cond_dim: int = None
    init_std: float = 1e-2
    init_fn: Callable = jax.nn.initializers.normal
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    pos_weights: bool = True
    conditions: jnp.ndarray = None
    factors: jnp.ndarray = None
    means: jnp.ndarray = None

    def setup(self):
        self.num_hidden = len(self.dim_hidden)
        self.input_dim = self.dim_data
        self.num_conditions = len(self.means)
        units = [self.num_conditions] + list(self.dim_hidden)
        self.n_layers = len(units)

        # self layers for hidden state u, when contributing all ~0
        w = []
        for odim in [units[0]] + units[:-1]:
            _w = nn.Dense(
                odim,
                use_bias=True,
                kernel_init=self.init_fn(self.init_std),
                bias_init=self.init_fn(self.init_std),
            )
            w.append(_w)
        self.w = w

        # first layer for hidden state performs a comparison with database
        self.w_0 = self.conditions

        # auto layers for z, should be mean operators with no bias
        # keep track of previous size to normalize accordingly
        wz = []
        # keep track of previous size to normalize accordingly
        normalization = 1

        for odim in units[1:] + [1]:
            wz.append(
                nn.Dense(
                    odim,
                    kernel_init=initializers.constant(1.0 / normalization),
                    use_bias=False,
                )
            )
            normalization = odim
        self.wz = wz

        # for family of convex functions stored in z, if using init then first
        # vector z_0 has as many values as # of convex potentials.
        w_z0 = PosDefPotentials(
            dim_data=self.dim_data,
            num_potentials=self.num_conditions,
            use_bias=True,
            kernel_init=lambda *_: self.factors,
            bias_init=lambda *_: self.means,
        )
        self.w_z0 = w_z0

        # cross layers for convex functions z / hidden state u
        # initialized to be identity first with 0 bias
        # and then ~0 + 1 bias to ensure identity
        wzu = []
        _wzu = nn.Dense(
            units[0],
            use_bias=True,
            kernel_init=self.init_fn(self.init_std),
            bias_init=initializers.constant(1.0),
        )
        wzu.append(_wzu)

        for odim in units[1:]:
            _wzu = nn.Dense(
                odim,
                use_bias=True,
                kernel_init=self.init_fn(self.init_std),
                bias_init=initializers.constant(1.0),
            )
            wzu.append(_wzu)
        self.wzu = wzu

        # self layers for x, ~0
        wx = []
        for odim in units + [1]:
            _wx = nn.Dense(
                odim,
                use_bias=True,
                kernel_init=self.init_fn(self.init_std),
                bias_init=initializers.constant(0.0),
            )
            wx.append(_wx)
        self.wx = wx

        # cross layers for x / hidden state u, all ~0
        wxu = []
        for idim in [units[0]] + units:
            _wxu = nn.Dense(
                self.dim_data,
                use_bias=True,
                kernel_init=self.init_fn(self.init_std),
                bias_init=initializers.constant(0.0),
            )
            wxu.append(_wxu)
        self.wxu = wxu

        # self layers for hidden state u, to update z, all ~0
        wu = []
        for odim in units + [1]:
            _wu = nn.Dense(
                odim, use_bias=False, kernel_init=self.init_fn(self.init_std)
            )
            wu.append(_wu)
        self.wu = wu

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        u = jax.nn.softmax(10 * c @ self.w_0)
        z_0 = self.w_z0(x)
        z = self.act_fn(z_0 * u)
        # apply k layers - 1
        for i in range(1, self.n_layers):
            u = self.act_fn(self.w[i](u))
            t_u = jax.nn.softplus(self.wzu[i - 1](u))
            z = self.act_fn(
                self.wz[i - 1](jnp.multiply(z, t_u))
                + self.wx[i](jnp.multiply(x, self.wxu[i](u)))
                + self.wu[i](u)
            )

        z = (
            self.wz[-1](jnp.multiply(z, jax.nn.softplus(self.wzu[-1](u))))
            + self.wx[-1](jnp.multiply(x, self.wxu[-1](u)))
            + self.wu[-1](u)
        )
        return z.squeeze()

    def create_train_state(
        self,
        rng: jnp.ndarray,
        optimizer: optax.OptState,
        input_shape: Union[int, Tuple[int, ...]],
        **kwargs: Any,
    ) -> NeuralTrainState:
        """Create initial `TrainState`."""
        condition = jnp.ones((1, self.cond_dim))
        params = self.init(rng, x=jnp.ones((1, input_shape)), c=condition)["params"]
        return NeuralTrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=optimizer,
            potential_value_fn=self.potential_value_fn,
            potential_gradient_fn=self.potential_gradient_fn,
            **kwargs,
        )

    def potential_value_fn(
        self,
        params: frozen_dict.FrozenDict[str, jnp.ndarray],
    ) -> PotentialValueFn_t:
        """A function that can be evaluated to obtain a potential value, or a linear
        interpolation of a potential.
        """
        return lambda x, c: self.apply({"params": params}, x=x, c=c)  # type: ignore[misc]

    def potential_gradient_fn(
        self,
        params: frozen_dict.FrozenDict[str, jnp.ndarray],
    ) -> PotentialGradientFn_t:
        """Return a function returning a vector or the gradient of the potential.
        Args:
        params: parameters of the module
        Returns
        -------
        A function that can be evaluated to obtain the potential's gradient
        """
        return jax.vmap(jax.grad(self.potential_value_fn(params), argnums=0))


class ConditionalMLP(ModelBase):
    dim_hidden: Sequence[int] = None
    dim_data: int = None
    dim_cond: int = None
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    is_potential: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:  # noqa: D102
        n_input = x.shape[-1]
        z = jnp.concatenate((x, c), axis=1)

        for n_hidden in self.dim_hidden:
            wx = nn.Dense(n_hidden, use_bias=True)
            z = self.act_fn(wx(z))
        wx = nn.Dense(n_input, use_bias=True)
        z = x + wx(z)
        return z

    def create_train_state(
        self,
        rng: jnp.ndarray,
        optimizer: optax.OptState,
        **kwargs: Any,
    ) -> NeuralTrainState:
        """Create initial `TrainState`."""
        c = jnp.ones((1, self.dim_cond))
        x = jnp.ones((1, self.dim_data))
        params = self.init(rng, x=x, c=c)["params"]
        return NeuralTrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=optimizer,
            potential_value_fn=self.potential_value_fn,
            potential_gradient_fn=self.potential_gradient_fn,
            **kwargs,
        )


class DummyMLP(ModelBase):
    """A generic, typically not-convex (w.r.t input) MLP.

    Args:
      dim_hidden: sequence specifying size of hidden dimensions. The output
        dimension of the last layer is automatically set to 1 if
        :attr:`is_potential` is ``True``, or the dimension of the input otherwise
      is_potential: Model the potential if ``True``, otherwise
        model the gradient of the potential
      act_fn: Activation function
    """

    dim_hidden: Sequence[int] = None
    dim_data: int = None
    dim_cond: int = None
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    is_potential: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:  # noqa: D102
        squeeze = x.ndim == 1
        if squeeze:
            x = jnp.expand_dims(x, 0)
        assert x.ndim == 2, x.ndim
        n_input = x.shape[-1]

        z = x
        for n_hidden in self.dim_hidden:
            wx = nn.Dense(n_hidden, use_bias=True)
            z = self.act_fn(wx(z))

        if self.is_potential:
            wx = nn.Dense(1, use_bias=True)
            z = wx(z).squeeze(-1)

            quad_term = 0.5 * jax.vmap(jnp.dot)(x, x)
            z += quad_term
        else:
            wx = nn.Dense(n_input, use_bias=True)
            z = x + wx(z)

        return z.squeeze(0) if squeeze else z

    def create_train_state(
        self,
        rng: jnp.ndarray,
        optimizer: optax.OptState,
        **kwargs: Any,
    ) -> NeuralTrainState:
        """Create initial `TrainState`."""
        c = jnp.ones((1, self.dim_cond))
        x = jnp.ones((1, self.dim_data))
        params = self.init(rng, x=x, c=c)["params"]
        return NeuralTrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=optimizer,
            potential_value_fn=self.potential_value_fn,
            potential_gradient_fn=self.potential_gradient_fn,
            **kwargs,
        )


class ConditionalPerturbationNetwork(ModelBase):
    dim_hidden: Sequence[int] = None
    dim_data: int = None
    dim_cond: int = None
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    is_potential: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:  # noqa: D102
        n_input = x.shape[-1]
        drug_embedding, _ = c[:, :-1], c[:, -1]

        sc = nn.Dense(1, use_bias=True)
        dose_embedding = self.act_fn(sc(c))
        m = nn.Dense(self.dim_cond, use_bias=True)
        drug_embedding = self.act_fn(m(drug_embedding))

        z = jnp.concatenate((x, drug_embedding, dose_embedding), axis=1)

        for n_hidden in self.dim_hidden:
            wx = nn.Dense(n_hidden, use_bias=True)
            z = self.act_fn(wx(z))
        wx = nn.Dense(n_input, use_bias=True)

        return x + wx(z)
        # for n_hidden in self.dim_hidden:
        #     wx = nn.Dense(n_hidden, use_bias=True)
        #     drug_embedding = self.act_fn(wx(c))

        # sc = nn.Dense(1, use_bias=True)
        # dose_embedding = sc(c)

        # m = nn.Dense(n_input, use_bias=True)
        # drug_embedding = m(drug_embedding)

        # return x + drug_embedding * dose_embedding

    def create_train_state(
        self,
        rng: jnp.ndarray,
        optimizer: optax.OptState,
        **kwargs: Any,
    ) -> NeuralTrainState:
        """Create initial `TrainState`."""
        c = jnp.ones((1, self.dim_cond))
        x = jnp.ones((1, self.dim_data))
        params = self.init(rng, x=x, c=c)["params"]
        return NeuralTrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=optimizer,
            potential_value_fn=self.potential_value_fn,
            potential_gradient_fn=self.potential_gradient_fn,
            **kwargs,
        )
