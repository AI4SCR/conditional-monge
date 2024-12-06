import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry import costs
from ott.geometry.pointcloud import PointCloud
from ott.neural.methods.monge_gap import monge_gap_from_samples
from ott.solvers.linear import sinkhorn
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
from sklearn.metrics.pairwise import rbf_kernel


def average_r2(target: jnp.ndarray, transport: jnp.ndarray) -> float:
    """
    Calculate the correlation coefficient r^2 between the means of average features in target and tansport.
    """
    target_means = jnp.mean(target, axis=0)
    transport_means = jnp.mean(transport, axis=0)
    average_r2 = np.corrcoef(target_means, transport_means)[0, 1] ** 2
    return float(average_r2)


def drug_signature(target: jnp.ndarray, transport: jnp.ndarray) -> float:
    """Calculates the euclidien distance between the marginal means of the target and transported measures."""
    target_means = jnp.mean(target, 0)
    transport_means = jnp.mean(transport, 0)
    return float(jnp.linalg.norm(target_means - transport_means))


def maximum_mean_discrepancy(
        target: jnp.ndarray, transport: jnp.ndarray, gamma: float
) -> float:
    """Calculates the maximum mean discrepancy between two measures."""
    xx = rbf_kernel(target, target, gamma)
    xy = rbf_kernel(target, transport, gamma)
    yy = rbf_kernel(transport, transport, gamma)

    return float(xx.mean() + yy.mean() - 2 * xy.mean())


def compute_scalar_mmd(
    target: jnp.ndarray,
    transport: jnp.ndarray,
    gammas: list[float] = [2, 1, 0.5, 0.1, 0.01, 0.005],
):
    """
    Calculates the maximum mean discrepancy between the target
    and the transported measures, using gaussian kernel,averaging for different gammas.
    """

    def safe_mmd(*args):
        try:
            mmd = maximum_mean_discrepancy(*args)
        except ValueError:
            mmd = jnp.nan
        return mmd

    return float(np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas))))


def wasserstein_distance(
        target: jnp.ndarray, transport: jnp.ndarray, epsilon: float = 0.1
) -> float:
    """
    Calculates the Wasserstain distance between two measures
    using the Sinkhorn algorithm on the regularized OT formulation.
    """
    geom = PointCloud(target, transport, cost_fn=costs.Euclidean(), epsilon=epsilon)
    solver = jax.jit(sinkhorn.solve)
    ot = solver(geom)
    return ot.reg_ot_cost


def fitting_loss(
        target: jnp.ndarray, transport: jnp.ndarray, epsilon_fitting: float
) -> float:
    """Calculates the sinkhorn divergence between two measures."""
    out = sinkhorn_divergence(
        PointCloud,
        target,
        transport,
        cost_fn=costs.Euclidean(),
        epsilon=epsilon_fitting,
    )
    return out.divergence


def sinkhorn_div(target: jnp.ndarray, transport: jnp.ndarray) -> float:
    """Calculates the sinkhorn divergence between two measures."""
    return fitting_loss(target, transport, 0.1)


def regularizer(
    target: jnp.ndarray,
    transport: jnp.ndarray,
    epsilon_regularizer: float,
    cost: str,
):
    """Calculates the Monge Gap between two measures."""
    cost_fn = cost_factory[cost]
    gap = monge_gap_from_samples(
        target,
        transport,
        cost_fn=cost_fn,
        epsilon=epsilon_regularizer,
        return_output=False,
    )
    return gap


def eucledian_monge_gap(target: jnp.ndarray, transport: jnp.ndarray) -> float:
    return regularizer(target, transport, 1, "euclidean")


cost_factory = {"euclidean": costs.Euclidean()}
