import jax.numpy as jnp
import jax


@jax.jit
def r2(y_pred: jnp.ndarray, y_true: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """ R2 metric """
    SS_res = jnp.sum(jnp.square(y_true - y_pred), axis=0)
    SS_tot = jnp.sum(jnp.square(y_true - jnp.mean(y_true, axis=0)), axis=0)
    return 1 - SS_res / jnp.clip(SS_tot, a_min=eps)


@jax.jit
def mae(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """ Mean Absolute Error """
    return jnp.mean(jnp.abs(y_true - y_pred))


@jax.jit
def mse(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """ Mean Squared Error """
    return jnp.mean(jnp.square(y_true - y_pred))


@jax.jit
def rmse(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """ Root Mean Squared Error """
    return jnp.sqrt(mse(y_true, y_pred))


@jax.jit
def gaussian_negative_log_likelihood(mean: jnp.ndarray, variance: jnp.ndarray, targets: jnp.ndarray,
                                     eps: float = 1e-6) -> jnp.ndarray:
    first_term = jnp.log(jnp.maximum(2 * jnp.pi * variance, eps))
    second_term = jnp.square((targets - mean)) / jnp.clip(variance, a_min=eps)
    return jnp.mean(0.5 * (first_term + second_term))


@jax.jit
def mape(y_pred: jnp.ndarray, y_true: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """ Mean Absolute Percentage Error """
    return jnp.mean(jnp.abs(y_true - y_pred) / jnp.clip(jnp.abs(y_true), a_min=eps)) * 100


@jax.jit
def binary_cross_entropy(y_pred: jnp.ndarray, y_true: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """ Binary Cross Entropy """
    return jnp.mean(-y_true * jnp.log(y_pred + eps) - (1 - y_true) * jnp.log(1 - y_pred + eps))
