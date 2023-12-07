import jax.numpy as jnp
import jax


@jax.jit
def r2(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """ R2 metric """
    SS_res = jnp.sum(jnp.square(y_true - y_pred), axis=0)
    SS_tot = jnp.sum(jnp.square(y_true - jnp.mean(y_true, axis=0)), axis=0)
    return 1 - SS_res / (SS_tot + 1.e-8)

@jax.jit
def mae(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """ Mean Absolute Error """
    return jnp.mean(jnp.abs(y_true - y_pred))

@jax.jit
def mse(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """ Mean Squared Error """
    return jnp.mean(jnp.square(y_true - y_pred))

@jax.jit
def rmse(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """ Root Mean Squared Error """
    return jnp.sqrt(mse(y_true, y_pred))

@jax.jit
def gaussian_negative_log_likelihood(mean: jnp.ndarray, variance: jnp.ndarray, targets: jnp.ndarray):
    mean_coef = 0.2
    var_coef = 1. - mean_coef
    return jnp.mean(mean_coef * jnp.log(2 * jnp.pi * variance) + var_coef * ((targets - mean) ** 2) / (variance + 1.e-8))


@jax.jit
def mape(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """ Mean Absolute Percentage Error """
    return jnp.mean(jnp.abs((y_true - y_pred) / (y_true + 1.e-8))) * 100
