import jax.numpy as jnp
import jax
from functools import partial
from jaxer.utils.normalizer import denormalize


@jax.jit
def r2(y_pred: jnp.ndarray, y_true: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """ R2 metric """
    SS_res = jnp.sum(jnp.square(y_true - y_pred), axis=0)
    SS_tot = jnp.sum(jnp.square(y_true - jnp.mean(y_true, axis=0)), axis=0)
    return 1 - SS_res / jnp.clip(SS_tot, a_min=eps)


@jax.jit
def huber_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """ Huber Loss """
    abs_error = jnp.abs(y_true - y_pred)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return jnp.mean(0.5 * jnp.square(quadratic) + delta * linear)


@jax.jit
def mae(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """ Mean Absolute Error """
    return jnp.mean(jnp.abs(y_true - y_pred))


@jax.jit
def mse(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """ Mean Squared Error """
    return 0.5 * jnp.mean(jnp.square(y_true - y_pred))


@jax.jit
def rmse(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """ Root Mean Squared Error """
    return jnp.sqrt(mse(y_true, y_pred))


@jax.jit
def gaussian_negative_log_likelihood(mean: jnp.ndarray, std: jnp.ndarray, targets: jnp.ndarray,
                                     eps: float = 1e-6) -> jnp.ndarray:
    first_term = jnp.log(jnp.maximum(2 * jnp.pi * jnp.square(std)))
    second_term = jnp.square((targets - mean)) / jnp.clip(jnp.square(std), a_min=eps)
    return 0.5 * jnp.mean(first_term + second_term)


@jax.jit
def mape(y_pred: jnp.ndarray, y_true: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """ Mean Absolute Percentage Error """
    return jnp.mean(ape(y_pred, y_true, eps))


@jax.jit
def ape(y_pred: jnp.ndarray, y_true: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """ Absolute Percentage Error """
    return jnp.abs(y_true - y_pred) / jnp.clip(jnp.abs(y_true), a_min=eps) * 100


@jax.jit
def binary_cross_entropy(y_pred: jnp.ndarray, y_true: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """ Binary Cross Entropy """
    return jnp.mean(-y_true * jnp.log(y_pred + eps) - (1 - y_true) * jnp.log(1 - y_pred + eps))


@jax.jit
def acc_dir(y_pred: jnp.ndarray, y_true: jnp.ndarray, last_price: jnp.ndarray) -> jnp.ndarray:
    """Direction accuracy metric

    :param y_pred: predicted values
    :type y_pred: jnp.ndarray

    :param y_true: true values
    :type y_true: jnp.ndarray

    :param last_price: last close price
    :type last_price: jnp.ndarray

    :return: direction accuracy (percentage)
    :rtype: jnp.ndarray
    """



    # jax.debug.print("🤯 {y_pred} {y_true} {last_price}", y_pred=y_pred, y_true=y_true, last_price=last_price)
    # jax.debug.print("🤯 {true_sign} {pred_sign}", true_sign=true_sign, pred_sign=pred_sign)

    return jnp.mean(acc_dir_raw(y_pred, y_true, last_price)) * 100


@jax.jit
def acc_dir_raw(y_pred: jnp.ndarray, y_true: jnp.ndarray, last_price: jnp.ndarray) -> jnp.ndarray:
    """Direction accuracy metric

    :param y_pred: predicted values
    :type y_pred: jnp.ndarray

    :param y_true: true values
    :type y_true: jnp.ndarray

    :param last_price: last close price
    :type last_price: jnp.ndarray

    :return: direction accuracy (percentage)
    :rtype: jnp.ndarray
    """
    true_sign = jnp.sign(y_true[:, 0] - last_price)  # y_true is (batch, 1) and last_price is (batch,)
    pred_sign = jnp.sign(y_pred[:, 0] - last_price)

    return true_sign == pred_sign


@jax.jit
def acc_dir_discrete(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Direction accuracy metric

    :param y_pred: predicted values
    :type y_pred: jnp.ndarray

    :param y_true: true values
    :type y_true: jnp.ndarray

    :return: direction accuracy (percentage)
    :rtype: jnp.ndarray
    """

    y_true = jnp.sign(y_true)
    y_pred = jnp.sign(y_pred)

    return jnp.mean(y_true == y_pred) * 100


@partial(jax.jit, static_argnames=('denormalize_values',))
def compute_metrics(x: jnp.ndarray, y_pred: jnp.ndarray, y_true: jnp.ndarray, normalizer: jnp.ndarray,
                    denormalize_values: bool = True):
    """ Compute metrics for a batch of data
    """

    x_hist = x[0]
    if denormalize_values:
        x_hist_ohlc_last = denormalize(x_hist[:, -1, 0:4], normalizer[:, 0:4])
        y_true = denormalize(y_true, normalizer[:, 0:4])
        y_pred = denormalize(y_pred, normalizer[:, 0:4])

    else:
        x_hist_ohlc_last = x_hist[:, -1, 0:4]

    mape_ = ape(y_pred, y_true)
    acc_dir_ = acc_dir_raw(y_pred, y_true, x_hist_ohlc_last[:, 3])

    metrics = {
        'acc_dir': acc_dir_,
        'mape': mape_,
    }

    return metrics
