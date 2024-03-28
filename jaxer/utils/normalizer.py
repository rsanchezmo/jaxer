import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def denormalize(data: jnp.ndarray, normalizer: jnp.ndarray) -> jnp.ndarray:
    """ Denormalizes the data """

    min_ = normalizer[:, [2]]
    max_ = normalizer[:, [3]]
    temp_data = data * (max_ - min_) + min_

    mean_ = normalizer[:, [0]]
    std_ = normalizer[:, [1]]
    return temp_data * std_ + mean_


def normalize(data: np.ndarray, normalizer: np.ndarray) -> np.ndarray:
    """ Normalizes the data """

    min_ = normalizer[:, [2]]
    max_ = normalizer[:, [3]]
    temp_data = (data - min_) / (max_ - min_ + 1e-6)

    mean_ = normalizer[:, [0]]
    std_ = normalizer[:, [1]] + 1e-6
    return (temp_data - mean_) / std_
