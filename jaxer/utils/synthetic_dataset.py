import jax.numpy as jnp
import jax
import jax.random
from functools import partial


class SyntheticDataset:
    def __init__(self, window_size: int, seed: int = 0, hist_features_num: int = 4):
        self._key = jax.random.PRNGKey(seed)
        self._window_size = window_size
        self._hist_features_num = hist_features_num  # OHLC


    def get_batch(self, batch_size: int):
        self._key, sub_key = jax.random.split(self._key)
        yield self._get_data(batch_size, sub_key)

    @staticmethod
    @partial(jax.jit, static_argnames=('window_size', 'batch_size', 'hist_features_num'))
    def _get_data(batch_size: int, jax_random_key: jax.Array, window_size: int, hist_features_num: int,
                  min_amplitude: float = 0.1, max_amplitude: float = 1.0, min_frequency: float = 0.1,
                  max_frequency: float = 0.5):

        time = jnp.linspace(0, window_size, window_size)
        historical_timepoints = jnp.zeros((batch_size, window_size, hist_features_num))
        extra_tokens = jnp.zeros((batch_size, 3), dtype=jnp.int16)  # now we have 3 extra tokens (std_price, std_volume, std_trades)
        num_sinusoids = jax.random.randint(jax_random_key, (batch_size,), 1, 10)

        for idx in range(batch_size):
            jax_random_key, sub_key = jax.random.split(jax_random_key)
            amplitude = jax.random.uniform(sub_key, (num_sinusoids[idx].item(),), jnp.float32,
                                           min_amplitude, max_amplitude)
            frequency = jax.random.uniform(sub_key, (num_sinusoids[idx].item(),), jnp.float32,
                                           min_frequency, max_frequency)
            phase = jax.random.uniform(sub_key, (num_sinusoids[idx].item(),), jnp.float32, 0, 2 * jnp.pi)

            historical_timepoints[idx, :] += amplitude * jnp.sin(2 * jnp.pi * frequency * time + phase) + amplitude
            extra_tokens[idx, 0] = jnp.std(historical_timepoints[idx, :, 0:4])

            # TODO: must encode extra_tokens as in real world

        # mask variables that cannot be computed
        historical_timepoints[:, :, 4:] = -1

        return (historical_timepoints, extra_tokens),
