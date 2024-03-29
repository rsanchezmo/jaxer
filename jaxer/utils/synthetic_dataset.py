import random
import time

import jax.numpy as jnp
import numpy as np
from jaxer.utils.plotter import plot_predictions
from jaxer.utils.dataset import Dataset
from jaxer.utils.normalizer import normalize
from jaxer.config.synthetic_dataset_config import SyntheticDatasetConfig
from typing import Optional


class SyntheticDataset:
    """ Synthetic dataset generator. It generates a window size historic data with sinusoidal signals with the same shape
    as the real dataset. Only historical prices are generated because volume and trades were not easily simulated.

    :param config: configuration for the synthetic dataset
    :type config: SyntheticDatasetConfig
    """

    def __init__(self, config: SyntheticDatasetConfig):
        self._config = config
        self._dtype = np.float32

        self._window_info = {
            'ticker': 'synthetic',
            'initial_date': None,
            'end_date': None,
            'output_mode': self._config.output_mode,
            'norm_mode': self._config.normalizer_mode,
            'window_size': self._config.window_size,
            'discrete_grid_levels': [],
            'resolution': 'delta',
            'precision': self._config.precision
        }

    def get_random_input(self):
        """ Get a random input for the synthetic dataset

        :return: random input for the synthetic dataset
        """
        return self._get_data(batch_size=1)[0]

    def generator(self, batch_size: int, seed: Optional[int] = None):
        """ Generator for the synthetic dataset

        :param batch_size: batch size
        :type batch_size: int

        :param seed: seed for reproducibility
        :type seed: int

        :return: generator for the synthetic dataset
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        while True:
            yield self._get_data(batch_size=batch_size)

    def _get_data(self, batch_size: int):
        """ Generate a batch of synthetic data

        :param batch_size: batch size
        :type batch_size: int

        :return: batch of synthetic data
        """
        # start_t = time.time()
        time_ = np.linspace(0, 1, self._config.window_size + 1, dtype=self._dtype)
        historical_timepoints = np.zeros((batch_size, self._config.window_size, 7), dtype=self._dtype)
        labels = np.zeros((batch_size, 1), dtype=self._dtype)
        extra_tokens = np.zeros((batch_size, 3), dtype=self._dtype)
        num_sinusoids = np.random.randint(1, self._config.num_sinusoids, size=batch_size)
        normalizers = np.zeros((batch_size, 12), dtype=self._dtype)
        normalizers[:, [1, 3, 5, 7, 9, 11]] = 1.0

        amplitudes = np.random.uniform(self._config.min_amplitude, self._config.max_amplitude, size=(batch_size, self._config.num_sinusoids))
        frequencies = np.random.uniform(self._config.min_frequency, self._config.max_frequency, size=(batch_size, self._config.num_sinusoids))
        phases = np.random.uniform(0, 2 * np.pi, size=(batch_size, self._config.num_sinusoids))

        if self._config.add_noise:
            noises = np.random.uniform(0, np.max(amplitudes) / 2, size=(batch_size, self._config.window_size + 1))

        tendencies = np.random.choice([-1, 1], size=(batch_size, 2))
        linear_tendencies = np.random.uniform(0, self._config.max_linear_trend, size=(batch_size,))
        exp_tendencies = np.random.uniform(0, self._config.max_exp_trend, size=(batch_size,))

        scalers_sum = np.random.uniform(0, 60_000, size=batch_size)
        scalers_prod = np.random.uniform(1, 10_000, size=batch_size)

        high_noise = np.random.uniform(0, 0.05, size=(batch_size, self._config.window_size))
        low_noise = np.random.uniform(-0.05, 0, size=(batch_size, self._config.window_size))

        for idx in range(batch_size):
            # init_t = time.time()
            # amplitude = np.random.uniform(self._config.min_amplitude,
            #                               self._config.max_amplitude, size=num_sinusoids[idx])
            # # weight them to sum 1
            # amplitude = self.softmax(amplitude) * self._config.max_amplitude
            # frequency = np.random.uniform(self._config.min_frequency,
            #                               self._config.max_frequency, size=num_sinusoids[idx])
            # phase = np.random.uniform(0, 2 * np.pi, size=num_sinusoids[idx])

            amplitude = amplitudes[idx, :num_sinusoids[idx]]
            frequency = frequencies[idx, :num_sinusoids[idx]]
            phase = phases[idx, :num_sinusoids[idx]]

            window_signal = np.zeros((self._config.window_size + 1,), dtype=self._dtype)
            for sinusoid in range(num_sinusoids[idx]):
                window_signal += amplitude[sinusoid] * np.sin(2 * np.pi * frequency[sinusoid] * time_ + phase[sinusoid])
            window_signal += 2 * self._config.max_amplitude  # to be positive

            if self._config.add_noise:
                window_signal += noises[idx]

            # add trends
            # linear
            tendency = tendencies[idx, 0]
            window_signal += tendency * time_ * linear_tendencies[idx]
            # exponential
            tendency = tendencies[idx, 1]
            window_signal += tendency * np.exp(time_) * exp_tendencies[idx]

            scaler_sum = scalers_sum[idx]
            scaler_prod = scalers_prod[idx]
            window_signal = window_signal * scaler_prod + scaler_sum

            historical_timepoints[idx, :, 0] = np.roll(window_signal[:-1], 1)
            historical_timepoints[idx, 0, 0] = window_signal[0]
            historical_timepoints[idx, :, 3] = window_signal[:-1]
            historical_timepoints[idx, :, 1] = (window_signal[:-1] +
                                                np.mean(window_signal[:-1]) * high_noise[idx])
            historical_timepoints[idx, :, 2] = (window_signal[:-1] +
                                                np.mean(window_signal[:-1]) * low_noise[idx])

            normalizers[idx, 0:4] = self._compute_normalizer(historical_timepoints[idx, :, 0:4],
                                                             self._config.normalizer_mode)

            historical_timepoints[idx, :, 0:4] = normalize(historical_timepoints[idx, :, 0:4],
                                                           normalizers[idx, 0:4][np.newaxis, ...])

            labels[idx] = window_signal[-1]
            labels[idx] = normalize(labels[idx], normalizers[idx, 0:4][np.newaxis, ...])

            price_std = np.std(historical_timepoints[idx, :, 0])
            extra_tokens[idx, 0] = price_std
            # end_t = time.time()
            # print(f"Time to generate sample inside batch: {1e3*(end_t - init_t):.2f}ms")

        extra_tokens = Dataset.encode_tokens(extra_tokens).astype(np.int16)

        # mask variables that cannot be computed
        historical_timepoints[:, :, 4:6] = -1
        historical_timepoints[:, :, -1] = time_[:-1]

        # end_t = time.time()
        # print(f"Time to generate batch: {1e3*(end_t - start_t):.2f}ms")

        dtype = jnp.float32 if self._config.precision == 'fp32' else jnp.float16

        return (jnp.array(historical_timepoints).astype(dtype), jnp.array(extra_tokens)), jnp.array(labels).astype(
            dtype), jnp.array(
            normalizers, dtype=dtype), self._window_info

    def _compute_normalizer(self, x: np.ndarray, normalizer_mode: str):
        if normalizer_mode == 'window_meanstd':
            mean_values = np.mean(x, axis=0)
            std_values = np.std(x, axis=0)

            return np.array([mean_values.max(), std_values.max(), 0, 1], dtype=self._dtype)

        if normalizer_mode == 'window_minmax':
            min_values = np.min(x)
            max_values = np.max(x)

            return np.array([0, 1, min_values, max_values], dtype=self._dtype)

        if normalizer_mode == 'window_mean':
            mean_values = np.mean(x, axis=0)
            return np.array([0, mean_values.max(), 0, 1],
                            dtype=self._dtype)  # mean scaling is just to divide by the mean

        raise ValueError('Not supported normalizer mode')

    @staticmethod
    def softmax(x: np.array):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


if __name__ == '__main__':

    dataset_config = SyntheticDatasetConfig(window_size=128,
                                            add_noise=True,
                                            normalizer_mode='window_mean',
                                            min_amplitude=.05,
                                            max_amplitude=.2,
                                            min_frequency=0.1,
                                            max_frequency=50,
                                            num_sinusoids=20,
                                            max_linear_trend=0.5,
                                            max_exp_trend=0.01,
                                            precision='fp32')

    dataset = SyntheticDataset(config=dataset_config)

    dataset_generator = dataset.generator(1, seed=0)
    for _ in range(20):
        start_t = time.time()
        x, y_true, normalizer, window_info = next(dataset_generator)
        end_t = time.time()
        print(f"Time to generate batch: {1e3 * (end_t - start_t):.2f}ms")
        y_pred = y_true
        plot_predictions(x=x, y_true=y_true, y_pred=y_pred, normalizer=normalizer, window_info=window_info,
                         denormalize_values=False)
