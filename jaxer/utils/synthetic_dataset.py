import random
import jax.numpy as jnp
import numpy as np
from jaxer.utils.plotter import plot_predictions
from jaxer.utils.dataset import Dataset, normalize
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

        time_ = np.linspace(0, 1, self._config.window_size+1)
        historical_timepoints = np.zeros((batch_size, self._config.window_size, 7))
        labels = np.zeros((batch_size, 1))
        extra_tokens = np.zeros((batch_size, 3), dtype=np.float32)
        num_sinusoids = np.random.randint(1, 10, size=batch_size)
        normalizers = np.zeros((batch_size, 12))
        normalizers[:, [1, 3, 5, 7, 9, 11]] = 1.0

        for idx in range(batch_size):
            amplitude = np.random.uniform(self._config.min_amplitude,
                                          self._config.max_amplitude, size=num_sinusoids[idx])
            # weight them to sum 1
            amplitude = self.softmax(amplitude) * self._config.max_amplitude
            frequency = np.random.uniform(self._config.min_frequency,
                                          self._config.max_frequency, size=num_sinusoids[idx])
            phase = np.random.uniform(0, 2 * np.pi, size=num_sinusoids[idx])

            window_signal = np.zeros((self._config.window_size+1, ))
            for sinusoid in range(num_sinusoids[idx]):
                window_signal += amplitude[sinusoid] * np.sin(2 * np.pi * frequency[sinusoid] * time_ + phase[sinusoid])
            window_signal += 2 * self._config.max_amplitude  # to be positive
            # window_signal /= 2  # num_sinusoids[idx]  # to make everything a bit smaller

            if self._config.add_noise:
                window_signal += np.random.uniform(0, np.max(amplitude)/2, size=(self._config.window_size+1, ))

            historical_timepoints[idx, :, 0] = np.roll(window_signal[:-1], 1)
            historical_timepoints[idx, 0, 0] = window_signal[0]
            historical_timepoints[idx, :, 3] = window_signal[:-1]
            historical_timepoints[idx, :, 1] = window_signal[:-1] + np.random.uniform(0, np.max(amplitude),
                                                                                      size=(self._config.window_size,))
            historical_timepoints[idx, :, 2] = window_signal[:-1] + np.random.uniform(-np.max(amplitude), 0,
                                                                                      size=(self._config.window_size,))

            normalizers[idx, 0:4] = self._compute_normalizer(historical_timepoints[idx, :, 0:4],
                                                             self._config.normalizer_mode)

            labels[idx, :] = window_signal[-1]

            historical_timepoints[idx, :, 0:4] = normalize(historical_timepoints[idx, :, 0:4],
                                                           normalizers[idx, 0:4][np.newaxis, ...])
            labels[idx] = normalize(labels[idx], normalizers[idx, 0:4][np.newaxis, ...])

            price_std = np.std(historical_timepoints[idx, :, 0])
            extra_tokens[idx, 0] = price_std

        extra_tokens = Dataset.encode_tokens(extra_tokens).astype(np.int16)

        # mask variables that cannot be computed
        historical_timepoints[:, :, 4:6] = -1
        historical_timepoints[:, :, -1] = time_[:-1]

        window_info = {
            'ticker': 'synthetic',
            'initial_date': None,
            'end_date': None,
            'output_mode': self._config.output_mode,
            'norm_mode': self._config.normalizer_mode,
            'window_size': self._config.window_size,
            'discrete_grid_levels': [],
            'resolution': 'delta'
        }

        return (jnp.array(historical_timepoints), jnp.array(extra_tokens)), jnp.array(labels), jnp.array(normalizers), window_info

    @staticmethod
    def _compute_normalizer(x: np.ndarray, normalizer_mode: str):
        if normalizer_mode == 'window_meanstd':
            mean_values = np.mean(x, axis=0)
            std_values = np.std(x, axis=0)

            return np.array([mean_values.max(), std_values.max(), 0, 1])

        if normalizer_mode == 'window_minmax':
            min_values = np.min(x)
            max_values = np.max(x)

            return np.array([0, 1, min_values, max_values])

        raise ValueError('Not supported normalizer mode')

    @staticmethod
    def softmax(x: np.array):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


if __name__ == '__main__':

    dataset_config = SyntheticDatasetConfig(window_size=100,
                                            add_noise=False,
                                            normalizer_mode='window_meanstd',
                                            min_amplitude=.1,
                                            max_amplitude=.2,
                                            min_frequency=0.5,
                                            max_frequency=20)

    dataset = SyntheticDataset(config=dataset_config)

    dataset_generator = dataset.generator(1, seed=0)
    for _ in range(10):
        x, y_true, normalizer, window_info = next(dataset_generator)
        y_pred = y_true
        plot_predictions(x=x, y_true=y_true, y_pred=y_pred, normalizer=normalizer, window_info=window_info,
                         denormalize_values=True)

