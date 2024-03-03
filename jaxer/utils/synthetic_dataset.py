import random
import jax.numpy as jnp
import numpy as np
from jaxer.utils.plotter import plot_predictions
from jaxer.utils.dataset import Dataset, normalize


class SyntheticDataset:
    def __init__(self, window_size: int, seed: int = 0, output_mode: str = 'mean',
                 normalizer_mode: str = 'window_minmax', add_noise: bool = False):
        np.random.seed(seed)
        random.seed(seed)
        self._window_size = window_size
        self._output_mode = output_mode
        self._add_noise = add_noise
        self._normalizer_mode = normalizer_mode

    def generator(self, batch_size: int):
        while True:
            yield self._get_data(batch_size=batch_size,
                                 window_size=self._window_size,
                                 output_mode=self._output_mode
                                 )

    def _get_data(self, batch_size: int, window_size: int,
                  min_amplitude: float = 0.1, max_amplitude: float = 1.0, min_frequency: float = 0.5,
                  max_frequency: float = 50, output_mode: str = 'mean'):

        time_ = np.linspace(0, 1, window_size+1)
        historical_timepoints = np.zeros((batch_size, window_size, 7))
        labels = np.zeros((batch_size, 1))
        extra_tokens = np.zeros((batch_size, 3), dtype=np.int16)
        num_sinusoids = np.random.randint(1, 10, size=batch_size)
        normalizers = np.zeros((batch_size, 12))
        normalizers[:, [1, 3, 5, 7, 9, 11]] = 1.0

        for idx in range(batch_size):
            amplitude = np.random.uniform(min_amplitude, max_amplitude, size=num_sinusoids[idx])
            # weight them to sum 1
            amplitude = self.softmax(amplitude) * max_amplitude
            frequency = np.random.uniform(min_frequency, max_frequency, size=num_sinusoids[idx])
            phase = np.random.uniform(0, 2 * np.pi, size=num_sinusoids[idx])

            window_signal = np.zeros((window_size+1, ))
            for sinusoid in range(num_sinusoids[idx]):
                window_signal += amplitude[sinusoid] * np.sin(2 * np.pi * frequency[sinusoid] * time_ + phase[sinusoid])
            window_signal += 2 * max_amplitude

            if self._add_noise:
                window_signal += np.random.uniform(0, np.max(amplitude)/2, size=(window_size+1, ))

            historical_timepoints[idx, :, 0] = np.roll(window_signal[:-1], 1)
            historical_timepoints[idx, 0, 0] = window_signal[0]
            historical_timepoints[idx, :, 3] = window_signal[:-1]
            historical_timepoints[idx, :, 1] = window_signal[:-1] + np.random.uniform(0, np.max(amplitude),
                                                                                      size=(window_size,))
            historical_timepoints[idx, :, 2] = window_signal[:-1] + np.random.uniform(-np.max(amplitude), 0,
                                                                                      size=(window_size,))

            normalizers[idx, 0:4] = self._compute_normalizer(historical_timepoints[idx, :, 0:4], self._normalizer_mode)

            labels[idx, :] = window_signal[-1]

            historical_timepoints[idx, :, 0:4] = normalize(historical_timepoints[idx, :, 0:4], normalizers[idx, 0:4][np.newaxis, ...])
            labels[idx] = normalize(labels[idx], normalizers[idx, 0:4][np.newaxis, ...])

            extra_tokens[idx, 0] = np.std(historical_timepoints[idx, :, 0:4])

        extra_tokens = Dataset.encode_tokens(extra_tokens)

        # mask variables that cannot be computed
        historical_timepoints[:, :, 4:6] = -1
        historical_timepoints[:, :, -1] = time_[:-1]

        window_info = {
            'ticker': 'synthetic',
            'initial_date': None,
            'end_date': None,
            'output_mode': output_mode,
            'norm_mode': self._normalizer_mode,
            'window_size': window_size,
            'discrete_grid_levels': [],
            'resolution': 'synthetic'
        }

        return ((jnp.array(historical_timepoints), jnp.array(extra_tokens)),
                jnp.array(labels), jnp.array(normalizers),
                window_info)

    @staticmethod
    def _compute_normalizer(x: np.ndarray, normalizer_mode: str):
        if normalizer_mode == 'window_meanstd':
            mean_values = np.mean(x)
            std_values = np.std(x)

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
    dataset = SyntheticDataset(window_size=100, seed=0, add_noise=False, normalizer_mode='window_minmax')

    dataset_generator = dataset.generator(1)
    for _ in range(10):
        x, y_true, normalizer, window_info = next(dataset_generator)
        y_pred = y_true
        plot_predictions(x=x, y_true=y_true, y_pred=y_pred, normalizer=normalizer, window_info=window_info,
                         denormalize_values=False)

