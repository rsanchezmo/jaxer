from dataclasses import dataclass


@dataclass
class SyntheticDatasetConfig:
    """ Configuration class for a synthetic dataset

    :param window_size: size of the window
    :type window_size: int

    :param output_mode: output mode (mean, distribution, discrete_grid)
    :type output_mode: str

    :param normalizer_mode: normalizer mode (window_meanstd, window_minmax)
    :type normalizer_mode: str

    :param add_noise: add noise to the signal
    :type add_noise: bool

    :param min_amplitude: minimum amplitude for the sinusoids
    :type min_amplitude: float

    :param max_amplitude: maximum amplitude for the sinusoids
    :type max_amplitude: float

    :param min_frequency: minimum frequency for the sinusoids
    :type min_frequency: float

    :param max_frequency: maximum frequency for the sinusoids
    :type max_frequency: float
    """
    window_size: int
    output_mode: str = 'mean'
    normalizer_mode: str = 'window_minmax'
    add_noise: bool = False
    min_amplitude: float = 0.1
    max_amplitude: float = 1.0
    min_frequency: float = 0.5
    max_frequency: float = 30

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
