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

    :param return_mode: return mode. If true, then normalizer cant be window_mean
    :type return_mode: bool

    :param min_amplitude: minimum amplitude for the sinusoids
    :type min_amplitude: float

    :param max_amplitude: maximum amplitude for the sinusoids
    :type max_amplitude: float

    :param min_frequency: minimum frequency for the sinusoids
    :type min_frequency: float

    :param max_frequency: maximum frequency for the sinusoids
    :type max_frequency: float

    :param num_sinusoids: number of sinusoids to generate
    :type num_sinusoids: int

    :param max_linear_trend: maximum linear trend (for short term trends)
    :type max_linear_trend: float

    :param max_exp_trend: maximum exponential trend. Must be low as it grows exponentially (for long term trends)
    :type max_exp_trend: float

    :param precision: precision of the model (fp32 or fp16)
    :type precision: str

    :param close_only: whether to use only the close price
    :type close_only: bool
    """
    window_size: int
    return_mode: bool = False
    output_mode: str = 'mean'
    normalizer_mode: str = 'window_minmax'
    add_noise: bool = False
    min_amplitude: float = 0.1
    max_amplitude: float = 1.0
    min_frequency: float = 0.5
    max_frequency: float = 30
    num_sinusoids: int = 3
    max_linear_trend: float = 0.5
    max_exp_trend: float = 0.01
    precision: str = 'fp32'
    close_only: bool = False

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
