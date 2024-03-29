from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DatasetConfig:
    """ Configuration class for the dataset

    :param datapath: path to the dataset
    :type datapath: str

    :param seq_len: sequence length
    :type seq_len: int

    :param norm_mode: normalization mode
    :type norm_mode: str

    :param initial_date: initial date to start the dataset
    :type initial_date: Optional[str]

    :param output_mode: output mode of the model (mean, distribution or discrete_grid)
    :type output_mode: str

    :param discrete_grid_levels: levels of the discrete grid (in percentage: e.g. [-9.e6, -2., 0.0, 2., 9.e6])
    :type discrete_grid_levels: Optional[List[float]]

    :param resolution: resolution of the dataset (30m, 1h, 4h, all)
    :type resolution: str

    :param tickers: list of tickers (e.g. ['btc_usd', 'eth_usd'])
    :type tickers: List[str]

    :param indicators: list of indicators (e.g. ['rsi', 'bb_upper', 'bb_lower', 'bb_middle', 'ema_2h', 'ema_4h'])
    :type indicators: Optional[List[str]]

    :param ohlc_only: whether to use only ohlc data (pad everything else with -1)
    :type ohlc_only: bool

    """
    datapath: str
    seq_len: int
    norm_mode: str
    output_mode: str
    resolution: str
    tickers: List[str]
    initial_date: Optional[str] = None
    indicators: Optional[List[str]] = None
    discrete_grid_levels: Optional[List[float]] = None
    ohlc_only: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> 'DatasetConfig':
        datapath = d['datapath']
        seq_len = d['seq_len']
        norm_mode = d['norm_mode']
        initial_date = d.get('initial_date')
        output_mode = d['output_mode']
        discrete_grid_levels = d.get('discrete_grid_levels')
        resolution = d['resolution']
        tickers = d['tickers']
        indicators = d.get('indicators')
        return cls(datapath, seq_len, norm_mode, initial_date, output_mode, discrete_grid_levels, resolution, tickers, indicators)


