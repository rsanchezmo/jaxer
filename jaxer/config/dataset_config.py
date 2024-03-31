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

    :param return_mode: whether to return the dataset
    :type return_mode: bool

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
    return_mode: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> 'DatasetConfig':
        datapath = d['datapath']
        seq_len = d['seq_len']
        norm_mode = d['norm_mode']
        initial_date = d.get('initial_date', None)
        output_mode = d['output_mode']
        discrete_grid_levels = d.get('discrete_grid_levels', None)
        resolution = d['resolution']
        tickers = d['tickers']
        indicators = d.get('indicators', None)
        ohlc_only = d.get('ohlc_only', False)
        return_mode = d.get('return_mode', False)
        return cls(datapath=datapath,
                   seq_len=seq_len,
                   norm_mode=norm_mode,
                   output_mode=output_mode,
                   resolution=resolution,
                   tickers=tickers,
                   initial_date=initial_date,
                   indicators=indicators,
                   discrete_grid_levels=discrete_grid_levels,
                   ohlc_only=ohlc_only,
                   return_mode=return_mode)
