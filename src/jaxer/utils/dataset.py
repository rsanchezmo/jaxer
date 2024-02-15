import torch
import pandas as pd
from typing import Tuple, Any
import numpy as np
import torch.utils.data
import jax.numpy as jnp
from typing import Union, Optional, List
from dataclasses import dataclass
import os
import itertools
from jaxer.utils.logger import get_logger


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

    :param resolution: resolution of the dataset (30m, 1h, 4h)
    :type resolution: str

    :param tickers: list of tickers (e.g. ['btc_usd', 'eth_usd'])
    :type tickers: List[str]

    :param indicators: list of indicators (e.g. ['rsi', 'bb_upper', 'bb_lower', 'bb_middle', 'ema_2h', 'ema_4h'])
    :type indicators: Optional[List[str]]

    """
    datapath: str
    seq_len: int
    norm_mode: str
    initial_date: Optional[str]
    output_mode: str
    discrete_grid_levels: Optional[List[float]]
    resolution: str
    tickers: List[str]
    indicators: Optional[List[str]]


class Dataset(torch.utils.data.Dataset):
    """ Finance dataset class for training jaxer (pytorch dataset)

    :param dataset_config: DatasetConfig object
    :type dataset_config: DatasetConfig
    """

    OHLC = ['open', 'high', 'low', 'close']
    NORM_MODES = ['window_minmax', 'window_meanstd', 'global_minmax', 'global_meanstd', 'none']
    INDICATORS = ['rsi', 'bb_upper', 'bb_lower', 'bb_middle']
    INDICATORS_TO_NORMALIZE = ['bb_upper', 'bb_lower', 'bb_middle']

    def __init__(self, dataset_config: DatasetConfig) -> None:
        """ Reads a csv file """
        self._datapath = os.path.join(dataset_config.datapath, dataset_config.resolution)

        self._data = []
        self._data_len = []
        self._unrolled_len = []
        cum_len = 0

        # load tickers
        for ticker in dataset_config.tickers:
            data = pd.read_json(os.path.join(self._datapath, f"{ticker}_{dataset_config.resolution}.json"))
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            data.sort_index(inplace=True)

            if dataset_config.initial_date is not None:
                data = data[data.index >= dataset_config.initial_date]

            self._data.append(data)
            self._data_len.append(len(data) - dataset_config.seq_len)

            self._unrolled_len.append(cum_len + self._data_len[-1])
            cum_len += self._data_len[-1]

        self._seq_len = dataset_config.seq_len

        self._logger = get_logger()

        self.output_mode = dataset_config.output_mode
        self._discrete_grid_levels = dataset_config.discrete_grid_levels
        self._tickers = dataset_config.tickers
        self._indicators = dataset_config.indicators
        if self._indicators is not None:
            self._indicators = [indicator for indicator in self._indicators if indicator in Dataset.INDICATORS]

            # check if there is some indicator that is not in the dataset
            for indicator in self._indicators:
                for ticker in range(len(self._data)):
                    if indicator not in self._data[ticker].columns:
                        raise ValueError(f"Indicator {indicator} is not in the dataset")

        if dataset_config.discrete_grid_levels is None and dataset_config.output_mode == 'discrete_grid':
            raise ValueError('discrete_grid_levels must be provided if output_mode is discrete_grid')

        if dataset_config.norm_mode not in Dataset.NORM_MODES:
            raise ValueError(f"norm_mode must be one of {Dataset.NORM_MODES}")

        self._norm_mode = dataset_config.norm_mode

        if self._norm_mode == "global_minmax":
            # normalizing with the first ticker (for now)
            self._global_normalizer = dict(
                price=dict(min_val=self._data[0][Dataset.OHLC].min().min(),
                           max_val=self._data[0][Dataset.OHLC].max().max(),
                           mode="minmax"),
                volume=dict(min_val=self._data[0]['volume'].min().min(),
                            max_val=self._data[0]['volume'].max().max(),
                            mode="minmax"),
                trades=dict(min_val=self._data[0]['tradesDone'].min().min(),
                            max_val=self._data[0]['tradesDone'].max().max(),
                            mode="minmax")
            )
        elif self._norm_mode == "global_meanstd":
            self._global_normalizer = dict(
                price=dict(mean_val=self._data[0][Dataset.OHLC].mean().max(),
                           std_val=self._data[0][Dataset.OHLC].std().max(),
                           mode="meanstd"),
                volume=dict(mean_val=self._data[0]['volume'].mean().max(),
                            std_val=self._data[0]['volume'].std().max(),
                            mode="meanstd"),
                trades=dict(mean_val=self._data[0]['tradesDone'].mean().max(),
                            std_val=self._data[0]['tradesDone'].std().max(),
                            mode="meanstd")
            )
        elif self._norm_mode == 'none':
            self._global_normalizer = dict(
                price=dict(min_val=0, max_val=1, mode="minmax"),
                volume=dict(min_val=0, max_val=1, mode="minmax"),
                trades=dict(min_val=0, max_val=1, mode="minmax")
            )

        self._logger.info(
            f"Dataset loaded with {len(self._data)} tickers and {len(self)} samples in total. "
            f"Each ticker has {self._data_len} samples.")

    def _map_idx(self, index: int) -> Tuple[int, int]:
        """ Maps the index to the corresponding ticker and index using self._unrolled_len """
        for idx in range(len(self._unrolled_len)):
            if index < self._unrolled_len[idx]:
                if idx == 0:
                    return index, idx
                return index - self._unrolled_len[idx - 1], idx

        raise ValueError("Index out of range")

    def get_random_input(self):
        """ Returns a random input from the dataset

        :return: sequence_tokens, extra_tokens
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """
        idx = np.random.randint(0, len(self))
        data = self[idx]
        sequence_tokens = jnp.expand_dims(data[0], axis=0)
        extra_tokens = jnp.expand_dims(data[1], axis=0)
        return sequence_tokens, extra_tokens

    def __getitem__(self, index):
        index, ticker_idx = self._map_idx(index)
        start_idx = index
        end_idx = index + self._seq_len

        # retrieve the sequence data
        sequence_data_price = self._data[ticker_idx].iloc[start_idx:end_idx][Dataset.OHLC]
        sequence_data_volume = self._data[ticker_idx].iloc[start_idx:end_idx][['volume']]
        sequence_data_trades = self._data[ticker_idx].iloc[start_idx:end_idx][['tradesDone']]
        sequence_data_time = np.array([idx / len(self._data[ticker_idx]) for idx in range(start_idx, end_idx)])[:, None]

        sequence_data_indicators = None
        if self._indicators is not None:
            sequence_data_indicators = self._data[ticker_idx].iloc[start_idx:end_idx][self._indicators]

        # Compute normalizers
        normalizer_price, normalizer_volume, normalizer_trades = self._compute_normalizers(sequence_data_price,
                                                                                           sequence_data_volume,
                                                                                           sequence_data_trades)

        # normalize data
        sequence_data_price = normalize(sequence_data_price, normalizer_price)
        sequence_data_price_std = sequence_data_price.std().std() * 100
        sequence_data_volume = normalize(sequence_data_volume, normalizer_volume)
        sequence_data_volume_std = sequence_data_volume.std()['volume'] * 100
        sequence_data_trades = normalize(sequence_data_trades, normalizer_trades)
        sequence_data_trades_std = sequence_data_trades.std()['tradesDone'] * 100

        if sequence_data_indicators is not None:
            sequence_data_indicators = self._normalize_indicators(sequence_data_indicators, normalizer_price)

        # Extract the labels
        label_idx = index + self._seq_len
        label = self._data[ticker_idx].iloc[label_idx]['close']

        if self.output_mode == 'mean' or self.output_mode == 'distribution':
            label = normalize(label, normalizer_price)
            label = np.array([label], dtype=np.float32)

        else:
            prev_close_price = self._data[ticker_idx].iloc[label_idx - 1]['close']
            percent = (label - prev_close_price) / prev_close_price * 100
            label = np.digitize(percent, self._discrete_grid_levels) - 1
            # to one-hot
            label = np.eye(len(self._discrete_grid_levels) - 1)[label]

        # Create a normalizer dict
        normalizer = dict(price=normalizer_price, volume=normalizer_volume, trades=normalizer_trades)

        # Convert to NumPy arrays
        to_concatenate = [sequence_data_price, sequence_data_volume, sequence_data_trades, sequence_data_time]
        if sequence_data_indicators is not None:
            to_concatenate.append(sequence_data_indicators)
        timepoints_tokens = np.concatenate(to_concatenate, axis=1, dtype=np.float32)

        # get the initial timestep
        timestep = self._data[ticker_idx].iloc[start_idx].name

        extra_tokens = np.array([sequence_data_price_std, sequence_data_volume_std, sequence_data_trades_std],
                                dtype=np.float32)

        return jnp.array(timepoints_tokens), jnp.array(extra_tokens), jnp.array(label), normalizer, \
            timestep.strftime("%Y-%m-%d")

    def __len__(self) -> int:
        return sum(self._data_len)

    def _normalize_indicators(self, sequence_data_indicators, normalizer_price):
        """ Normalizes the indicators that require to be normalized with the price """
        for indicator in self._indicators:
            if indicator in Dataset.INDICATORS_TO_NORMALIZE:
                sequence_data_indicators[indicator] = normalize(sequence_data_indicators[indicator], normalizer_price)
        return sequence_data_indicators

    def _compute_normalizers(self, sequence_data_price, sequence_data_volume, sequence_data_trades):
        if self._norm_mode == "window_minmax":
            min_vals = sequence_data_price.min().min()
            max_vals = sequence_data_price.max().max()
            normalizer_price = dict(min_val=min_vals, max_val=max_vals, mode="minmax")

            min_vals = sequence_data_volume.min().min()
            max_vals = sequence_data_volume.max().max()
            normalizer_volume = dict(min_val=min_vals, max_val=max_vals, mode="minmax")

            min_vals = sequence_data_trades.min().min()
            max_vals = sequence_data_trades.max().max()
            normalizer_trades = dict(min_val=min_vals, max_val=max_vals, mode="minmax")

        elif self._norm_mode == "window_meanstd":
            mean_vals = sequence_data_price.mean().max()
            std_vals = sequence_data_price.std().max()
            normalizer_price = dict(mean_val=mean_vals, std_val=std_vals, mode="meanstd")

            mean_vals = sequence_data_volume.mean().max()
            std_vals = sequence_data_volume.std().max()
            normalizer_volume = dict(mean_val=mean_vals, std_val=std_vals, mode="meanstd")

            mean_vals = sequence_data_trades.mean().max()
            std_vals = sequence_data_trades.std().max()
            normalizer_trades = dict(mean_val=mean_vals, std_val=std_vals, mode="meanstd")

        else:
            normalizer_price = self._global_normalizer['price']
            normalizer_volume = self._global_normalizer['volume']
            normalizer_trades = self._global_normalizer['trades']

        return normalizer_price, normalizer_volume, normalizer_trades

    def get_train_test_split(self, test_size: float = 0.1) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """ Returns a train and test set from the dataset

        :param test_size: test size
        :type test_size: float

        :return: train and test dataset
        :rtype: Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]
        """

        # Split the dataset ranges with itertools.chain
        train_ranges = []
        test_ranges = []
        for ticker in range(len(self._data_len)):
            test_samples = int(self._data_len[ticker] * test_size)
            train_samples = self._data_len[ticker] - test_samples

            if ticker == 0:
                train_ranges.append(range(0, train_samples))
                test_ranges.append(range(train_samples, self._data_len[ticker]))
            else:
                train_ranges.append(
                    range(self._unrolled_len[ticker - 1], self._unrolled_len[ticker - 1] + train_samples))
                test_ranges.append(range(self._unrolled_len[ticker - 1] + train_samples,
                                         self._unrolled_len[ticker - 1] + self._data_len[ticker]))

        train_ranges = itertools.chain(*train_ranges)
        test_ranges = itertools.chain(*test_ranges)

        train_dataset = torch.utils.data.Subset(self, list(train_ranges))
        test_dataset = torch.utils.data.Subset(self, list(test_ranges))

        return train_dataset, test_dataset


def _normalize_minmax(data: Union[np.ndarray, jnp.ndarray], normalizer) -> Union[np.ndarray, jnp.ndarray]:
    """ Normalizes the data """
    min_ = normalizer['min_val']
    max_ = normalizer['max_val']
    return (data - min_) / (max_ - min_ + 1e-6)


def _normalize_meanstd(data: Union[np.ndarray, jnp.ndarray], normalizer) -> Union[np.ndarray, jnp.ndarray]:
    """ Normalizes the data """
    mean_ = normalizer['mean_val']
    std_ = normalizer['std_val'] + 1e-6
    return (data - mean_) / std_


def _denormalize_minmax(data: Union[np.ndarray, jnp.ndarray], normalizer) -> Union[np.ndarray, jnp.ndarray]:
    """ Denormalizes the data """
    min_ = normalizer['min_val']
    max_ = normalizer['max_val']
    return data * (max_ - min_) + min_


def _denormalize_meanstd(data: Union[np.ndarray, jnp.ndarray], normalizer) -> Union[np.ndarray, jnp.ndarray]:
    """ Denormalizes the data """
    mean_ = normalizer['mean_val']
    std_ = normalizer['std_val']
    return data * std_ + mean_


def normalize(data: Union[np.ndarray, jnp.ndarray], normalizer: dict) -> Union[np.ndarray, jnp.ndarray]:
    """ Normalizes the data """

    mode = normalizer.get('mode', None)
    if mode is None:
        raise ValueError("normalizer must contain a 'mode' key")

    if mode == "minmax":
        return _normalize_minmax(data, normalizer)

    if mode == "meanstd":
        return _normalize_meanstd(data, normalizer)

    raise ValueError("mode must be one of ['minmax', 'meanstd']")


def denormalize_wrapper(data: Union[np.ndarray, jnp.ndarray], normalizer: dict, component: str = "price") -> Union[
    np.ndarray, jnp.ndarray]:
    return denormalize(data, normalizer[component])


def denormalize(data: Union[np.ndarray, jnp.ndarray], normalizer: dict, ) -> Union[np.ndarray, jnp.ndarray]:
    """ Denormalizes the data """
    if isinstance(data, jnp.ndarray):
        data = np.asarray(data)  # BUG: if too large, the jnp array overflows

    mode = normalizer.get('mode', None)
    if mode is None:
        raise ValueError("normalizer must contain a 'mode' key")

    if mode == "minmax":
        return _denormalize_minmax(data, normalizer)
    if mode == "meanstd":
        return _denormalize_meanstd(data, normalizer)

    raise ValueError("mode must be one of ['minmax', 'meanstd']")


def jax_collate_fn(batch: List[np.ndarray]) -> Any:
    """ Collate function for the jax dataset

    :param batch: batch of data
    :type batch: np.ndarray

    :return: batched data (sequence_tokens, extra_tokens), labels, norms, timesteps
    :rtype: Tuple
    """
    sequence_tokens, extra_tokens, labels, norms, timesteps = zip(*batch)

    batched_jax_sequence_tokens = jnp.stack(sequence_tokens)
    batched_jax_extra_tokens = jnp.stack(extra_tokens)
    batched_jax_labels = jnp.stack(labels)

    return (batched_jax_sequence_tokens, batched_jax_extra_tokens), batched_jax_labels, norms, timesteps
