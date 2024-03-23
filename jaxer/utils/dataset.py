import time
import functools
import jax
import torch
import pandas as pd
from typing import Tuple, Any
import numpy as np
import torch.utils.data
import jax.numpy as jnp
from typing import List, Optional
import os
import itertools
from jaxer.utils.logger import get_logger
from jaxer.config.dataset_config import DatasetConfig


class Dataset(torch.utils.data.Dataset):
    """ Finance dataset class for training jaxer (pytorch dataset)

    :param dataset_config: DatasetConfig object
    :type dataset_config: DatasetConfig

    :raises ValueError: if the norm_mode is not one of ['window_minmax', 'window_meanstd', 'global_minmax', 'global_meanstd', 'none']
    :raises ValueError: if the indicators are not in the dataset ['rsi', 'bb_upper', 'bb_lower', 'bb_middle']
    :raises ValueError: if the discrete_grid_levels are not provided and the output_mode is discrete_grid
    :raises ValueError: if the ticker is not in the dataset
    """

    OHLC = ['open', 'high', 'low', 'close']
    NORM_MODES = ['window_minmax', 'window_meanstd', 'window_mean', 'global_minmax', 'global_meanstd', 'none']
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

        self._resolution = dataset_config.resolution

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
            self._global_normalizers = []
            for ticker in range(len(self._data)):
                min_vals = self._data[ticker][Dataset.OHLC].min().min()
                max_vals = self._data[ticker][Dataset.OHLC].max().max()
                ohlc = np.array([[0, 1, min_vals, max_vals]])  # mean, std, min_vals, max_vals
                min_vals = self._data[ticker]['volume'].min().min()
                max_vals = self._data[ticker]['volume'].max().max()
                volume = np.array([[0, 1, min_vals, max_vals]])  # mean, std, min_vals, max_vals
                min_vals = self._data[ticker]['tradesDone'].min().min()
                max_vals = self._data[ticker]['tradesDone'].max().max()
                trades = np.array([[0, 1, min_vals, max_vals]])  # mean, std, min_vals, max_vals
                normalizer = np.concatenate((ohlc, volume, trades), axis=1)
                self._global_normalizers.append(normalizer)

        elif self._norm_mode == "global_meanstd":
            self._global_normalizers = []
            for ticker in range(len(self._data)):
                mean_vals = self._data[ticker][Dataset.OHLC].mean().mean()
                std_vals = self._data[ticker][Dataset.OHLC].std().max()
                ohlc = np.array([[mean_vals, std_vals, 0, 1]])
                mean_vals = self._data[ticker]['volume'].mean().mean()
                std_vals = self._data[ticker]['volume'].std().max()
                volume = np.array([[mean_vals, std_vals, 0, 1]])
                mean_vals = self._data[ticker]['tradesDone'].mean().mean()
                std_vals = self._data[ticker]['tradesDone'].std().max()
                trades = np.array([[mean_vals, std_vals, 0, 1]])
                normalizer = np.concatenate([ohlc, volume, trades], axis=1)
                self._global_normalizers.append(normalizer)
        elif self._norm_mode == 'none':
            self._global_normalizers = []
            for ticker in range(len(self._data)):
                ohlc = np.array([[0, 1, 0, 1]])
                volume = np.array([[0, 1, 0, 1]])
                trades = np.array([[0, 1, 0, 1]])
                normalizer = np.concatenate((ohlc, volume, trades), axis=1)
                self._global_normalizers.append(normalizer)

        self._logger.info(
            f"Dataset loaded with {len(self._data)} tickers and {len(self)} samples in total. "
            f"Each ticker has {self._data_len} samples.")

        self._sequence_data_time = np.arange(0, self._seq_len)[:, None]

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
        sequence_data = self._data[ticker_idx].iloc[start_idx:end_idx]
        sequence_data_values = sequence_data.loc[:, Dataset.OHLC + ['volume', 'tradesDone']].values

        sequence_data_time = (self._sequence_data_time + start_idx) / self._data_len[ticker_idx]

        sequence_data_indicators = None
        if self._indicators is not None:
            sequence_data_indicators = sequence_data.loc[:, self._indicators]

        # Compute normalizers
        normalizer = self._compute_normalizers(sequence_data_values, ticker_idx)

        # data normalization
        normalizer_ = np.concatenate([normalizer[:, 0:4], normalizer[:, 0:4], normalizer[:, 0:4],
                                      normalizer[:, 0:4], normalizer[:, 4:8], normalizer[:, 8:12]], axis=0)
        timepoints_tokens = self._normalize_all_at_once(sequence_data_values, normalizer_)

        std_ = timepoints_tokens.std(axis=0)
        std_[0] = std_[0:4].max()

        if sequence_data_indicators is not None:
            sequence_data_indicators = self._normalize_indicators(sequence_data_indicators, normalizer[:, 0:4])

        # Extract the labels
        label_idx = index + self._seq_len
        label = self._data[ticker_idx].iloc[label_idx]['close']

        if self.output_mode == 'mean' or self.output_mode == 'distribution':
            label = np.array([label], dtype=np.float32)
            label = normalize(label, normalizer[:, 0:4])[0]
        else:
            prev_close_price = self._data[ticker_idx].iloc[label_idx - 1]['close']
            percent = (label - prev_close_price) / prev_close_price * 100
            label = np.digitize(percent, self._discrete_grid_levels) - 1
            # to one-hot
            label = np.eye(len(self._discrete_grid_levels) - 1)[label]
            label = np.array(label, dtype=np.float32)

        # Convert to NumPy arrays
        timepoints_tokens = np.concatenate((timepoints_tokens, sequence_data_time), axis=1, dtype=np.float32)
        if sequence_data_indicators is not None:
            timepoints_tokens = np.concatenate((timepoints_tokens, sequence_data_indicators), axis=1, dtype=np.float32)

        # get the initial timestep
        timestep = self._data[ticker_idx].iloc[start_idx].name

        extra_tokens = std_[[0, 4, 5]]

        if self._norm_mode.__contains__('global'):
            extra_tokens *= 5  # to increase resolution

        extra_tokens = self.encode_tokens(extra_tokens)

        window_info = {
            'ticker': self._tickers[ticker_idx],
            'initial_date': timestep,
            'end_date': self._data[ticker_idx].iloc[label_idx - 1].name,
            'output_mode': self.output_mode,
            'norm_mode': self._norm_mode,
            'discrete_grid_levels': self._discrete_grid_levels,
            'resolution': self._resolution,
            'window_size': self._seq_len
        }

        return timepoints_tokens, extra_tokens, label, normalizer, window_info

    def __len__(self) -> int:
        return sum(self._data_len)

    @staticmethod
    def _normalize_all_at_once(all_data: np.ndarray,
                               normalizer: np.ndarray) -> np.ndarray:
        """ Normalizes the data at once """
        tmp = (all_data - normalizer[:, 2]) / (normalizer[:, 3] - normalizer[:, 2] + 1e-6)
        return (tmp - normalizer[:, 0]) / (normalizer[:, 1] + 1e-6)

    @staticmethod
    def encode_tokens(tokens: np.ndarray) -> np.ndarray:
        """ Encodes the tokens into integer (tokens are expected to be on [0, 1])

        :param tokens: tokens to encode
        :type tokens: np.ndarray

        :return: encoded tokens
        :rtype: np.ndarray
        """
        tokens = np.round(tokens * 100)  # tokens are positive
        tokens = np.minimum(tokens, 100)
        return tokens

    def _normalize_indicators(self, sequence_data_indicators, normalizer_price):
        """ Normalizes the indicators that require to be normalized with the price """
        for indicator in self._indicators:
            if indicator in Dataset.INDICATORS_TO_NORMALIZE:
                sequence_data_indicators[indicator] = normalize(sequence_data_indicators[indicator], normalizer_price)
        return sequence_data_indicators

    def _compute_normalizers(self, sequence_data_values: np.ndarray, ticker_idx: int):
        if self._norm_mode == "window_minmax":
            min_values = sequence_data_values.min(axis=0)
            max_values = sequence_data_values.max(axis=0)

            return np.array([
                [0, 1, min_values[0:4].min(), max_values[0:4].max(),
                 0, 1, min_values[4], max_values[4],
                 0, 1, min_values[5], max_values[5]]
            ])

        if self._norm_mode == "window_meanstd":
            mean_values = sequence_data_values.mean(axis=0)
            std_values = sequence_data_values.std(axis=0)

            return np.array([
                [mean_values[0:4].max(), std_values[0:4].max(), 0, 1,
                 mean_values[4], std_values[4], 0, 1,
                 mean_values[5], std_values[5], 0, 1]
            ])

        if self._norm_mode =="window_mean":
            mean_values = sequence_data_values.mean(axis=0)
            return np.array([
                [0, mean_values[0:4].max(), 0, 1,  # mean scaling is just to divide by the mean (as if it was std)
                 0, mean_values[4], 0, 1,
                 0, mean_values[5], 0, 1]
            ])

        return self._global_normalizers[ticker_idx]

    def get_train_test_split(self,
                             test_size: float = 0.1,
                             test_tickers: Optional[List[str]] = None) \
            -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """ Returns a train and test set from the dataset

        :param test_size: test size
        :type test_size: float

        :param test_tickers: tickers to include in the test set. If None, all tickers are included
        :type test_tickers: Optional[List[str]]

        :return: train and test dataset
        :rtype: Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]
        """

        # Split the dataset ranges with itertools.chain
        train_ranges = []
        test_ranges = []

        test_samples_btc = int(self._data_len[0] * test_size)
        train_samples_btc = self._data_len[0] - test_samples_btc
        initial_date_btc = self._data[0].index[0]

        for ticker in range(len(self._data_len)):
            if self._data[ticker].index[0] != initial_date_btc:
                btc_transition_date = self._data[0].index[train_samples_btc]
                train_samples = self._data[ticker].index.get_loc(btc_transition_date)
            else:
                train_samples = train_samples_btc

            # get the index of self._data[ticker].index for the date that is the limit

            if ticker == 0:  # first ticker (btc_usd always)
                train_ranges.append(range(0, train_samples))

                if test_tickers is not None and self._tickers[ticker] not in test_tickers:
                    continue

                test_ranges.append(range(train_samples, self._data_len[ticker]))
            else:
                train_ranges.append(
                    range(self._unrolled_len[ticker - 1], self._unrolled_len[ticker - 1] + train_samples))

                if test_tickers is not None and self._tickers[ticker] not in test_tickers:
                    continue

                test_ranges.append(range(self._unrolled_len[ticker - 1] + train_samples,
                                         self._unrolled_len[ticker - 1] + self._data_len[ticker]))

        train_ranges = itertools.chain(*train_ranges)
        test_ranges = itertools.chain(*test_ranges)

        train_dataset = torch.utils.data.Subset(self, list(train_ranges))
        test_dataset = torch.utils.data.Subset(self, list(test_ranges))

        self._logger.info(f"Train dataset has {len(train_dataset)} samples and test "
                          f"dataset has {len(test_dataset)} samples")

        self._logger.info(f"Training from {initial_date_btc} to {self._data[0].index[train_samples_btc]} and testing "
                          f"since then to {self._data[0].index[-1]}")

        return train_dataset, test_dataset


def normalize(data: np.ndarray, normalizer: np.ndarray) -> np.ndarray:
    """ Normalizes the data """

    min_ = normalizer[:, [2]]
    max_ = normalizer[:, [3]]
    temp_data = (data - min_) / (max_ - min_ + 1e-6)

    mean_ = normalizer[:, [0]]
    std_ = normalizer[:, [1]] + 1e-6
    return (temp_data - mean_) / std_


@jax.jit
def denormalize(data: jnp.ndarray, normalizer: jnp.ndarray) -> jnp.ndarray:
    """ Denormalizes the data """

    min_ = normalizer[:, [2]]
    max_ = normalizer[:, [3]]
    temp_data = data * (max_ - min_) + min_

    mean_ = normalizer[:, [0]]
    std_ = normalizer[:, [1]]
    return temp_data * std_ + mean_


def jax_collate_fn(batch: List[np.ndarray]) -> Tuple:
    """ Collate function for the jax dataset

    :param batch: batch of data
    :type batch: List[jnp.ndarray]

    :return: batched data (sequence_tokens, extra_tokens), labels, norms, window_info
    :rtype: Tuple
    """
    sequence_tokens, extra_tokens, labels, norms, window_info = zip(*batch)

    batched_jax_sequence_tokens = jnp.stack(sequence_tokens)
    batched_jax_extra_tokens = jnp.stack(extra_tokens)
    batched_jax_labels = jnp.stack(labels)
    batched_norms = jnp.concatenate(norms, axis=0)

    return (batched_jax_sequence_tokens, batched_jax_extra_tokens), batched_jax_labels, batched_norms, window_info
