import torch
import pandas as pd
from typing import Tuple
import numpy as np
import torch.utils.data
import jax.numpy as jnp
from typing import Union


class Dataset(torch.utils.data.Dataset):
    def __init__(self, datapath: str, seq_len: int) -> None:
        """ Reads a csv file """
        self._datapath = datapath
        self._data = pd.read_csv(self._datapath)
        self._data['Date'] = pd.to_datetime(self._data['Date'])
        self._data.set_index('Date', inplace=True)
        self._data.sort_index(inplace=True)
        self._seq_len = seq_len


    def __getitem__(self, index):
        start_idx = index
        end_idx = index + self._seq_len
        sequence_data = self._data.iloc[start_idx:end_idx][['Open', 'Close', 'High', 'Low', 'Adj Close', 'Volume']] 

        # Window Min-Max scaling for each feature
        min_vals = sequence_data.min()
        max_vals = sequence_data.max()
        sequence_data = (sequence_data - min_vals) / (max_vals - min_vals)

        # Extract the label (next Close price)
        label_idx = index + self._seq_len
        label = self._data.iloc[label_idx]['Close']

        min_close = min_vals['Close']
        max_close = max_vals['Close']
        normalizer = dict(min_close=min_close, max_close=max_close)

        label = normalize(label, normalizer)

        # Convert to NumPy arrays
        sequence_data = np.array(sequence_data, dtype=np.float32)
        label = np.array([label], dtype=np.float32)
        return sequence_data, label, normalizer

    def __len__(self):
        return len(self._data) - self._seq_len
    
    def get_train_test_split(self, test_size: float = 0.1) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """ Returns a train and test dataset"""
        total_samples = len(self)
        test_samples = int(total_samples * test_size)
        train_samples = total_samples - test_samples

        train_dataset = torch.utils.data.Subset(self, range(0, train_samples))
        test_dataset = torch.utils.data.Subset(self, range(train_samples, total_samples))

        return train_dataset, test_dataset
    

def normalize(data: Union[np.ndarray, jnp.ndarray], normalizer: dict) -> Union[np.ndarray, jnp.ndarray]:
    """ Normalizes the data """
    return (data - normalizer['min_close']) / (normalizer['max_close'] - normalizer['min_close'])

def denormalize(data: Union[np.ndarray, jnp.ndarray], normalizer: dict) -> Union[np.ndarray, jnp.ndarray]:
    """ Denormalizes the data """
    return data * (normalizer['max_close'] - normalizer['min_close']) + normalizer['min_close']