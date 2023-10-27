import torch
import pandas as pd
from typing import Tuple
import numpy as np
import torch.utils.data
import jax.numpy as jnp
from typing import Union


class Dataset(torch.utils.data.Dataset):
    def __init__(self, datapath: str, seq_len: int, norm_mode: str = 'window') -> None:
        """ Reads a csv file """
        self._datapath = datapath
        self._data = pd.read_csv(self._datapath)
        self._data['Date'] = pd.to_datetime(self._data['Date'])
        self._data.set_index('Date', inplace=True)
        self._data.sort_index(inplace=True)
        self._seq_len = seq_len

        if norm_mode not in ["window", "global", "none"]:
            raise ValueError("norm_mode must be one of ['window', 'global', 'none']")
        
        self._norm_mode = norm_mode
        if self._norm_mode == "global":
            self._global_normalizer = dict(
                price=dict(min_val=self._data[['Open', 'Close', 'High', 'Low', 'Adj Close']].min().min(), 
                           max_val=self._data[['Open', 'Close', 'High', 'Low', 'Adj Close']].max().max()),
                volume=dict(min_val=self._data['Volume'].min().min(), 
                            max_val=self._data['Volume'].max().max())
            )
        elif self._norm_mode == 'none':
            self._global_normalizer = dict(
                price=dict(min_val=0, max_val=1),
                volume=dict(min_val=0, max_val=1)
            )


    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray, dict]:
        start_idx = index
        end_idx = index + self._seq_len
        sequence_data_price = self._data.iloc[start_idx:end_idx][['Open', 'Close', 'High', 'Low', 'Adj Close']] 
        sequence_data_volume = self._data.iloc[start_idx:end_idx][['Volume']]

        if self._norm_mode == "window":
            min_vals = sequence_data_price.min().min()
            max_vals = sequence_data_price.max().max()
            normalizer_price = dict(min_val=min_vals, max_val=max_vals)
        else:
            normalizer_price = self._global_normalizer['price']
        sequence_data_price = normalize(sequence_data_price, normalizer_price)

        if self._norm_mode == "window":
            min_vals = sequence_data_volume.min().min()
            max_vals = sequence_data_volume.max().max()
            normalizer_volume = dict(min_val=min_vals, max_val=max_vals)
        else:
            normalizer_volume = self._global_normalizer['volume']
        sequence_data_volume = normalize(sequence_data_volume, normalizer_volume)

        # Extract the label
        label_idx = index + self._seq_len
        label = self._data.iloc[label_idx]['Close']
        label = normalize(label, normalizer_price)

        # Create a normalizer dict
        normalizer = dict(price=normalizer_price, volume=normalizer_volume)

        # Convert to NumPy arrays
        sequence_data = np.array(sequence_data_price.join(sequence_data_volume), dtype=np.float32)

        label = np.array([label], dtype=np.float32)
        return sequence_data, label, normalizer

    def __len__(self) -> int:
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
    min_ = normalizer['min_val']
    max_ = normalizer['max_val']
    return (data - min_) / (max_ - min_)


def denormalize(data: Union[np.ndarray, jnp.ndarray], normalizer: dict) -> Union[np.ndarray, jnp.ndarray]:
    """ Denormalizes the data """
    if isinstance(data, jnp.ndarray):
        data = np.asarray(data)  # BUG: if too large, the jnp array overflows
    min_ = normalizer['min_val']
    max_ = normalizer['max_val']
    return data * (max_ - min_) + min_


def jax_collate_fn(batch):
    # Convert PyTorch tensors to JAX arrays for both inputs and labels
    jax_inputs = [jnp.array(item[0]) for item in batch]
    jax_labels = [jnp.array(item[1]) for item in batch]
    norms = [item[2] for item in batch]

    # Stack them to create batched JAX arrays
    batched_jax_inputs = jnp.stack(jax_inputs)
    batched_jax_labels = jnp.stack(jax_labels)

    return batched_jax_inputs, batched_jax_labels, norms