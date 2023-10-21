import torch
import pandas as pd
from typing import Tuple
import numpy as np
import torch.utils.data


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
        sequence_data = self._data.iloc[start_idx:end_idx][['Open', 'Close', 'High', 'Low', 'Adj Close', 'Volume']]  # Add more columns as needed

        # Extract the label (next Close price)
        label_idx = index + self._seq_len
        label = self._data.iloc[label_idx]['Close']

        # Convert to NumPy arrays
        sequence_data = np.array(sequence_data, dtype=np.float32)
        label = np.array([label], dtype=np.float32)
        return sequence_data, label

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
    