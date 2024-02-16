.. _dataset:

Dataset
=======

Cryptocurrency data
~~~~~~~~~~~~~~~~~~~

The dataset has been downloaded from `Tiingo <https://www.tiingo.com/>`_ using the free `Tiingo Python API REST <https://www.tiingo.com/documentation/crypto>`_. The dataset consists of the BTC/USD price from 2018-01-01 to 2024-01-01. Three time resolutions are available: 4h, 1h, 30m in order to evaluate the model performance on different time resolutions. I first downloaded one day time resolution from Yahoo Finance, but data was really scarce. Tiingo provides a JSON with timepoints and was the best option from my search on the internet for free data.

The available information is:

- Low/High price
- Close/Open price
- Volume
- Number of trades

The dataset class is implemented in PyTorch due to the easiness of creating a dataloader. However, as we are working with Jax arrays, it was necessary to pass a function to the dataloader to map the Torch tensors to Jax.ndarrays.

The test set is not randomly computed across the entire dataset. For better generalization capabilities understanding, the test set is taken from the last dataset components; simulating real-world prediction.

Normalization techniques
~~~~~~~~~~~~~~~~~~~~~~~~

Data must be normalized to avoid exploding gradients during training. Several normalization methods are implemented. First, a normalization across the entire dataset. Second, a window normalization. The second one seems more suitable to avoid losing too much resolution and also to ensure working over time and not become obsolete (BTC may surpass the current max BTC price or volume). However, this approach can introduce inconsistencies in data scaling across different windows, potentially affecting the model's performance and its ability to generalize to new data. For each approach, two normalizers are implemented: a min-max scaler and a standard scaler.

