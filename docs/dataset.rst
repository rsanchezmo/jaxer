.. _dataset:

Dataset
=======

Cryptocurrency data
~~~~~~~~~~~~~~~~~~~
The main idea was to obtain stock data, but the quickest way to get the transformer to work was to use cryptocurrency data
because the markets don't close; you don't have to deal with weekends or holidays.
Ultimately, it would have required some data mining of the data, which was not the main objective of this project.

After researching different platforms for historical cryptocurrency data, free limits, and time resolutions, the platform that best solved my needs was `Tiingo <https://www.tiingo.com/>`_.
I using the free `Tiingo Python API REST <https://www.tiingo.com/documentation/crypto>`_ to get data from different tickers and time resolutions.

After exploring its API, I was able to obtain data for different tickers (e.g., 'btc_usd', 'eth_usd') from January 2018
to January 2024 with a maximum resolution of 30 minutes. Higher resolution was impossible with the free API. However,
as an initial approximation, I decided not to invest money in obtaining data with higher resolution. This project purpose
was not to create a trading bot, but to exploit the capabilities of the transformer model.

The available information that Tiingo provides is:

- **Date**: the date of the time point
- **Low/High price**: the lowest and highest price of the asset in the time resolution
- **Close/Open price**: the price of the asset at the end and the beginning of the time resolution
- **Volume**:  total number of shares or contracts traded during a specified resolution
- **Notional Volume**: total value of the assets traded, rather than the number of units
- **Number of trades**: the number of trades done in the time resolution

An example of a time point of 1h resolution on *'btc_usd'* ticker is:

.. code-block:: json

    {
        "date": "2018-01-01T00:00:00+00:00",
        "open": 13792.20816155334,
        "high": 13821.764356890988,
        "low": 13513.67115362916,
        "close": 13602.87085911669,
        "volume": 2953.1744688868007,
        "volumeNotional": 40171650.92470767,
        "tradesDone": 15347.0
    },

You can find the data in the `data` folder. Data has been compressed to be uploaded to the repository. Available data is:

+------------+--------------+----------------+----------------+
| Resolution | Tickers      | Initial Date   | End Date       |
+============+==============+================+================+
| 4h         | 'btc_usd',   | 2018-01-01T00: | 2024-01-01T00: |
|            | 'eth_usd'    | 00:00+00:00    | 00:00+0000     |
+------------+--------------+----------------+----------------+
| 1h         | 'btc_usd',   | 2018-01-01T00: | 2024-01-01T00: |
|            | 'eth_usd'    | 00:00+00:00    | 00:00+0000     |
+------------+--------------+----------------+----------------+
| 30m        | 'btc_usd',   | 2018-01-01T00: | 2024-01-01T00: |
|            | 'eth_usd'    | 00:00+00:00    | 00:00+0000     |
+------------+--------------+----------------+----------------+

Additionally, some financial indicators such as EMA, RSI, or Bollinger Bands have been included. Indicators were computed
with code from another project, so code is not available here, but I introduced them inside each time point as additional field.
I am not a financial expert, and I am sure the model can capture its own indicators, but I thought it would be a good idea to
introduce them as a starting point to guide the model using a simpler architecture of the model.

- **Bollinger bands (BB)**: a technical analysis indicator measuring asset volatility with upper and lower bands around a simple moving average (20 values window).
- **Relative Strength Index (RSI)**: a momentum indicator comparing average gains and losses over a specified time period to determine potential overbought or oversold conditions (14 values window).
- **Exponential Moving Average (EMA)**: a type of moving average giving more weight to recent data points, commonly used to identify trends in different timeframes.

An example of 'btc_usd' ticker with 1h resolution is:

.. code-block:: json

    {
        "bb_upper": 42761.209557622904,
        "bb_middle": 42474.74199031915,
        "bb_lower": 42188.27442301539,
        "rsi": 0.5179402932082888,
        "ema_2h": 42403.00465622137,
        "ema_3h": 42395.20711448087,
        "ema_4h": 42401.48829729378,
        "ema_8h": 42434.884435038424,
        "ema_12h": 42445.37817184411,
        "ema_16h": 42440.6523757373
    }

As you will see in the :ref:`dataset_configuration`, you can choose to use them or not for training the model.

Normalization techniques
~~~~~~~~~~~~~~~~~~~~~~~~

Data must be normalized to avoid exploding gradients during training. Several normalization methods are implemented.

- A normalization across the entire dataset. Second, a window normalization. The second one seems more suitable to avoid losing too much resolution and also to ensure working over time and not become obsolete (BTC may surpass the current max BTC price or volume). However, this approach can introduce inconsistencies in data scaling across different windows, potentially affecting the model's performance and its ability to generalize to new data. For each approach, two normalizers are implemented: a min-max scaler and a standard scaler.

Dataset class
~~~~~~~~~~~~~
The dataset class is implemented in PyTorch due to the easiness of creating a dataloader. However, as we are working with Jax arrays, it was necessary to pass a function to the dataloader to map the Torch tensors to Jax.ndarrays.

The test set is not randomly computed across the entire dataset. For better generalization capabilities understanding, the test set is taken from the last dataset components; simulating real-world prediction.


.. _dataset_configuration:

Dataset configuration
~~~~~~~~~~~~~~~~~~~~~
TBC