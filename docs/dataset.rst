.. _dataset:

Dataset
=======

Cryptocurrency data
~~~~~~~~~~~~~~~~~~~
The main goal was to obtain stock data, but the **quickest way** to get the **transformer to work** was to use **cryptocurrency** data
because crypto market is always open so there is no need to provide special care to deal with day open and close, weekends or holidays.
Ultimately, it would have required some **data mining** of the data, which was not the main objective of this project.

After researching different platforms for historical cryptocurrency data, free limits, and time resolutions, the platform that best solved my needs was `Tiingo <https://www.tiingo.com/>`_.
I used the free `Tiingo Python API REST <https://www.tiingo.com/documentation/crypto>`_ to get data from different tickers and time resolutions.

After exploring its API, I was able to obtain data for **different tickers** (e.g., 'btc_usd', 'eth_usd') from January 2018
to January 2024 with a **maximum resolution of 30 minutes**. Higher resolution was impossible with the free API. However,
as an initial approximation, I decided not to invest money in obtaining data with higher resolution. This project purpose
was not to create a trading bot, but to **exploit the capabilities of the transformer model**.

The available information that Tiingo provides is:

- **Date**: the date of the time point
- **Low/High price**: the lowest and highest price of the asset in the time resolution
- **Close/Open price**: the price of the asset at the end and the beginning of the time resolution
- **Volume**:  total number of shares or contracts traded during a specified resolution
- **Notional volume**: total value of the assets traded, rather than the number of units
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

You can find the data in the `/data/dataset/` folder. Data has been compressed to be uploaded to the repository. Available data is:

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

Additionally, some **financial indicators** such as EMA, RSI, or Bollinger Bands (BB) have been included. Indicators were computed
with code from another project, so code is not available here, but I introduced them inside each time point on the jsons as additional fields.
I am not a trader expert (and don't pretend to!). I am **pretty sure** the model (or a future version of it) could capture
its own **internal patterns or representations** valid for predicting price. However, I thought it would be a good idea to
introduce them as a starting point to guide training of a simpler architecture of the model.

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

As you will see in the :ref:`dataset_configuration`, you can choose to use them (or some of them) for training the model.

.. note::
    As a result of reviewing the literature, I found that **better prediction accuracy** is obtained with the inclusion of **sentiment analysis**, as it captures traders feelings and emotions quicker. However, I did not include it in this project for **simplicity** reasons.


Normalization techniques
~~~~~~~~~~~~~~~~~~~~~~~~

Several methods of data normalization have been implemented. In the literature,
different approaches such as **window** or **global normalization** have been observed.
Therefore, all of them have been implemented with to test and determine which method allows for
better performance and generalization of the model. It is true that each one has its advantages and disadvantages.

**Window normalization** seems more suitable to **avoid losing too much resolution** on the data and also to ensure working over time and
not become obsolete (ticker may end up surpassing the current max price or volume). Window normalization is particularly
useful when the underlying **distribution** of the data **varies** significantly across different segments or time intervals within the dataset.
This approach allows to capture local variations in the data and adapt the normalization strategy accordingly.

**Global normalization** is a normalization across the entire dataset. This method is more suitable for ensuring that
the dataset is on a similar scale, regardless of the distribution of individual subsets of the data. If min and max range is too
large then resolution may be lost. If using multiple tickers, it is more pronounced (e.g., 'btc_usd' and 'eth_usd' have different scales).

.. list-table:: Implemented normalization methods
   :widths: 25 25 25 25 25

   * - `window_minmax`
     - `window_meanstd`
     - `global_minmax`
     - `global_meanstd`
     - `none`

Dataset class
~~~~~~~~~~~~~
The dataset class has been implemented using **PyTorch** since there is no native version in Flax or JAX that provides
the same functionality. To make it compatible with **JAX**, a function `jax_collate_fn` has been implemented to transform data into `jnp.array`s
according to the `JAX official documentation <https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html>`_.

.. code-block:: python

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


The dataset class allows training with **multiple tickers**. Internally, it loads into a pandas dataframe the file of each ticker
(in the specified JSON format) and manages training with data from each one altogether. This has been added because training
with only one ticker resulted in too few data (you will see on :ref:`results` section), and because the more variability and patterns the agent sees, the better
generalization it will have, regardless of the ticker.

For better understanding of generalization capabilities, the test set is taken from the last dataset components; simulating real-world prediction.
When training with multiple tickers, the test set is taken from the last dataset components of every ticker.
Therefore, we **can test the model's performance on each ticker separately**.

Dataset compute internally the **normalization method**, and return it on the `__item__` function to later plotting or denormalizing for metric computing.
As previously mentioned, dataset can manage the inclusion of financial indicators if provided in the configuration file.

As you must have noticed, the `jax_collate_fn` return several components:

#. **batched_jax_sequence_tokens**: batched sequence tokens (aka time points).
#. **batched_jax_extra_tokens**: batched extra tokens (values that are not sequences, just single values as window std, sentiment analysis, etc.). Sequence is split from extra tokens as they cannot be batched together in a `jnp.array`. For the moment, only std values are included here.
#. **batched_jax_labels**: next time point to predict (aka labels).
#. **norms**: a dict with the normalizers for that window (price, volume, etc.).

   .. code-block:: python

      self._global_normalizer = dict(
         price=dict(min_val=self._data[0][Dataset.OHLC].min().min(),
                    max_val=self._data[0][Dataset.OHLC].max().max(),
                    mode="minmax"),
         volume=dict(min_val=self._data[0]['volume'].min().min(),
                     max_val=self._data[0]['volume'].max().max(),
                     mode="minmax"),
         trades=dict(min_val=self._data[0]['tradesDone'].min().min(),
                     max_val=self._data[0]['tradesDone'].max().max(),
                     mode="minmax"))
#. **timesteps**: the time value of each time point (useful for plotting and for time2vec).


.. _dataset_configuration:

Dataset configuration
~~~~~~~~~~~~~~~~~~~~~
The dataset configuration acts as an **abstraction of the dataset class**:

.. code-block:: python

    datapath: str # path to the data ('./data/datasets/')
    seq_len: int  # sequence length (window size)
    norm_mode: str  # normalization mode (window_minmax, window_meanstd, global_minmax, global_meanstd, none)
    initial_date: Optional[str]  # initial date to start the dataset (you may have data from 2016, but you want to start from 2018)
    output_mode: str  # output mode (related to model config: 'mean', 'distribution', 'discrete_grid')
    discrete_grid_levels: Optional[List[float]] # levels for the discrete grid output mode
    resolution: str # resolution of the data (4h, 1h, 30m)
    tickers: List[str]  # tickers to train with (must be in the data folder)
    indicators: Optional[List[str]]  # financial indicators to include in the dataset (e.g., ['rsi'])
