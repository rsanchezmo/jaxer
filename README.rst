Jaxer
=====

Jax and Flax Time Series Prediction Transformer. The goal of this repo is to learn `Jax <https://jax.readthedocs.io/en/latest/>`_ and `Flax <https://flax.readthedocs.io/en/latest/>`_ by implementing a deep learning model. In this case, a **transformer** for time series prediction. I have decided to predict BTC due to the availability of the data. However, this could be used for any other time series data, such as market stocks.

.. note::
   This repository is yet under development.

.. image:: /data/docs/btc_transformer.png
   :alt: Jaxer Logo
   :align: center

Roadmap
-------

- Download BTC dataset - Done
- Create a dataset class - Done
- Add normalizer as dataset output - Done
- Create the trainer class - Done
- Train the model:
    - **Encoder** only (next day prediction) - Done
    - **Encoder + Decoder** (N days predictions)
- Create an agent that loads the model and acts as a predictor - Done
- Add tensorboard support to the trainer class - Done
- Add a logger to the trainer class - Done
- Add a lr scheduler to the trainer class: warmup cosine scheduler - Done
- Make the prediction head output a distribution instead of a single value to cover uncertainty - Done
- Include Time2Vec as time embedding - Done
- Select output head format [resblocks, mlp, etc] - Done
- Select input feature extractor format [resblocks, mlp, etc] - Done
- Create HP tuner - Done
- Add support for outputting discrete ranges instead of a distribution or the mean - Done
- Add support for multiple tickers dataset (to get more points)
- Refactor the plotter code to be more general
- Make logger a singleton

Description
-----------

Dataset
~~~~~~~

The dataset has been downloaded from `Tiingo <https://www.tiingo.com/>`_ using the free `Tiingo Python API REST <https://www.tiingo.com/documentation/crypto>`_. The dataset consists of the BTC/USD price from 2018-01-01 to 2024-01-01. Three time resolutions are available: 4h, 1h, 30m in order to evaluate the model performance on different time resolutions. I first downloaded one day time resolution from Yahoo Finance, but data was really scarce. Tiingo provides a JSON with timepoints and was the best option from my search on the internet for free data.

The available information is:

- Low/High price
- Close/Open price
- Volume
- Number of trades

The dataset class is implemented in PyTorch due to the easiness of creating a dataloader. However, as we are working with Jax arrays, it was necessary to pass a function to the dataloader to map the Torch tensors to Jax.ndarrays.

The test set is not randomly computed across the entire dataset. For better generalization capabilities understanding, the test set is taken from the last dataset components; simulating real-world prediction.

Normalization
~~~~~~~~~~~~~

Data must be normalized to avoid exploding gradients during training. Several normalization methods are implemented. First, a normalization across the entire dataset. Second, a window normalization. The second one seems more suitable to avoid losing too much resolution and also to ensure working over time and not become obsolete (BTC may surpass the current max BTC price or volume). However, this approach can introduce inconsistencies in data scaling across different windows, potentially affecting the model's performance and its ability to generalize to new data. For each approach, two normalizers are implemented: a min-max scaler and a standard scaler.

Model
~~~~~

Encoder Only
^^^^^^^^^^^^

Model is implemented in Flax. Feature extractors and prediction head consist of an MLP with residual blocks and layer normalization. The encoder consists of L x Encoder blocks with self-attention, layer norms, and feedforwarding. The output of the encoder is flattened to feed the prediction head. Each token of the output sequence contains the "context" of the others related to itself.

We could get the last token instead of flattening the encoder output, but as we are not masking the attention, each contextual embedding can be valuable for the prediction head. However, this is something I am exploring these days. The need to mask the attention, and if not, the need to use the positional encoding.

The output of the prediction head is a probability distribution. I want to have a better understanding of the uncertainty of the model, so I have decided to use a distribution instead of a single value. The distribution is a normal distribution with mean and variance. By doing so, the loss function is the negative log likelihood of the distribution.

Results
-------

Some of the predictions are shown below.

Output Distribution
^^^^^^^^^^^^^^^^^^^

As we are predicting a distribution, the 95% confidence interval is shown in the plots to have a better understanding of the uncertainty of the model. The upper and lower bounds are computed as [mean + 1.96*std, mean - 1.96*std] respectively.

.. image:: /data/docs/1.png
   :alt: Jaxer Predictions Distribution 1
   :align: center

.. image:: /data/docs/4.png
   :alt: Jaxer Predictions Distribution 2
   :align: center

Now, the model is either predicting the window mean or lagging the input sequence. It is related to model size and hp and hope to fix it soon.

.. image:: /data/docs/mean_test.png
   :alt: Jaxer Predictions Test
   :align: center

.. image:: /data/docs/mean_test_2.png
   :alt: Jaxer Predictions Train
   :align: center

Output Mean
^^^^^^^^^^^

.. image:: /data/docs/3.png
   :alt: Jaxer Predictions Mean 1
   :align: center

Output Discrete Grid
^^^^^^^^^^^^^^^^^^^

.. image:: /data/docs/6.png
   :alt: Jaxer Predictions Discrete 1
   :align: center

.. image:: /data/docs/9.png
   :alt: Jaxer Predictions Discrete 2
   :align: center

Conclusions
-----------

- Jax and Flax are easy to use once you learn the basics. I have tried to make the code as simple as possible so it can be easily understood, encapsulating the complexity of the libraries.
- The model performance is not bad at all considering the input information that feeds the model. However, dataset size is small due to one day resolution. Increasing resolution may improve the model performance.
- The general idea from this repo is that the transformer can be applied to time series prediction and can be implemented with state-of-the-art GPU-accelerated deep learning libraries such as Jax and Flax.

Future Work
-----------

- TBD

Contributors
------------

Rodrigo Sánchez Molina

- Email: rsanchezm98@gmail.com
- Linkedin: `rsanchezm98 <https://www.linkedin.com/in/rsanchezm98/>`_
- Github: `rsanchezmo <https://github.com/rsanchezmo>`_
