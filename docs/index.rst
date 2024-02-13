Welcome to Jaxer's documentation!
===================================

Jax and Flax Time Series Prediction Transformer. The initial goal of this repo was to learn
`Jax <https://jax.readthedocs.io/en/latest/>`_ and `Flax <https://flax.readthedocs.io/en/latest/>`_ by implementing a
deep learning model. In this case, a **transformer** for time series prediction. I have decided to predict cryptocurrency due to
the availability of the data. However, this could be used for any other time series data, such as market stocks.

However, it ended up in trying to solve cryptocurrency prediction, which is a complex problem, with different approaches:

- **Mean prediction**: Predict the mean of the next step.
- **Distribution prediction**: Predict the distribution of the next step (mean and log std) to cover the uncertainty.
- **Classification prediction**: Predict the next step as a classification problem (discrete grid).

.. warning::
   This repository is yet under development.

.. image:: images/btc_transformer.png

Contents
--------

.. toctree::

   Home <self>
   usage
   model
   results
   api.

------

.. _Citation:

Citation
========

If you find Jaxer useful, please consider citing:

.. code-block:: bibtex

    @misc{2024RSM,
      title     = {Jaxer: Jax and Flax Time Series Prediction Transformer},
      author    = {Rodrigo Sánchez Molina},
      year      = {2024},
      howpublished = {\url{https://github.com/rsanchezmo/jaxer}}
    }