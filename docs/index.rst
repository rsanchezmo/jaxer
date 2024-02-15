Welcome to Jaxer's documentation!
===================================

Jax and Flax Time Series Prediction Transformer. The initial goal of this repo was to learn
`Jax <https://jax.readthedocs.io/en/latest/>`_ and `Flax <https://flax.readthedocs.io/en/latest/>`_ by implementing a
deep learning model. In this case, a **transformer** for time series prediction. I have decided to predict cryptocurrency due to
the availability of the data. However, this could be used for any other time series data, such as market stocks or energy demand.

In order to solve cryptocurrency prediction, several approaches are proposed:

- **Mean prediction**: Predict the next value of the time series (mean).
- **Distribution prediction**: Predict the distribution over the next step (mean and log std) to cover the uncertainty.
- **Classification prediction**: Predict the next step as a classification problem (% interval ranges).

In the section :ref:`usage` you will find how to use the model.
In the section :ref:`model` you will find the details of the model architectures for each approach.
The section :ref:`results` will cover some results obtained with each model and end up with overall conclusions.
In the section :ref:`api` you will find the API documentation to better understand how code is structured.

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
   api

------

.. _Citation:

Citation
========

If you find Jaxer useful, please consider citing:

.. code-block:: bibtex

    @misc{2024rsanchezmo,
      title     = {Jaxer: Jax and Flax Time Series Prediction Transformer},
      author    = {Rodrigo Sánchez Molina},
      year      = {2024},
      howpublished = {https://github.com/rsanchezmo/jaxer}
    }