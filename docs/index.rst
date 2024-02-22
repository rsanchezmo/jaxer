Welcome to Jaxer's documentation!
===================================

Jax and Flax Time Series Prediction Transformer. The initial goal of this repo was to learn
`Jax <https://jax.readthedocs.io/en/latest/>`_ and `Flax <https://flax.readthedocs.io/en/latest/>`_ by implementing a
deep learning model. In this case, a **transformer** for time series prediction. I have decided to predict **cryptocurrency** due to
the availability of the data. However, this could be used for any other time series data, such as **market stocks** or **energy demand**.

.. note::
   This doc has been generated not only to **show my work** but also to document a **real example on how to use Jax and Flax** to build a sota deep learning model from scratch. Hope you find this library useful. **Feel free to contribute to the project and make it better**. You are welcome to open an **issue** or a **pull request**!

In order to solve cryptocurrency prediction, several approaches are proposed:

- **Mean prediction**: Predict the next value of the time series (mean).
- **Distribution prediction**: Predict the distribution over the next step (mean and log std) to cover the uncertainty.
- **Classification prediction**: Predict the next step as a classification problem (% interval ranges).

In the section :ref:`usage` you will find how to use the model. Dataset and normalization techniques are covered in the section :ref:`dataset`.
In the section :ref:`model` you will find the details of the model architectures for each approach.
Training details are covered in the section :ref:`training` (e.g. learning rate functions, performance metrics, etc).
The section :ref:`results` will cover some results obtained with each model and end up with overall conclusions.
In the section :ref:`api` you will find the API documentation to better understand how code is structured.

.. image:: images/btc_transformer.png

.. raw:: html

   <p style="text-align: center;"><em>Render of a transformer model as a hologram, projecting from a digital device, with a faint BTC logo in the holographic projection, without any text (DALLE·3)</em></p>

Contents
--------

.. toctree::
   :maxdepth: 3

   Home <self>
   usage
   dataset
   model
   training
   results
   api

------

.. _Citation:

Citation
========

If you find :code:`jaxer` useful, please consider citing:

.. code-block:: bibtex

    @misc{2024rsanchezmo,
      title     = {Jaxer: Jax and Flax Time Series Prediction Transformer},
      author    = {Rodrigo Sánchez Molina},
      year      = {2024},
      howpublished = {https://github.com/rsanchezmo/jaxer}
    }
