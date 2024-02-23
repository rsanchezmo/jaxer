.. _results:

Results
=======

Hardware
--------
The code was run on my laptop which has an :code:`Intel Core i9-13900H CPU @ 5.4GHz` with :code:`32GB of RAM` and a
:code:`NVIDIA RTX 4070 8GB GPU`.
As you will see on the :ref:`conclusions` section, **this was my biggest bottleneck**, even though it is a very powerful machine.

To get a **representative study** over each model, I run the **hyperparameter search** for :code:`N` experiments (N adapted to the computation time and HW of my laptop)
over a set of hyperparameters. Metrics were computed during training only on :code:`'btc_usd'` ticker, even if trained with
all tickers. :code:`EarlyStopper` was used to save time by stopping the training when the model stops improving for a fixed number of epochs.

Mean Prediction
---------------
TBC

Distribution Prediction
-------------------------
TBC

Classification Prediction
-------------------------
TBC


.. _conclusions:

Conclusions
-----------
TBC