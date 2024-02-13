Usage
=====

.. _installation:

Installation
------------

Clone the repo:

.. code-block:: bash

    git clone https://github.com/rsanchezmo/jaxer.git
    cd jaxer

Create a python venv and source it:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate

Install your desired Jax version. For example, if already installed CUDA 12 on Linux (make sure to have exported to PATH your CUDA version):

.. code-block:: bash

    (venv) $ pip install jax[cuda12_local] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Then install the rest of the dependencies (which are in the `requirements.txt` file):

.. code-block:: bash

    (venv) $ pip install .

.. _usage:

Usage
-----

Training
~~~~~~~

Set the configuration parameters in the `training_config.py` file. The training of the model is made really easy, as simple as creating a trainer and calling the `train_and_evaluate` method:

.. code-block:: python

    from jaxer.utils.trainer import FlaxTrainer as Trainer
    from training_config import config

    trainer = Trainer(config=config)
    trainer.train_and_evaluate()

Inference
~~~~~~~~~

An agent class has been created so you can load a model and use it to predict any data you want:

.. code-block:: python

    from jaxer.utils.agent import Agent
    from jaxer.utils.config import get_best_model
    from jaxer.utils.plotter import predict_entire_dataset
    import jax.numpy as jnp

    # the experiment in the "results" folder
    experiment = "transformer_encoder_only_window"
    agent = Agent(experiment=experiment, model_name=get_best_model(experiment))

    # a random input
    x_test = jnp.ones((1, agent.config.model_config["max_seq_len"], agent.config.model_config["input_features"]))

    # predict
    pred = agent(x_test)

    # plot the test set predictions
    dataset = Dataset(agent.config.dataset_path, agent.config.model_config["max_seq_len"], norm_mode=agent.config.normalizer_mode, initial_date=agent.config.initial_date, output_mode=agent.model_config["output_mode"])

    train_ds, test_ds = dataset.get_train_test_split(test_size=agent.config.test_split)

    predict_entire_dataset(agent, test_ds, mode='test', output_mode=agent.config.model_config["output_mode"])

