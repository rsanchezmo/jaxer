.. _usage:

Usage
=====

.. _installation:

Installation
------------

I highly recommend to clone and run the repo on :code:`linux` device because the installation of :code:`jax` and related libraries is easier,
but with a bit of time you may end up running on other compatible platform. To start, just clone the repo:

.. code-block:: bash

    git clone https://github.com/rsanchezmo/jaxer.git
    cd jaxer

Create a python **venv** and source it:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate

Install your desired :code:`jax` version (more info at `jax installation official doc <https://jax.readthedocs.io/en/latest/installation.html>`_).
For example, if already installed :code:`CUDA 12` on linux (make sure to have exported to :code:`PATH` your :code:`CUDA` version):

.. code-block:: bash

    (venv) $ pip install jax[cuda12_local] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Then install the rest of the dependencies (which are in the :code:`requirements.txt` file):

.. code-block:: bash

    (venv) $ pip install .

You could have omitted the :code:`jax` installation as :code:`flax` library installs it, but it **is preferred to select the proper version** of :code:`jax`
**according to the hardware**.

.. _running_the_code:

Running the code
----------------
Dataset
~~~~~~~
You should first unzip the dataset in the :code:`dataset` folder. You will end up with several subfolders regarding each time resolution (e.g. :code:`1h`):

.. code-block:: bash

    cd ./data/datasets/
    unzip data.zip


Training
~~~~~~~~

The training of the model is made really easy, as simple as creating a **trainer** with the experiment configuration and calling the :code:`train_and_evaluate` method:

.. code-block:: python

    import jaxer
    from training_config import config


    if __name__ == '__main__':

        trainer = jaxer.run.FlaxTrainer(config=config)
        trainer.train_and_evaluate()


Configuration
~~~~~~~~~~~~~
An example of the **experiment configuration** is in the :code:`training_config.py` file:

.. code-block:: python

    import jaxer

    output_mode = 'mean'
    seq_len = 24
    d_model = 128

    model_config = jaxer.config.ModelConfig(
        d_model=d_model,
        num_layers=2,
        head_layers=2,
        n_heads=4,
        dim_feedforward=4 * d_model,
        dropout=0.05,
        max_seq_len=seq_len,
        flatten_encoder_output=False,
        fe_blocks=0,
        use_time2vec=False,
        output_mode=output_mode,
        use_resblocks_in_head=False,
        use_resblocks_in_fe=True,
        average_encoder_output=False,
        norm_encoder_prev=True
    )

    dataset_config = jaxer.config.DatasetConfig(
        datapath='./data/datasets/data/',
        output_mode=output_mode,
        discrete_grid_levels=None,
        initial_date='2018-01-01',
        norm_mode="global_minmax",
        resolution='4h',
        tickers=['btc_usd'],
        indicators=None,
        seq_len=seq_len,
    )

    config = jaxer.config.ExperimentConfig(
        model_config=model_config,
        log_dir="results",
        experiment_name="exp_1",
        num_epochs=500,
        learning_rate=5e-4,
        lr_mode='cosine',
        warmup_epochs=15,
        dataset_config=dataset_config,
        batch_size=128,
        test_split=0.1,
        test_tickers=['btc_usd'],
        seed=0,
        save_weights=True,
        early_stopper=100
    )

You can find a more detailed explanation of each parameter in the :ref:`api`, :ref:`dataset`,
:ref:`training` and :ref:`model` sections.

Inference
~~~~~~~~~

An **agent class** has been created so you can load a trained model and use it by providing the agent with a proper input. Agent
can infer by using :code:`__call__` method:

.. code-block:: python

    import jaxer

    from torch.utils.data import DataLoader

    if __name__ == '__main__':
        # load the agent with best model weights
        experiment = "results/exp_1"
        agent = jaxer.run.Agent(experiment=experiment, model_name=jaxer.utils.get_best_model(experiment))

        # create dataloaders
        dataset = jaxer.utils.Dataset(dataset_config=jaxer.config.DatasetConfig.from_dict(agent.config.dataset_config))
        train_ds, test_ds = dataset.get_train_test_split(test_size=agent.config.test_split, test_tickers=agent.config.test_tickers)

        # infer entire dataset
        plot_entire_dataset = False
        if plot_entire_dataset:
            jaxer.utils.predict_entire_dataset(agent, test_ds, mode='test')
            jaxer.utils.predict_entire_dataset(agent, train_ds, mode='train')

        # infer once over the test set
        test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=jaxer.utils.jax_collate_fn)
        for batch in train_dataloader:
            x, y_true, normalizer, window_info = batch
            y_pred = agent(x)
            jaxer.utils.plot_predictions(x=x, y_true=y_true, y_pred=y_pred, normalizer=normalizer, window_info=window_info)

On this example, a :code:`jaxer` **agent** is created with the **best weights** of the experiment :code:`exp_1`.
The :code:`plot_entire_dataset` flag is used to plot over the entire dataset (:code:`train` and :code:`test`), which is useful to see model performance (debug if overfitting or generalization).
Finally, the agent is used to predict on separate windows from the test set to see a more detailed prediction plot.