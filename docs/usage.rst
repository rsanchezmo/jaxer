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

You can install your desired :code:`jax` version (more info at `jax installation official doc <https://jax.readthedocs.io/en/latest/installation.html>`_).
You must notice that jax :code:`currently` only provides 2 distributions with cuda (:code:`12.3` and :code:`11.8`).
As this repo also depends on :code:`torch`, you could install :code:`torch` in cpu and :code:`jax` in gpu, :code:`torch`
is only used for the dataloaders. However, I decided to better install cuda 11.8 versions as they are compatible.
For example, if already installed :code:`CUDA 11.8` on linux (make sure to have exported to :code:`PATH` your :code:`CUDA` version):

.. code-block:: bash

    (venv) $ pip install jax[cuda11_local] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    (venv) $ pip install torch --index-url https://download.pytorch.org/whl/cu118

If :code:`CUDA` is not installed, you may better use pip installation and let pip install the right version for your gpu. Make sure that
when doing ::code:`nvcc --version`, it says that you should install cuda toolkit (just remove if you have a release already reported on
:code:`.bashrc`):

.. code-block:: bash

    (venv) $ pip install --upgrade jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_releases.html
    (venv) $ pip install torch --index-url https://download.pytorch.org/whl/cu118

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

    output_mode = 'mean'  # 'mean' or 'distribution' or 'discrete_grid
    seq_len = 100
    d_model = 128

    model_config = jaxer.config.ModelConfig(
        d_model=d_model,
        num_layers=4,
        head_layers=2,
        n_heads=2,
        dim_feedforward=4 * d_model,  # 4 * d_model
        dropout=0.05,
        max_seq_len=seq_len,
        flatten_encoder_output=False,
        fe_blocks=0,  # feature extractor is incremental, for instance input_shape, 128/2, 128 (d_model)
        use_time2vec=False,
        output_mode=output_mode,  # 'mean' or 'distribution' or 'discrete_grid'
        use_resblocks_in_head=False,
        use_resblocks_in_fe=True,
        average_encoder_output=False,
        norm_encoder_prev=True
    )

    dataset_config = jaxer.config.DatasetConfig(
        datapath='./data/datasets/data/',
        output_mode=output_mode,  # 'mean' or 'distribution' or 'discrete_grid
        discrete_grid_levels=[-9e6, 0.0, 9e6],
        initial_date='2018-01-01',
        norm_mode="window_minmax",
        resolution='30m',
        tickers=['btc_usd', 'eth_usd', 'sol_usd'],
        indicators=None,
        seq_len=seq_len
    )

    synthetic_dataset_config = jaxer.config.SyntheticDatasetConfig(
        window_size=seq_len,
        output_mode=output_mode,  # 'mean' or 'distribution' or 'discrete_grid
        normalizer_mode='window_minmax',  # 'window_meanstd' or 'window_minmax' or 'window_mean'
        add_noise=False,
        min_amplitude=0.1,
        max_amplitude=1.0,
        min_frequency=0.5,
        max_frequency=30,
        num_sinusoids=5,
    )

    pretrained_folder = "results/exp_synthetic_context"
    pretrained_path_subfolder, pretrained_path_ckpt = jaxer.utils.get_best_model(pretrained_folder)
    pretrained_model = (pretrained_folder, pretrained_path_subfolder, pretrained_path_ckpt)

    config = jaxer.config.ExperimentConfig(
        model_config=model_config,
        pretrained_model=pretrained_model,
        log_dir="results",
        experiment_name="exp_both_pretrained_synthetic",
        num_epochs=1000,
        steps_per_epoch=500,  # for synthetic dataset only
        learning_rate=5e-4,
        lr_mode='cosine',  # 'cosine'
        warmup_epochs=15,
        dataset_mode='both',  # 'real' or 'synthetic' or 'both'
        dataset_config=dataset_config,
        synthetic_dataset_config=synthetic_dataset_config,
        batch_size=256,
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
        experiment = "exp_1"
        agent = jaxer.run.Agent(experiment=experiment, model_name=jaxer.utils.get_best_model(experiment))

        # creater dataloaders
        if agent.config.dataset_mode == 'synthetic':
            dataset = jaxer.utils.SyntheticDataset(config=jaxer.config.SyntheticDatasetConfig.from_dict(agent.config.synthetic_dataset_config))
            test_dataloader = dataset.generator(batch_size=1, seed=100)
            train_dataloader = dataset.generator(batch_size=1, seed=200)
        else:
            dataset = jaxer.utils.Dataset(dataset_config=jaxer.config.DatasetConfig.from_dict(agent.config.dataset_config))

            train_ds, test_ds = dataset.get_train_test_split(test_size=agent.config.test_split,
                                                             test_tickers=agent.config.test_tickers)
            train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=jaxer.utils.jax_collate_fn)
            test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=jaxer.utils.jax_collate_fn)

        infer_test = False

        if infer_test:
            dataloader = test_dataloader
        else:
            dataloader = train_dataloader

        if agent.config.dataset_mode == 'synthetic':
            for i in range(30):
                x, y_true, normalizer, window_info = next(dataloader)
                y_pred = agent(x)
                jaxer.utils.plot_predictions(x=x, y_true=y_true, y_pred=y_pred, normalizer=normalizer,
                                             window_info=window_info, denormalize_values=True)

        else:
            for batch in dataloader:
                x, y_true, normalizer, window_info = batch
                y_pred = agent(x)
                jaxer.utils.plot_predictions(x=x, y_true=y_true, y_pred=y_pred, normalizer=normalizer,
                                             window_info=window_info[0], denormalize_values=True)

On this example, a :code:`jaxer` **agent** is created with the **best weights** of the experiment :code:`exp_1`.
The :code:`plot_entire_dataset` flag is used to plot over the entire dataset (:code:`train` and :code:`test`), which is useful to see model performance (debug if overfitting or generalization).
Finally, the agent is used to predict on separate windows from the test set to see a more detailed prediction plot.