.. _training:

Training
========
This section describes the optimization process of the model. I want to delve a bit deeper into the training process in
:code:`jax` and :code:`flax` as it has **some differences** from the usual `torch` **workflow**. I assume you are familiar with deep learning so
this will not be a tutorial.

First of all, we need an **optimization library** for :code:`jax`: `optax <https://optax.readthedocs.io/en/latest/>`_. This library
allows to define **learning rate schedulers** and also **optimizers** such as :code:`adamw`. In the configuration file you can chose from
a linear learning rate scheduler or a warmup cosine learning rate scheduler. Last one is widely used when training transformer models.
:code:`optax` does not provide this scheduler by default, but can be easily implemented by joining the :code:`cosine` and :code:`linear` schedulers:

.. code-block:: python

    def create_warmup_cosine_schedule(learning_rate: float,
                                      warmup_epochs: int,
                                      num_epochs: int,
                                      steps_per_epoch: int) -> optax.Schedule:
        """Creates learning rate cosine schedule

        :param learning_rate: initial learning rate
        :type learning_rate: float

        :param warmup_epochs: number of warmup epochs
        :type warmup_epochs: int

        :param num_epochs: total number of epochs
        :type num_epochs: int

        :param steps_per_epoch: number of steps per epoch
        :type steps_per_epoch: int

        :return: learning rate schedule
        :rtype: optax.Schedule
        """
        warmup_fn = optax.linear_schedule(
            init_value=0., end_value=learning_rate,
            transition_steps=warmup_epochs * steps_per_epoch)
        cosine_epochs = max(num_epochs - warmup_epochs, 1)
        cosine_fn = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=cosine_epochs * steps_per_epoch)
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_epochs * steps_per_epoch])
        return schedule_fn


Even :code:`optax` also had several loss functions implemented, I coded my own functions (code is indeed very similar). :code:`flax` **does not store the model parameters in the model itself**. This the main difference from :code:`torch`. Instead, the model
parameters are stored in a separate :code:`params` object. This object is passed to the :code:`apply` method of the model. Therefore, the
process of calling (inference) the model will be:

#. :code:`params = model.init(rng, input_shape)` to initialize the model parameters
#. :code:`output = model.apply(params, input)` to make a forward pass

However, :code:`flax` has a :code:`TrainState` object that **stores the model parameters, the optimizer state and the learning rate scheduler** to
keep everything in one place. I found this really helpful. You can find more information on the `official documentation <https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.train_state.TrainState>`_.
:code:`TrainState` is easily created by:

.. code-block:: python

    optimizer = optax.adamw(learning_rate=learning_rate_scheduler)

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=initial_params,
        tx=optimizer
    )

Another key difference is :code:`jax` **randomness**. To obtain random numbers during training, we start with a :code:`random.PRNGKey(seed)` and then
we split this key to obtain a new key and a new subkey for each operation. This must be done if using random layers in the model
such as :code:`dropout`. In :code:`torch` we only need to set the seed once and then
everything is run from there. :code:`jax` is a bit more complex, but comes to solve the following issue:

.. note::

    The problem with :code:`torch` magic :code:`PRNG` state is that it’s hard to reason about how it’s being used and updated across different threads, processes, and devices, and it’s very easy to screw up when the details of entropy production and consumption are hidden from the end user.

Additionally, I found interesting the use of :code:`orbax` to manage **checkpoint saving and loading** and it is recommended by `flax <https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html>`_.
For instance, we can define a checkpoint manager that saves up to 5 best models:

.. code-block:: python

        self._orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(create=True, max_to_keep=5)
        self._checkpoint_manager = orbax.checkpoint.CheckpointManager(
            str(self._ckpts_dir), self._orbax_checkpointer, options)

        # to save the model
        if test_metric < best_test_metric:
            ckpt = {'model': trained_state}
            save_args = orbax_utils.save_args_from_target(ckpt)
            self._checkpoint_manager.save(epoch, ckpt, save_kwargs={'save_args': save_args})

        # to load the model
        restored_state = self._orbax_checkpointer.restore('model_path')['model']


To later visualize the training process, I used :code:`tensorboard` to record logs of train/test metrics. This is a very useful tool to **visualize** the :ref:`metrics`
of **train and test set**. There is also an :code:`early stopper` class to **stop the training process** if the test metric does not improve after
a certain number of epochs.

.. code-block:: python

    @dataclass
    class EarlyStopper:
        """Early stopper class

        :param max_epochs: max number of epochs without improvement
        :type max_epochs: int

        :param n_epochs: number of epochs without improvement
        :type n_epochs: int

        :param optim_value: best optimization value
        :type optim_value: float
        """
        max_epochs: int
        n_epochs: int = 0
        optim_value: float = 1e9

        def __call__(self, optim_value: float):
            """ Returns True if the training should stop """
            if optim_value < self.optim_value:
                self.optim_value = optim_value
                self.n_epochs = 0
                return False

            self.n_epochs += 1

            if self.n_epochs >= self.max_epochs:
                return True

            return False

Configuration
-------------
Training configuration must be filled on its dataclass:

.. code-block:: python

    model_config: ModelConfig  # model configuration (transformer)
    log_dir: str  # directory to save logs
    experiment_name: str  # experiment name (logs will be saved on log_dir/experiment_name)
    num_epochs: int  # number of epochs
    learning_rate: float  # initial learning rate
    lr_mode: str  # learning rate scheduler mode (linear or cosine)
    warmup_epochs: int  # number of warmup epochs
    dataset_config: DatasetConfig  # dataset configuration
    batch_size: int  # batch size
    test_split: float  # test split (between 0 and 1)
    test_tickers: List[str]  # tickers to test
    seed: int  # initial seed for reproducibility
    save_weights: bool  # save weights during training
    early_stopper: int  # early stopper patience (number of epochs without improvement)


.. _metrics:

Metrics
-------
To **proper evaluate how good is the model**, we need to declare some metrics. As we have two main approaches: **classification** and **regression**, the
following table shows the metrics used for each case:

.. list-table::
    :header-rows: 1

    * - Task
      - Metric
    * - Classification
      - Accuracy (:code:`acc`)
    * - Regression
      - Mean Squared Error (:code:`mae`), Mean Average Percentage Error (:code:`mape`), R2 Score (:code:`r2`), Mean Absolute Error (:code:`mae`)

.. note::
    Metrics were initially computed with normalized data, but it did not allow to compare over different normalization methods (the only normalization independent metric was :code:`mape`).
    For comparison reasons, I decided to **denormalize predictions and compute metrics with the original data**. This way, we can compare the metrics over different models and normalization methods.
    I found absolute magnitudes such as :code:`mse` not to be very explanatory as it is not the same to have a :code:`mse` of :code:`2$` when price is around 1 than when price is at :code:`20000$`.

Hyperparameter search
---------------------
I have also included a **very simple hyperparameter search module**. This module just runs **multiple experiments sequentially**
by providing set of hyperparameters. May add complex hyperparameter search in the future, but kept as simple as possible
as this was not the main focus of the project. This module purpose was to get to the results presented in the :ref:`results` section.

.. code-block:: python

    model_config_ranges = {
        'd_model': [128, 256, 512],
        'num_layers': [1, 2, 3],
        'head_layers': [1, 2, 3],
        'n_heads': [1, 2, 4],
        'dim_feedforward': [2, 4],
        'dropout': [0.0, 0.05, 0.1],
        'max_seq_len': [12, 24, 36, 48],
        'flatten_encoder_output': [False],
        'fe_blocks': [1, 2],
        'use_time2vec': [False, True],
        'output_mode': ['mean'],
        'use_resblocks_in_head': [False, True],
        'use_resblocks_in_fe': [False, True],
        'average_encoder_output': [False, True],
        'norm_encoder_prev': [False, True]
    }

    training_ranges = {
        'learning_rate': [1e-4, 1e-5],
        'lr_mode': ['linear', 'cosine'],
        'warmup_epochs': [5, 10, 20],
        'batch_size': [64, 128],
        'normalizer_mode': ['global_minmax', 'global_meanstd', 'window_meanstd', 'window_minmax'],
        'resolution': ['4h'],
        'tickers': ['btc_usd', 'eth_usd']
    }

    n_trainings = 20
    trained_set = set()

    counter_trains = 0
    logger = get_logger()

    while counter_trains < n_trainings:
        logger.info(f"Training {counter_trains + 1}/{n_trainings}")

        d_model = int(np.random.choice(model_config_ranges['d_model']))
        num_layers = int(np.random.choice(model_config_ranges['num_layers']))
        head_layers = int(np.random.choice(model_config_ranges['head_layers']))
        n_heads = int(np.random.choice(model_config_ranges['n_heads']))
        dim_feedforward = d_model * int(np.random.choice(model_config_ranges['dim_feedforward']))
        dropout = float(np.random.choice(model_config_ranges['dropout']))
        max_seq_len = int(np.random.choice(model_config_ranges['max_seq_len']))
        flatten_encoder_output = bool(np.random.choice(model_config_ranges['flatten_encoder_output']))
        fe_blocks = int(np.random.choice(model_config_ranges['fe_blocks']))
        use_time2vec = bool(np.random.choice(model_config_ranges['use_time2vec']))
        output_mode = str(np.random.choice(model_config_ranges['output_mode']))
        use_resblocks_in_head = bool(np.random.choice(model_config_ranges['use_resblocks_in_head']))
        use_resblocks_in_fe = bool(np.random.choice(model_config_ranges['use_resblocks_in_fe']))
        average_encoder_output = bool(np.random.choice(model_config_ranges['average_encoder_output']))
        norm_encoder_prev = bool(np.random.choice(model_config_ranges['norm_encoder_prev']))

        learning_rate = float(np.random.choice(training_ranges['learning_rate']))
        lr_mode = str(np.random.choice(training_ranges['lr_mode']))
        warmup_epochs = int(np.random.choice(training_ranges['warmup_epochs']))
        batch_size = int(np.random.choice(training_ranges['batch_size']))
        normalizer_mode = str(np.random.choice(training_ranges['normalizer_mode']))
        resolution = str(np.random.choice(training_ranges['resolution']))
        tickers = str(np.random.choice(training_ranges['tickers']))

        params = (d_model, num_layers, head_layers, n_heads, dim_feedforward, dropout, max_seq_len,
                  flatten_encoder_output, fe_blocks, use_time2vec, output_mode, use_resblocks_in_head,
                  use_resblocks_in_fe, learning_rate, lr_mode, warmup_epochs, batch_size, normalizer_mode,
                  resolution, tickers)

        if params in trained_set:
            logger.warning(f"Already trained {params}, skipping...")
            continue

        counter_trains += 1
        trained_set.add(params)

        model_config = ModelConfig(
            d_model=d_model,
            num_layers=num_layers,
            head_layers=head_layers,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            flatten_encoder_output=flatten_encoder_output,
            fe_blocks=fe_blocks,
            use_time2vec=use_time2vec,
            output_mode=output_mode,
            use_resblocks_in_head=use_resblocks_in_head,
            use_resblocks_in_fe=use_resblocks_in_fe,
            average_encoder_output=average_encoder_output,
            norm_encoder_prev=norm_encoder_prev
        )

        dataset_config = DatasetConfig(
            datapath='./data/datasets/data',
            output_mode=output_mode,
            discrete_grid_levels=[],
            initial_date='2018-01-01',
            norm_mode=normalizer_mode,
            resolution=resolution,
            tickers=[tickers],
            indicators=None,
            seq_len=max_seq_len
        )

        config = ExperimentConfig(model_config=model_config,
                                  log_dir="hp_search_report",
                                  experiment_name=output_mode,
                                  num_epochs=500,
                                  learning_rate=learning_rate,
                                  lr_mode=lr_mode,
                                  warmup_epochs=warmup_epochs,
                                  batch_size=batch_size,
                                  test_split=0.15,
                                  test_tickers=['btc_usd'],
                                  seed=0,
                                  save_weights=True,
                                  dataset_config=dataset_config,
                                  early_stopper=10000
                                  )

        trainer = Trainer(config=config)
        trainer.train_and_evaluate()
        del trainer

.. tip::
    Debugging in :code:`jax` is not as easy as in :code:`torch`. I found it very useful to use the :code:`jax.debug.print` to evaluate traced arrays
    inside :code:`jax.jit` functions or inside the :code:`flax` model. `Here <https://jax.readthedocs.io/en/latest/debugging/print_breakpoint.html>`_  you can find more info.