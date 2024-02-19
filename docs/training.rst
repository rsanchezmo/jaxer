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

    `"The problem with torch magic PRNG state is that it’s hard to reason about how it’s being used and updated across different threads, processes, and devices, and it’s very easy to screw up when the details of entropy production and consumption are hidden from the end user"`.

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
    seed: int  # initial seed for reproducibility
    save_weights: bool  # save weights during training
    early_stopper: int  # early stopper patience (number of epochs without improvement)


.. _metrics:

Metrics
-------
To proper evaluate how good is the model, we need to declare some metrics. As we have two main approaches: **classification** and **regression**, the
following table shows the metrics used for each case:

.. list-table::
    :header-rows: 1

    * - Task
      - Metric
    * - Classification
      - Accuracy
    * - Regression
      - Mean Squared Error, Mean Average Percentage Error, R2 Score, Mean Absolute Error

.. note::
    Metrics are computed with normalized data, so we must be careful. I found absolute magnitudes such as :code:`mse` meaningful as it is not the same
    to have a :code:`mse` of 2 :math:`$` when price is around 1 than when price is at 20000 :math:`$`. That is why I ended up looking only at the :code:`mape` on regression tasks and :code:`accuracy` on
    classification tasks. Nevertheless, the rest of the metrics are also computed and logged.