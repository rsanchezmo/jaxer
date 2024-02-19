import optax


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
