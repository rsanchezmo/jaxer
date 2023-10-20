from ..model.transformer import Transformer
from ..configs.config import Config
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tensorboard import SummaryWriter
import os
from typing import Tuple, Dict


class Trainer:
    def __init__(self, config: Config) -> None:
        """ Trainer class """
        self._config = config

        """ Training logs and checkpoints """
        self._log_dir = os.path.join(os.getcwd(), self._config.log_dir)
        os.makedirs(self._log_dir, exist_ok=True)
        self._summary_writer = SummaryWriter(self._log_dir)


    def train_and_evaluate(self) -> None:
        """ Runs a training loop """

        """ Get dataloaders for training and evaluation """


        """ Create a training state """
        rng, init_rng = jax.random.key(0)
        train_state = self._create_train_state(init_rng)


        for epoch in range(self._config.num_epochs):
            rng, input_rng = jax.random.split(rng)

            """ Training """
            state, loss, metrics = self._train_step(train_state, train_ds, self._config.batch_size, input_rng)
            
            """ Logging """
            _, test_loss, test_metrics = self._evaluate_step(state, test_ds)

            """ Logging """
            self._summary_writer.scalar("train_loss", loss, epoch)
            self._summary_writer.scalar("test_loss", test_loss, epoch)


    def _create_train_state(self, rng: jax.random.PRNGKey) -> train_state.TrainState:
        """ Creates a training state """
        
        model = Transformer(self._config.model_config)

        input_shape = (1, self._config.model_config.max_seq_len, 1)
        params = model.init(rng, jnp.ones((input_shape), dtype=jnp.float32))

        # optimizer
        tx = optax.adamw(learning_rate=self.config.learning_rate)

        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        

    def _train_step(self, state: train_state.TrainState, data) -> Tuple[train_state.TrainState, float, Dict]:
        """ Runs a training step """
        pass

    def _evaluate_step(self, state: train_state.TrainState, data) -> Tuple[train_state.TrainState, float, Dict]:
        """ Runs an evaluation step """
        pass