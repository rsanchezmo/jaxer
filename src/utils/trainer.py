from ..model.flax_transformer import Transformer, TransformerConfig
from ..configs.config import Config
from ..utils.dataset import Dataset
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tensorboard import SummaryWriter
import os
from typing import Tuple, Dict


class TrainerBase:
    def __init__(self, config: Config) -> None:
        """ Trainer Base Class: All trainers should inherit from this, FLAX or PyTorch [in the future] """
        self._config = config

        """ Training logs and checkpoints """
        self._log_dir = os.path.join(os.getcwd(), self._config.log_dir)
        os.makedirs(self._log_dir, exist_ok=True)
        self._summary_writer = SummaryWriter(self._log_dir)

    def train_and_evaluate(self) -> None:
        """ Runs a training loop """
        raise NotImplementedError


class FlaxTrainer(TrainerBase):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._flax_model_config = TransformerConfig(
            d_model=self._config.model_config.d_model,
            num_layers=self._config.model_config.num_layers,
            n_heads=self._config.model_config.n_heads,
            dim_feedforward=self._config.model_config.dim_feedforward,
            dropout=self._config.model_config.dropout,
            max_seq_len=self._config.model_config.max_seq_len,
            dtype=jnp.float32
        )

    def train_and_evaluate(self) -> None:
        """ Runs a training loop """

        """ Get dataloaders for training and evaluation """


        """ Create a training state """
        rng, init_rng = jax.random.key(0)
        train_state = self._create_train_state(init_rng)


        best_loss = float("inf")


        for epoch in range(self._config.num_epochs):
            rng, input_rng = jax.random.split(rng)

            """ Training """
            state, loss, metrics = self._train_step(train_state, train_ds, self._config.batch_size, input_rng)
            
            """ Logging """
            test_loss, test_metrics = self._evaluate_step(state, test_ds)

            """ Logging """
            self._summary_writer.scalar("train_loss", loss, epoch)
            self._summary_writer.scalar("test_loss", test_loss, epoch)
            self._summary_writer.scalar("train_mae", metrics["mae"], epoch)
            self._summary_writer.scalar("test_mae", test_metrics["mae"], epoch)
            self._summary_writer.scalar("train_r2", metrics["r2"], epoch)
            self._summary_writer.scalar("test_r2", test_metrics["r2"], epoch)

            if test_loss < best_loss:
                best_loss = test_loss
                self._save_model(epoch, state)

        self._summary_writer.flush()


    def _create_train_state(self, rng: jax.random.PRNGKey) -> train_state.TrainState:
        """ Creates a training state """
        
        model = Transformer(self._flax_model_config)

        input_shape = (1, self._flax_model_config.max_seq_len, self._config.model_config.input_features)
        params = model.init(rng, jnp.ones((input_shape), dtype=jnp.float32))

        # optimizer
        tx = optax.adamw(learning_rate=self._config.learning_rate)

        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
        
    @jax.jit
    def _apply_model(self, state: train_state.TrainState, inputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:

        def loss_fn(params):
            predictions = state.apply_fn({"params": params}, inputs)
            loss = jnp.mean((predictions - targets) ** 2)  # MSE loss
            mae = jnp.mean(jnp.abs(predictions - targets))  # MAE loss
            r2 = 1 - jnp.sum((predictions - targets)**2) / jnp.sum((targets - jnp.mean(targets))**2)  # R2 loss
            return loss, mae, r2

        (loss, mae, r2), grad = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        metrics = {"mae": mae, "r2": r2, "loss": loss}

        return grad, loss, metrics
    
    @jax.jit
    def _update_model(self, state: train_state.TrainState, grads) -> train_state.TrainState:
        return state.apply_gradients(grads=grads)
    

    def _train_step(self, state: train_state.TrainState, data) -> Tuple[train_state.TrainState, float, Dict]:
        """ Runs a training step """

        inputs, targets = data
        grad, loss, metrics = self._apply_model(state, inputs, targets)
        state = self._update_model(state, grad)
        return state, loss, metrics


    def _evaluate_step(self, state: train_state.TrainState, data) -> Tuple[train_state.TrainState, float, Dict]:
        """ Runs an evaluation step """
        inputs, targets = data
        _, loss, metrics = self._apply_model(state, inputs, targets)
        return loss, metrics

    def _save_model(self, epoch: int, state: train_state.TrainState) -> None:
        """ Saves a model checkpoint """
        pass