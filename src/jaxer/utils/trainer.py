from ..model.flax_transformer import Transformer, TransformerConfig
from ..configs.config import Config
from ..utils.dataset import Dataset
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from torch.utils.tensorboard import SummaryWriter
import os
from typing import Tuple, Dict, Optional
from flax.training import orbax_utils
import orbax
from torch.utils.data import DataLoader
import time



class TrainerBase:
    def __init__(self, config: Config) -> None:
        """ Trainer Base Class: All trainers should inherit from this, FLAX or PyTorch [in the future] """
        self._config = config

        """ Training logs """
        self._log_dir = os.path.join(os.getcwd(), self._config.log_dir, self._config.experiment_name)
        os.makedirs(self._log_dir, exist_ok=True)
        self._summary_writer = SummaryWriter(os.path.join(self._log_dir, "tensorboard"))

        """ Checkpoints """
        self._ckpts_dir = os.path.join(self._log_dir, "ckpt")
        os.makedirs(self._ckpts_dir, exist_ok=True)
        self._orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
        self._checkpoint_manager = orbax.checkpoint.CheckpointManager(
            self._ckpts_dir, self._orbax_checkpointer, options)
        

        """ Dataloaders """
        dataset = Dataset(self._config.dataset_path, self._config.model_config.max_seq_len)
        train_ds, test_ds = dataset.get_train_test_split(test_size=self._config.test_split)
        self._train_dataloader = DataLoader(train_ds, batch_size=self._config.batch_size, shuffle=True, collate_fn=self._jax_collate_fn)
        self._test_dataloader = DataLoader(test_ds, batch_size=self._config.batch_size, shuffle=True, collate_fn=self._jax_collate_fn)


    def train_and_evaluate(self) -> None:
        """ Runs a training loop """
        raise NotImplementedError
    
    @staticmethod
    def _jax_collate_fn(batch):
        # Convert PyTorch tensors to JAX arrays for both inputs and labels
        jax_inputs = [jnp.array(item[0]) for item in batch]
        jax_labels = [jnp.array(item[1]) for item in batch]

        # Stack them to create batched JAX arrays
        batched_jax_inputs = jnp.stack(jax_inputs)
        batched_jax_labels = jnp.stack(jax_labels)

        return batched_jax_inputs, batched_jax_labels
    
    def _save_model(self, epoch: int, state: train_state.TrainState) -> None:
        """ Saves a model checkpoint """
        ckpt = {'model': state}
        save_args = orbax_utils.save_args_from_target(ckpt)
        self._checkpoint_manager.save(epoch, ckpt, save_kwargs={'save_args': save_args})

    def _load_model(self, epoch: int) -> train_state.TrainState:
        """ Loads a model checkpoint """
        ckpt = self._orbax_checkpoint.restore(os.path.join(self._ckpts_dir, epoch))
        return ckpt['model']


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
            dtype=jnp.float32,
        )


    def train_and_evaluate(self) -> None:
        """ Runs a training loop """

        """ Create a training state """
        rng = jax.random.PRNGKey(self._config.seed) 
        train_state = self._create_train_state(rng)


        best_loss = float("inf")
        for epoch in range(self._config.num_epochs):
            init_time = time.time() 
            rng, key = jax.random.split(rng) # creates a new subkey

            """ Training """
            # TODO: check input_rng and droputs
            state, metrics = self._train_step(train_state, key)
            
            """ Logging """
            test_metrics = self._evaluate_step(state, key)

            end_time = time.time()
            delta_time = end_time - init_time

            """ Logging """
            for key, value in metrics.items():
                self._summary_writer.add_scalar(f"Train/{key}", value, epoch)
            for key, value in test_metrics.items():
                self._summary_writer.add_scalar(f"Test/{key}", value, epoch)
            
            print(" *********************** ")
            print(f"Epoch: {epoch} \n"
                f"    Train Loss: {metrics['loss']:.4f} | Test Loss: {test_metrics['loss']:.4f} \n"
                f"    Train MAE: {metrics['mae']:.4f}   | Test MAE: {test_metrics['mae']:.4f} \n"
                f"    Train R2: {metrics['r2']:.4f}     | Test R2: {test_metrics['r2']:.4f}")
            print(f"    Elapsed epoch time: {delta_time} seconds")

            if test_metrics["loss"] < best_loss:
                best_loss = test_metrics["loss"]
                self._save_model(epoch, state)

        self._summary_writer.flush()


    def _create_train_state(self, rng: jax.random.PRNGKey) -> train_state.TrainState:
        """ Creates a training state """
        
        model = Transformer(self._flax_model_config)

        input_shape = (1, self._flax_model_config.max_seq_len, self._config.model_config.input_features)

        init_rng, dropout_rng = jax.random.split(rng)

        params = model.init({"dropout": dropout_rng, "params": init_rng}, jnp.ones((input_shape), dtype=jnp.float32))

        tx = optax.adamw(learning_rate=self._config.learning_rate)

        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    
    @staticmethod
    def _apply_model(state: train_state.TrainState, inputs: jnp.ndarray, targets: jnp.ndarray, 
                     rng: jax.random.PRNGKey, deterministic: bool) -> jnp.ndarray:

        def loss_fn(params):
            predictions = state.apply_fn(params, inputs, rngs={"dropout": rng}, deterministic=deterministic)
            loss = jnp.mean((predictions - targets) ** 2)  # MSE loss
            mae = jnp.mean(jnp.abs(predictions - targets))  # MAE loss
            r2 = 1 - jnp.sum((predictions - targets)**2) / jnp.sum((targets - jnp.mean(targets))**2)  # R2 loss
            return loss, (mae, r2)

        (loss, (mae, r2)), grad  = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        metrics = {"mae": mae, "r2": r2, "loss": loss}

        return grad, metrics
    
    @staticmethod
    @jax.jit
    def _update_model(state: train_state.TrainState, grads) -> train_state.TrainState:
        return state.apply_gradients(grads=grads)
    

    def _train_step(self, state: train_state.TrainState, rng: jax.random.PRNGKey) -> Tuple[train_state.TrainState, Dict]:
        """ Runs a training step """

        metrics = {"mae": [], "r2": [], "loss": []}
        for data in self._train_dataloader:
            inputs, targets = data
            grad, _metrics = self._apply_model(state, inputs, targets, rng, deterministic=False)
            state = self._update_model(state, grad)
            metrics["mae"].append(_metrics["mae"])
            metrics["r2"].append(_metrics["r2"])
            metrics["loss"].append(_metrics["loss"])

        metrics["mae"] = jnp.mean(jnp.array(metrics["mae"]))
        metrics["mae"] = jax.device_get(metrics["mae"])
        metrics["r2"] = jnp.mean(jnp.array(metrics["r2"]))
        metrics["r2"] = jax.device_get(metrics["r2"])
        metrics["loss"] = jnp.mean(jnp.array(metrics["loss"]))
        metrics["loss"] = jax.device_get(metrics["loss"])

        return state, metrics
        


    def _evaluate_step(self, state: train_state.TrainState, rng: jax.random.PRNGKey) -> Dict:
        """ Runs an evaluation step """

        metrics = {"mae": [], "r2": [], "loss": []}
        for data in self._test_dataloader:
            inputs, targets = data
            _, _metrics = self._apply_model(state, inputs, targets, rng=rng, deterministic=True)
            metrics["mae"].append(_metrics["mae"])
            metrics["r2"].append(_metrics["r2"])
            metrics["loss"].append(_metrics["loss"])

        metrics["mae"] = jnp.mean(jnp.array(metrics["mae"]))
        metrics["mae"] = jax.device_get(metrics["mae"])
        metrics["r2"] = jnp.mean(jnp.array(metrics["r2"]))
        metrics["r2"] = jax.device_get(metrics["r2"])
        metrics["loss"] = jnp.mean(jnp.array(metrics["loss"]))
        metrics["loss"] = jax.device_get(metrics["loss"])

        return metrics