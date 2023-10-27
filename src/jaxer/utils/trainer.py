from ..model.flax_transformer import Transformer, TransformerConfig
from ..configs.config import Config
from .dataset import Dataset, jax_collate_fn
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
import json
from ..utils.plotter import plot_predictions


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
        self._train_ds, self._test_ds = dataset.get_train_test_split(test_size=self._config.test_split)
        self._train_dataloader = DataLoader(self._train_ds, batch_size=self._config.batch_size, shuffle=True, collate_fn=jax_collate_fn)
        self._test_dataloader = DataLoader(self._test_ds, batch_size=self._config.batch_size, shuffle=True, collate_fn=jax_collate_fn)


        """ Save config """
        self._config.save_config(os.path.join(self._log_dir, "config.json"))

        """ Best model state: eval purposes """
        self._best_model_state: Optional[train_state.TrainState] = None


    def train_and_evaluate(self) -> None:
        """ Runs a training loop """
        raise NotImplementedError
    
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
            deterministic=False
        )

        self._flax_model_config_eval = self._flax_model_config.replace(deterministic=True)


    def _warmup_eval(self, state: train_state.TrainState) -> None:
        """ Warmup models """
        self._eval_model = Transformer(self._flax_model_config_eval)
        self._eval_model.apply(state.params, jnp.ones((1, self._config.model_config.max_seq_len, self._config.model_config.input_features)))


    def train_and_evaluate(self) -> None:
        """ Runs a training loop """

        """ Create a training state """
        rng = jax.random.PRNGKey(self._config.seed) 
        train_state = self._create_train_state(rng)

        self._warmup_eval(train_state)


        best_loss = float("inf")
        for epoch in range(self._config.num_epochs):
            init_time = time.time() 
            rng, key = jax.random.split(rng) # creates a new subkey

            """ Training """
            # TODO: check input_rng and droputs
            state, metrics = self._train_step(train_state, key)
            
            """ Logging """
            test_metrics = self._evaluate_step(state)

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
                self._save_best_model(epoch, state, test_metrics)
                # self.best_model_test()


        self.best_model_test()
        self._summary_writer.flush()


    def _save_best_model(self, epoch: int, state: train_state.TrainState, metrics: Dict) -> None:
        self._save_model(epoch, state)
        self._best_model_state = state

        for key, value in metrics.items():
            metrics[key] = value.item()

        # save metrics in a json file
        metrics["ckpt"] = epoch
        with open(os.path.join(self._log_dir, "best_model_train.json"), 'w') as f:
            f.write(json.dumps(metrics, indent=4))


    def _create_train_state(self, rng: jax.random.PRNGKey) -> train_state.TrainState:
        """ Creates a training state """

        """ Instance of the model. However, we could do this anywhere, as the state is saved outside of the model """
        model = Transformer(self._flax_model_config) 

        input_shape = (1, self._flax_model_config.max_seq_len, self._config.model_config.input_features)

        key_dropout, key_params = jax.random.split(rng)

        """ call the init func, that returns a pytree with the model params. Have to initialize the dropouts too """
        params =  model.init({"dropout": key_dropout, "params": key_params}, jnp.ones((input_shape), dtype=jnp.float32))

        """ Create optimizer """
        tx = optax.adamw(learning_rate=self._config.learning_rate)

        """ wrap params, apply_fn and tx in a TrainState, to not keep passing them around """
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    

    @staticmethod
    @jax.jit
    def _compute_metrics(predictions: jnp.ndarray, targets: jnp.ndarray) -> Dict:
        """ Computes metrics """
        loss = jnp.mean((predictions - targets) ** 2)
        mae = jnp.mean(jnp.abs(predictions - targets))
        r2 = 1 - jnp.sum((predictions - targets)**2) / jnp.sum((targets - jnp.mean(targets))**2)
        return {"mae": mae, "r2": r2, "loss": loss}

    @staticmethod
    @jax.jit
    def _model_train_step(state: train_state.TrainState, inputs: jnp.ndarray, targets: jnp.ndarray, 
                     rng: jax.random.PRNGKey) -> jnp.ndarray:

        def loss_fn(params):
            predictions = state.apply_fn(params, inputs, rngs={"dropout": rng})
            loss = jnp.mean((predictions - targets) ** 2)  # MSE loss
            mae = jnp.mean(jnp.abs(predictions - targets))  # MAE loss
            r2 = 1 - jnp.sum((predictions - targets)**2) / jnp.sum((targets - jnp.mean(targets))**2)  # R2 loss
            return loss, (mae, r2)

        (loss, (mae, r2)), grads  = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = {"mae": mae, "r2": r2, "loss": loss}
        return state, metrics
    

    def _model_eval_step(self, state: train_state.TrainState, inputs: jnp.ndarray, targets: jnp.ndarray, 
                     config: TransformerConfig) -> jnp.ndarray:
        
        predictions = self._eval_model.apply(state.params, inputs)

        metrics = self._compute_metrics(predictions, targets)

        return state, metrics  

    def _train_step(self, state: train_state.TrainState, rng: jax.random.PRNGKey) -> Tuple[train_state.TrainState, Dict]:
        """ Runs a training step """

        metrics = {"mae": [], "r2": [], "loss": []}
        for data in self._train_dataloader:
            inputs, targets, _ = data
            state, _metrics = self._model_train_step(state, inputs, targets, rng)
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

    def _evaluate_step(self, state: train_state.TrainState) -> Dict:
        """ Runs an evaluation step """

        metrics = {"mae": [], "r2": [], "loss": []}
        for data in self._test_dataloader:
            inputs, targets, _ = data
            _, _metrics = self._model_eval_step(state, inputs, targets, config=self._flax_model_config_eval)
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
    
    def best_model_test(self, max_seqs: int = 10):
        """ Generate images from the test set of the best model """

        # get test dataloader but with batch == 1
        test_dataloader = DataLoader(self._test_ds, batch_size=1, collate_fn=self._jax_collate_fn)
        save_folder = os.path.join(self._log_dir, "best_model_test")
        os.makedirs(save_folder, exist_ok=True)

        counter = 0
        for i, data in enumerate(test_dataloader):
            inputs, targets, normalizer = data
            predictions = self._eval_model.apply(self._best_model_state.params, inputs)
            plot_predictions(inputs.squeeze(0), targets.squeeze(0), predictions.squeeze(0), i, save_folder, normalizer=normalizer[0])
            counter += 1
            if counter == max_seqs:
                break

