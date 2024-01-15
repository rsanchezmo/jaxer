from jaxer.model.flax_transformer import Transformer, TransformerConfig
from jaxer.utils.config import Config
from jaxer.utils.dataset import Dataset, jax_collate_fn
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
from jaxer.utils.plotter import plot_predictions
from jaxer.utils.logger import Logger
from jaxer.utils.losses import gaussian_negative_log_likelihood, mae, r2, rmse, mape
from jaxer.utils.early_stopper import EarlyStopper


def create_warmup_cosine_schedule(learning_rate: float, warmup_epochs: int, num_epochs: int, steps_per_epoch: int) -> optax.Schedule:
    """Creates learning rate cosine schedule."""
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


class TrainerBase:
    def __init__(self, config: Config) -> None:
        """ Trainer Base Class: All trainers should inherit from this, FLAX or PyTorch [in the future] """
        self._config = config

        """ Training logs """
        self._log_dir = os.path.join(os.getcwd(), self._config.log_dir, self._config.experiment_name)
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(os.path.join(self._log_dir, "tensorboard"), exist_ok=True)
        # get num of experiments inside the folder
        self._summary_writer = SummaryWriter(os.path.join(self._log_dir, "tensorboard", self._config_to_str(self._config)))

        """ Checkpoints """
        self._ckpts_dir = os.path.join(self._log_dir, "ckpt", self._config_to_str(self._config))
        os.makedirs(self._ckpts_dir, exist_ok=True)
        self._orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(create=True, max_to_keep=5)
        self._checkpoint_manager = orbax.checkpoint.CheckpointManager(
            self._ckpts_dir, self._orbax_checkpointer, options)
        

        """ Dataloaders """
        dataset = Dataset(self._config.dataset_path, self._config.model_config.max_seq_len, norm_mode=self._config.normalizer_mode,
                          initial_date=self._config.initial_date)
        self._train_ds, self._test_ds = dataset.get_train_test_split(test_size=self._config.test_split)
        self._train_dataloader = DataLoader(self._train_ds, batch_size=self._config.batch_size, shuffle=True, collate_fn=jax_collate_fn)
        self._test_dataloader = DataLoader(self._test_ds, batch_size=self._config.batch_size, shuffle=True, collate_fn=jax_collate_fn)


        os.makedirs(os.path.join(self._log_dir, 'configs', self._config_to_str(self._config)), exist_ok=True)

        """ Save config """
        self._config.save_config(os.path.join(self._log_dir, 'configs', self._config_to_str(self._config), "config.json"))

        """ Best model state: eval purposes """
        self._best_model_state: Optional[train_state.TrainState] = None

        self.logger = Logger("Trainer")


    @staticmethod
    def to_binary(val: bool):
        return 1 if val else 0

    def _config_to_str(self, config: Config) -> str:
        """ Converts a config to a string """
        return f"{config.learning_rate}_{config.lr_mode}_bs_{config.batch_size}_ep_{config.num_epochs}_wmp_{config.warmup_epochs}_seed_{config.seed}_" \
                f"dmodel_{config.model_config.d_model}_nlayers_{config.model_config.num_layers}_nhdl_{config.model_config.head_layers}_" \
                f"nhds_{config.model_config.n_heads}_dimff_{config.model_config.dim_feedforward}_drpt_{config.model_config.dropout}_" \
                f"maxlen_{config.model_config.max_seq_len}_infeat_{config.model_config.input_features}_flat_{self.to_binary(config.model_config.flatten_encoder_output)}_" \
                f"feblk_{config.model_config.fe_blocks}_t2v_{self.to_binary(config.model_config.use_time2vec)}_{config.normalizer_mode}_" \
                f"ds_{config.test_split}_{config.initial_date}_" \
                f"outdst_{self.to_binary(config.model_config.output_distribution)}_" \
                f"reshd_{self.to_binary(config.model_config.use_resblocks_in_head)}_" \
                f"resfe_{self.to_binary(config.model_config.use_resblocks_in_fe)}_" \
                f"avout_{self.to_binary(config.model_config.average_encoder_output)}_nrmpre_{self.to_binary(config.model_config.norm_encoder_prev)}"


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
            head_layers=self._config.model_config.head_layers,
            n_heads=self._config.model_config.n_heads,
            dim_feedforward=self._config.model_config.dim_feedforward,
            dropout=self._config.model_config.dropout,
            max_seq_len=self._config.model_config.max_seq_len,
            dtype=jnp.float32,
            deterministic=False,
            input_features=self._config.model_config.input_features,
            flatten_encoder_output=self._config.model_config.flatten_encoder_output,
            fe_blocks=self._config.model_config.fe_blocks,
            use_time2vec=self._config.model_config.use_time2vec,
            output_distribution=self._config.model_config.output_distribution,
            use_resblocks_in_head=self._config.model_config.use_resblocks_in_head,
            use_resblocks_in_fe=self._config.model_config.use_resblocks_in_fe,
            average_encoder_output=self._config.model_config.average_encoder_output,
            norm_encoder_prev=self._config.model_config.norm_encoder_prev
        )

        self._flax_model_config_eval = self._flax_model_config.replace(deterministic=True)

        self._learning_rate_fn = None


    def _warmup_eval(self, state: train_state.TrainState) -> None:
        """ Warmup models """
        self._eval_model = Transformer(self._flax_model_config_eval)
        self._eval_model.apply(state.params, jnp.ones((1, self._config.model_config.max_seq_len, self._config.model_config.input_features)))


    def train_and_evaluate(self) -> None:
        """ Runs a training loop """

        """ Create a training state """
        rng = jax.random.PRNGKey(self._config.seed) 
        train_state = self._create_train_state(rng)

        self.logger.info("Warming up model...")
        self._warmup_eval(train_state)

        best_loss = float("inf")
        self.logger.info("Starting training...")
        begin_time = time.time()

        if self._config.early_stopper > 0:
            early_stopper = EarlyStopper(self._config.early_stopper)
        else:
            early_stopper = None

        for epoch in range(self._config.num_epochs):
            init_time = time.time() 
            rng, key = jax.random.split(rng) # creates a new subkey

            """ Training """
            train_state, metrics = self._train_epoch(train_state, key)
            
            """ Logging """
            test_metrics = self._evaluate_step(train_state)

            end_time = time.time()
            delta_time = end_time - init_time

            """ Logging """
            for key, value in metrics.items():
                self._summary_writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in test_metrics.items():
                self._summary_writer.add_scalar(f"test/{key}", value, epoch)
            
            self.logger.info(f" Epoch: {epoch}/{self._config.num_epochs} \n"
                f"                  Learning Rate: {metrics['lr']:.2e} \n"
                f"                  Train Loss:  {metrics['loss']:>8.4f}    Test Loss: {test_metrics['loss']:>8.4f} \n"
                f"                  Train MAE:   {metrics['mae']:>8.4f}    Test MAE:  {test_metrics['mae']:>8.4f} \n"
                f"                  Train R2:    {metrics['r2']:>8.4f}    Test R2:   {test_metrics['r2']:>8.4f}\n"
                f"                  Train RMSE:  {metrics['rmse']:>8.4f}    Test RMSE: {test_metrics['rmse']:>8.4f}\n"
                f"                  Train MAPE:  {metrics['mape']:>8.4f} %  Test MAPE: {test_metrics['mape']:>8.4f} % \n"
                f"                  Epoch time: {delta_time:.2f} seconds\n"
                f"                  Total training time: {(end_time - begin_time)/60:.2f} min\n")


            # TODO: get this back to test_metrics, but for intended purposes no I leave for training metrics
            should_stop = early_stopper(metrics["loss"]) if early_stopper is not None else False

            if should_stop:
                self.logger.warning(f"Early stopping at epoch {epoch}! Did not improve for {self._config.early_stopper} epochs")               
                break

            """ Save model """
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                self._save_best_model(epoch, train_state, test_metrics)


        self.logger.info("Training finished!")
        self.best_model_test()
        self._summary_writer.flush()


    def _save_best_model(self, epoch: int, state: train_state.TrainState, metrics: Dict) -> None:
        self._best_model_state = state

        for key, value in metrics.items():
            metrics[key] = value.item()

        # save metrics in a json file
        metrics["ckpt"] = epoch
        metrics["subfolder"] = self._config_to_str(self._config)
        with open(os.path.join(self._log_dir, "best_model_train.json"), 'w') as f:
            f.write(json.dumps(metrics, indent=4))

        if self._config.save_weights:
            self._save_model(epoch, state)



    def _create_train_state(self, rng: jax.random.PRNGKey) -> train_state.TrainState:
        """ Creates a training state """

        """ Instance of the model. However, we could do this anywhere, as the state is saved outside of the model """
        model = Transformer(self._flax_model_config) 


        input_shape = (1, self._flax_model_config.max_seq_len, self._config.model_config.input_features)

        key_dropout, key_params = jax.random.split(rng)

        self.logger.info(model.tabulate({"dropout": key_dropout, "params": key_params}, jnp.ones(input_shape), console_kwargs={'width': 120}))

        """ call the init func, that returns a pytree with the model params. Have to initialize the dropouts too """
        params =  model.init({"dropout": key_dropout, "params": key_params}, jnp.ones((input_shape), dtype=jnp.float32))

        """ Create lr scheduler """
        steps_per_epoch = len(self._train_dataloader)
        if self._config.lr_mode == 'cosine':
            self._learning_rate_fn = create_warmup_cosine_schedule(learning_rate=self._config.learning_rate, warmup_epochs=self._config.warmup_epochs,
                                                                num_epochs=self._config.num_epochs, steps_per_epoch=steps_per_epoch)
        elif self._config.lr_mode == 'linear':
            self._learning_rate_fn = optax.linear_schedule(init_value=self._config.learning_rate, end_value=0.0,
                                                        transition_steps=self._config.num_epochs * steps_per_epoch)
        else:
            raise NotImplementedError(f"Learning rate mode {self._config.lr_mode} not implemented")

        """ Create optimizer """
        tx = optax.adamw(self._learning_rate_fn)

        """ wrap params, apply_fn and tx in a TrainState, to not keep passing them around """
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    @staticmethod
    @jax.jit
    def _compute_metrics_dist(predictions: jnp.ndarray, targets: jnp.ndarray) -> Dict:
        """ Computes metrics """

        means, variances = predictions

        mae_ = mae(means, targets)
        rmse_ = rmse(means, targets)
        r2_ = r2(means, targets)
        mape_ = mape(means, targets)
        
        loss = gaussian_negative_log_likelihood(means, variances, targets)

        return {"mae": mae_, "r2": r2_, "loss": loss, "rmse": rmse_, "mape": mape_}
    
    @staticmethod
    @jax.jit
    def _compute_metrics_mean(predictions: jnp.ndarray, targets: jnp.ndarray) -> Dict:
        """ Computes metrics """
        means = predictions

        mae_ = mae(means, targets)
        rmse_ = rmse(means, targets)
        r2_ = r2(means, targets)
        mape_ = mape(means, targets)

        loss = rmse_

        return {"mae": mae_, "r2": r2_, "loss": loss, "rmse": rmse_, "mape": mape_}


    @staticmethod
    @jax.jit
    def _model_train_step_mean(state: train_state.TrainState, inputs: jnp.ndarray, targets: jnp.ndarray, 
                     rng: jax.random.PRNGKey) -> jnp.ndarray:

        def loss_fn(params):
            predictions = state.apply_fn(params, inputs, rngs={"dropout": rng})

            means = predictions

            mae_ = mae(means, targets)
            r2_ = r2(means, targets)
            rmse_ = rmse(means, targets)
            mape_ = mape(means, targets)

            loss = rmse_

            return loss, (mae_, r2_, rmse_, mape_)

        (loss, (mae_, r2_, rmse_, mape_)), grads  = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = {"mae": mae_, "r2": r2_, "loss": loss, "rmse": rmse_, "mape": mape_}
        return state, metrics
    
    @staticmethod
    @jax.jit
    def _model_train_step_dist(state: train_state.TrainState, inputs: jnp.ndarray, targets: jnp.ndarray,
                        rng: jax.random.PRNGKey) -> jnp.ndarray:
        
        def loss_fn(params):
            predictions = state.apply_fn(params, inputs, rngs={"dropout": rng})

            means, variances = predictions

            mae_ = mae(means, targets)
            r2_ = r2(means, targets)
            rmse_ = rmse(means, targets)
            mape_ = mape(means, targets)

            loss = gaussian_negative_log_likelihood(means, variances, targets)

            return loss, (mae_, r2_, rmse_, mape_)
        
        (loss, (mae_, r2_, rmse_, mape_)), grads  = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = {"mae": mae_, "r2": r2_, "loss": loss, "rmse": rmse_, "mape": mape_}
        return state, metrics
    

    def _model_eval_step(self, state: train_state.TrainState, inputs: jnp.ndarray, targets: jnp.ndarray, 
                     config: TransformerConfig) -> jnp.ndarray:
        
        predictions = self._eval_model.apply(state.params, inputs)

        if config.output_distribution:
            metrics = self._compute_metrics_dist(predictions, targets)
        else:
            metrics = self._compute_metrics_mean(predictions, targets)

        return state, metrics  

    def _train_epoch(self, state: train_state.TrainState, rng: jax.random.PRNGKey) -> Tuple[train_state.TrainState, Dict]:
        """ Runs a training step """

        metrics = {"mae": [], "r2": [], "loss": [], "rmse": [], "mape": []}
        for data in self._train_dataloader:
            inputs, targets, _, _ = data
            step = state.step
            lr = self._learning_rate_fn(step)
            if self._flax_model_config.output_distribution:
                state, _metrics = self._model_train_step_dist(state, inputs, targets, rng)
            else:
                state, _metrics = self._model_train_step_mean(state, inputs, targets, rng)
            metrics["mae"].append(_metrics["mae"])
            metrics["r2"].append(_metrics["r2"])
            metrics["rmse"].append(_metrics["rmse"])
            metrics["loss"].append(_metrics["loss"])
            metrics["mape"].append(_metrics["mape"])

        metrics["lr"] = jax.device_get(lr)
        metrics["mae"] = jnp.mean(jnp.array(metrics["mae"]))
        metrics["mae"] = jax.device_get(metrics["mae"])
        metrics["r2"] = jnp.mean(jnp.array(metrics["r2"]))
        metrics["r2"] = jax.device_get(metrics["r2"])
        metrics["rmse"] = jnp.mean(jnp.array(metrics["rmse"]))
        metrics["rmse"] = jax.device_get(metrics["rmse"])
        metrics["loss"] = jnp.mean(jnp.array(metrics["loss"]))
        metrics["loss"] = jax.device_get(metrics["loss"])
        metrics["mape"] = jnp.mean(jnp.array(metrics["mape"]))
        metrics["mape"] = jax.device_get(metrics["mape"])

        return state, metrics

    def _evaluate_step(self, state: train_state.TrainState) -> Dict:
        """ Runs an evaluation step """

        metrics = {"mae": [], "r2": [], "loss": [], "rmse": [], "mape": []}
        for data in self._test_dataloader:
            inputs, targets, _, _ = data
            _, _metrics = self._model_eval_step(state, inputs, targets, config=self._flax_model_config_eval)
            metrics["mae"].append(_metrics["mae"])
            metrics["r2"].append(_metrics["r2"])
            metrics["rmse"].append(_metrics["rmse"])
            metrics["loss"].append(_metrics["loss"])
            metrics["mape"].append(_metrics["mape"])

        metrics["mae"] = jnp.mean(jnp.array(metrics["mae"]))
        metrics["mae"] = jax.device_get(metrics["mae"])
        metrics["r2"] = jnp.mean(jnp.array(metrics["r2"]))
        metrics["r2"] = jax.device_get(metrics["r2"])
        metrics["rmse"] = jnp.mean(jnp.array(metrics["rmse"]))
        metrics["rmse"] = jax.device_get(metrics["rmse"])
        metrics["loss"] = jnp.mean(jnp.array(metrics["loss"]))
        metrics["loss"] = jax.device_get(metrics["loss"])
        metrics["mape"] = jnp.mean(jnp.array(metrics["mape"]))
        metrics["mape"] = jax.device_get(metrics["mape"])

        return metrics
    
    def best_model_test(self, max_seqs: int = 20):
        """ Generate images from the test set of the best model """

        # get test dataloader but with batch == 1
        test_dataloader = DataLoader(self._test_ds, batch_size=1, collate_fn=jax_collate_fn, shuffle=True)
        save_folder = os.path.join(self._log_dir, "best_model_test", self._config_to_str(self._config))
        os.makedirs(save_folder, exist_ok=True)

        counter = 0
        for i, data in enumerate(test_dataloader):
            inputs, targets, normalizer, _ = data
            output = self._eval_model.apply(self._best_model_state.params, inputs)
            plot_predictions(inputs.squeeze(0), targets.squeeze(0), output, i, save_folder, normalizer=normalizer[0])
            counter += 1
            if counter == max_seqs:
                break

