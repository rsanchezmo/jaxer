import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import os
from typing import Tuple, Dict
from torch.utils.data import DataLoader
import time
import json

from jaxer.utils.plotter import plot_predictions
from jaxer.utils.losses import gaussian_negative_log_likelihood, mae, r2, rmse, mape, binary_cross_entropy
from jaxer.utils.early_stopper import EarlyStopper
from jaxer.utils.trainer_base import TrainerBase
from jaxer.utils.schedulers import create_warmup_cosine_schedule
from jaxer.model.flax_transformer import Transformer, TransformerConfig
from jaxer.utils.config import Config
from jaxer.utils.dataset import jax_collate_fn


class FlaxTrainer(TrainerBase):
    """Main class for training jaxer using flax, optax and jax

    :param config: training config for running an experiment
    :type config: Config
    """

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
            output_mode=self._config.model_config.output_mode,
            discrete_grid_levels=len(
                self._config.dataset_config.discrete_grid_levels) - 1 if self._config.dataset_config.discrete_grid_levels is not None else None,
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
        self._eval_model.apply(state.params, jnp.ones(
            (1, self._config.model_config.max_seq_len, self._config.model_config.input_features)))

    def train_and_evaluate(self) -> None:
        """ Runs a training loop """

        """ Create a training state """
        rng = jax.random.PRNGKey(self._config.seed)
        train_state_ = self._create_train_state(rng)

        self.logger.info("Warming up model...")
        self._warmup_eval(train_state_)

        best_loss = float("inf")
        self.logger.info("Starting training...")
        begin_time = time.time()

        if self._config.early_stopper > 0:
            early_stopper = EarlyStopper(self._config.early_stopper)
        else:
            early_stopper = None

        for epoch in range(self._config.num_epochs):
            init_time = time.time()
            rng, key = jax.random.split(rng)  # creates a new subkey

            """ Training """
            train_state_, metrics = self._train_epoch(train_state_, key)

            """ Logging """
            test_metrics = self._evaluate_step(train_state_)

            end_time = time.time()
            delta_time = end_time - init_time

            """ Logging """
            train_msg = f"Epoch: {epoch}/{self._config.num_epochs} \n" \
                        f"                  Learning Rate: {metrics['lr']:.2e} \n"
            test_msg = ''
            for key, value in metrics.items():
                self._summary_writer.add_scalar(f"train/{key}", value, epoch)

                if key == 'lr':
                    continue
                train_msg += f"                  Train {key}: {value:>8.4f} \n"
            for key, value in test_metrics.items():
                self._summary_writer.add_scalar(f"test/{key}", value, epoch)
                test_msg += f"                  Test {key}: {value:>8.4f} \n"

            time_msg = f"                  Epoch time: {delta_time:.2f} seconds\n" \
                       f"                  Total training time: {(end_time - begin_time) / 60:.2f} min\n"

            self.logger.info(train_msg + test_msg + time_msg)

            # TODO: get this back to test_metrics, but for intended purposes now I leave for training metrics
            should_stop = early_stopper(metrics["loss"]) if early_stopper is not None else False

            if should_stop:
                self.logger.warning(
                    f"Early stopping at epoch {epoch}! Did not improve for {self._config.early_stopper} epochs")
                break

            """ Save model """
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                self._save_best_model(epoch, train_state_, test_metrics)

        self.logger.info("Training finished!")
        self.best_model_test()
        self._summary_writer.flush()

    def _save_best_model(self, epoch: int, state: train_state.TrainState, metrics: Dict) -> None:
        self._best_model_state = state

        for key, value in metrics.items():
            metrics[key] = value.item()

        # save metrics in a json file
        metrics["ckpt"] = epoch
        metrics["subfolder"] = self._config.__str__()
        with open(os.path.join(self._log_dir, "best_model_train.json"), 'w') as f:
            f.write(json.dumps(metrics, indent=4))

        if self._config.save_weights:
            self._save_model(epoch, state)

    def _create_train_state(self, rng: jax.random.PRNGKey) -> train_state.TrainState:
        """ Creates a training state """

        model = Transformer(self._flax_model_config)

        input_shape = (1, self._flax_model_config.max_seq_len, self._config.model_config.input_features)

        key_dropout, key_params = jax.random.split(rng)

        self.logger.info(model.tabulate({"dropout": key_dropout, "params": key_params}, jnp.ones(input_shape),
                                        console_kwargs={'width': 120}))

        params = model.init({"dropout": key_dropout, "params": key_params}, jnp.ones(input_shape, dtype=jnp.float32))

        """ Create lr scheduler """
        steps_per_epoch = len(self._train_dataloader)
        if self._config.lr_mode == 'cosine':
            self._learning_rate_fn = create_warmup_cosine_schedule(learning_rate=self._config.learning_rate,
                                                                   warmup_epochs=self._config.warmup_epochs,
                                                                   num_epochs=self._config.num_epochs,
                                                                   steps_per_epoch=steps_per_epoch)
        elif self._config.lr_mode == 'linear':
            self._learning_rate_fn = optax.linear_schedule(init_value=self._config.learning_rate, end_value=0.0,
                                                           transition_steps=self._config.num_epochs * steps_per_epoch)
        else:
            raise NotImplementedError(f"Learning rate mode {self._config.lr_mode} not implemented")

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
    def _compute_metrics_discrete(predictions: jnp.ndarray, targets: jnp.ndarray) -> Dict:
        """ Computes metrics """

        accuracy = jnp.mean(jnp.equal(jnp.argmax(predictions, axis=-1), jnp.argmax(targets, axis=-1)))

        loss = binary_cross_entropy(y_pred=predictions, y_true=targets)

        return {"accuracy": accuracy, "loss": loss}

    @staticmethod
    @jax.jit
    def _model_train_step_mean(state: train_state.TrainState, inputs: jnp.ndarray, targets: jnp.ndarray,
                               rng: jax.random.PRNGKey):

        def loss_fn(params):
            predictions = state.apply_fn(params, inputs, rngs={"dropout": rng})

            means = predictions

            mae_ = mae(means, targets)
            r2_ = r2(means, targets)
            rmse_ = rmse(means, targets)
            mape_ = mape(means, targets)

            loss = rmse_

            return loss, (mae_, r2_, rmse_, mape_)

        (loss, (mae_, r2_, rmse_, mape_)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = {"mae": mae_, "r2": r2_, "loss": loss, "rmse": rmse_, "mape": mape_}
        return state, metrics

    @staticmethod
    @jax.jit
    def _model_train_step_dist(state: train_state.TrainState, inputs: jnp.ndarray, targets: jnp.ndarray,
                               rng: jax.random.PRNGKey):

        def loss_fn(params):
            predictions = state.apply_fn(params, inputs, rngs={"dropout": rng})

            means, variances = predictions

            mae_ = mae(means, targets)
            r2_ = r2(means, targets)
            rmse_ = rmse(means, targets)
            mape_ = mape(means, targets)

            loss = gaussian_negative_log_likelihood(means, variances, targets)

            return loss, (mae_, r2_, rmse_, mape_)

        (loss, (mae_, r2_, rmse_, mape_)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = {"mae": mae_, "r2": r2_, "loss": loss, "rmse": rmse_, "mape": mape_}
        return state, metrics

    @staticmethod
    @jax.jit
    def _model_train_step_discrete(state: train_state.TrainState, inputs: jnp.ndarray, targets: jnp.ndarray,
                                   rng: jax.random.PRNGKey):

        def loss_fn(params):
            predictions = state.apply_fn(params, inputs, rngs={"dropout": rng})

            loss = binary_cross_entropy(y_pred=predictions, y_true=targets)

            accuracy = jnp.mean(jnp.equal(jnp.argmax(predictions, axis=-1), jnp.argmax(targets, axis=-1)))

            return loss, accuracy

        (loss, (accuracy)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = {"accuracy": accuracy, "loss": loss}
        return state, metrics

    def _model_eval_step(self, state: train_state.TrainState, inputs: jnp.ndarray, targets: jnp.ndarray,
                         config: TransformerConfig) -> Tuple[train_state.TrainState, Dict]:

        predictions = self._eval_model.apply(state.params, inputs)

        if config.output_mode == 'distribution':
            metrics = self._compute_metrics_dist(predictions, targets)
        elif config.output_mode == 'mean':
            metrics = self._compute_metrics_mean(predictions, targets)
        elif config.output_mode == 'discrete_grid':
            metrics = self._compute_metrics_discrete(predictions, targets)
        else:
            raise NotImplementedError(f"Output mode {config.output_mode} not implemented")

        return state, metrics

    def _train_epoch(self, state: train_state.TrainState, rng: jax.random.PRNGKey) \
            -> Tuple[train_state.TrainState, Dict]:
        """ Runs a training step """

        metrics = {}
        lr = None

        for data in self._train_dataloader:
            inputs, targets, _, _ = tuple(data)
            step = state.step
            lr = self._learning_rate_fn(step)
            if self._flax_model_config.output_mode == 'distribution':
                state, _metrics = self._model_train_step_dist(state, inputs, targets, rng)
            elif self._flax_model_config.output_mode == 'mean':
                state, _metrics = self._model_train_step_mean(state, inputs, targets, rng)
            elif self._flax_model_config.output_mode == 'discrete_grid':
                state, _metrics = self._model_train_step_discrete(state, inputs, targets, rng)
            else:
                raise NotImplementedError(f"Output mode {self._flax_model_config.output_mode} not implemented")

            for key, value in _metrics.items():
                metric_list = metrics.get(key, [])
                metric_list.append(value)
                metrics[key] = metric_list

        for key, value in metrics.items():
            metrics[key] = jnp.mean(jnp.array(value))
            metrics[key] = jax.device_get(metrics[key])

        metrics["lr"] = jax.device_get(lr)

        return state, metrics

    def _evaluate_step(self, state: train_state.TrainState) -> Dict:
        """ Runs an evaluation step """

        metrics = {}

        for data in self._test_dataloader:
            inputs, targets, _, _ = data
            _, _metrics = self._model_eval_step(state, inputs, targets, config=self._flax_model_config_eval)

            for key, value in _metrics.items():
                metric_list = metrics.get(key, [])
                metric_list.append(value)
                metrics[key] = metric_list

        for key, value in metrics.items():
            metrics[key] = jnp.mean(jnp.array(value))
            metrics[key] = jax.device_get(metrics[key])

        return metrics

    def best_model_test(self, max_seqs: int = 20):
        """ Generate images from the test set of the best model """

        test_dataloader = DataLoader(self._test_ds, batch_size=1, collate_fn=jax_collate_fn, shuffle=True)
        save_folder = os.path.join(self._log_dir, "best_model_test", self._config.__str__())
        os.makedirs(save_folder, exist_ok=True)

        counter = 0
        for i, data in enumerate(test_dataloader):
            inputs, targets, normalizer, _ = data
            output = self._eval_model.apply(self._best_model_state.params, inputs)
            plot_predictions(input=inputs.squeeze(0),
                             y_true=targets.squeeze(0),
                             output=output,
                             name=str(i),
                             foldername=save_folder,
                             normalizer=normalizer[0],
                             output_mode=self._config.model_config.output_mode,
                             discrete_grid_levels=self._config.dataset_config.discrete_grid_levels)
            counter += 1
            if counter == max_seqs:
                break
