import jax
import jax.numpy as jnp
import optax
import os
from typing import Tuple, Dict, Optional, Any, List
from torch.utils.data import DataLoader
import time
import json
import orbax
import numpy as np
from jaxer.utils.plotter import plot_predictions
from jaxer.utils.losses import (gaussian_negative_log_likelihood, mae,
                                r2, rmse, mape, binary_cross_entropy, acc_dir, acc_dir_discrete)
from jaxer.utils.early_stopper import EarlyStopper
from jaxer.run.trainer_base import TrainerBase
from jaxer.utils.schedulers import create_warmup_cosine_schedule
from jaxer.model.flax_transformer import Transformer, TransformerConfig
from jaxer.config.experiment_config import ExperimentConfig
from jaxer.config.model_config import ModelConfig
from jaxer.utils.dataset import jax_collate_fn, Dataset, denormalize
from jaxer.utils.synthetic_dataset import SyntheticDataset
from flax.training import orbax_utils, train_state


class FlaxTrainer(TrainerBase):
    """Trainer class for training jaxer using flax, optax and jax

    :param config: training config for running an experiment
    :type config: Config
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)

        """ check pretrained model """
        self._pretrained_params = None
        if self._config.pretrained_model is not None:
            experiment_folder, experiment_subfolder, best_model = self._config.pretrained_model
            pretrained_config = ExperimentConfig.load_config(
                os.path.join(experiment_folder, 'configs', experiment_subfolder, 'config.json'))
            pretrained_model_config = ModelConfig.from_dict(pretrained_config.model_config)

            if pretrained_model_config != self._config.model_config:
                raise ValueError(
                    f"Pretrained model config {pretrained_model_config} does not match current model config {self._config.model_config}")

            ckpt_path = os.path.join(experiment_folder, "ckpt", experiment_subfolder, best_model, 'default')
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            self._pretrained_params = orbax_checkpointer.restore(ckpt_path)["model"]["params"]

        """ Checkpoints """
        self._ckpts_dir = os.path.join(self._log_dir, "ckpt", self._config.__str__())
        os.makedirs(self._ckpts_dir, exist_ok=True)
        self._orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(create=True, max_to_keep=5)
        self._checkpoint_manager = orbax.checkpoint.CheckpointManager(
            str(self._ckpts_dir), self._orbax_checkpointer, options)

        """ Dataloaders """
        self._dataset_real = None
        self._dataset_synth = None
        self._train_dataloader_real = None
        self._test_dataloader_real = None
        self._train_dataloader_synth = None
        self._test_dataloader_synth = None

        batch_size_real = self._config.batch_size
        batch_size_synth = self._config.batch_size
        if self._config.dataset_mode == 'both':
            batch_size_real = int(0.25 * self._config.batch_size)
            batch_size_synth = self._config.batch_size - batch_size_real

        if self._config.dataset_mode in ['real', 'both']:
            self._dataset_real = Dataset(dataset_config=self._config.dataset_config)

            self._train_ds, self._test_ds = self._dataset_real.get_train_test_split(test_size=self._config.test_split,
                                                                                    test_tickers=self._config.test_tickers)

            self._train_dataloader_real = DataLoader(self._train_ds, batch_size=batch_size_real, shuffle=True,
                                                     collate_fn=jax_collate_fn)
            # leave the batch size on test at the maximum as if both mode, then will only test on real data
            self._test_dataloader_real = DataLoader(self._test_ds, batch_size=self._config.batch_size, shuffle=True,
                                                    collate_fn=jax_collate_fn)

        if self._config.dataset_mode in ['synthetic', 'both']:
            self._dataset_synth = SyntheticDataset(config=self._config.synthetic_dataset_config)
            self._train_dataloader_synth = self._dataset_synth.generator(batch_size=batch_size_synth,
                                                                         seed=self._config.seed)
            self._test_dataloader_synth = self._dataset_synth.generator(batch_size=self._config.batch_size,
                                                                        seed=self._config.seed + 1)

        discrete_grid_levels = None
        if self._config.dataset_mode == 'real' and self._config.dataset_config.discrete_grid_levels is not None:
            discrete_grid_levels = len(self._config.dataset_config.discrete_grid_levels) - 1

        """ Best model state: eval purposes """
        self._best_model_state: Optional[train_state.TrainState] = None

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
            flatten_encoder_output=self._config.model_config.flatten_encoder_output,
            fe_blocks=self._config.model_config.fe_blocks,
            use_time2vec=self._config.model_config.use_time2vec,
            output_mode=self._config.model_config.output_mode,
            discrete_grid_levels=discrete_grid_levels,
            use_resblocks_in_head=self._config.model_config.use_resblocks_in_head,
            use_resblocks_in_fe=self._config.model_config.use_resblocks_in_fe,
            average_encoder_output=self._config.model_config.average_encoder_output,
            norm_encoder_prev=self._config.model_config.norm_encoder_prev
        )

        self._flax_model_config_eval = self._flax_model_config.replace(deterministic=True)

        self._learning_rate_fn = None

    def _save_model(self, epoch: int, state: train_state.TrainState) -> None:
        """ Saves a model checkpoint

        :param epoch: the epoch number
        :type epoch: int

        :param state: the model state
        :type state: train_state.TrainState
        """

        ckpt = {'model': state}
        save_args = orbax_utils.save_args_from_target(ckpt)
        self._checkpoint_manager.save(epoch, ckpt, save_kwargs={'save_args': save_args})

    def _load_model(self, epoch: int) -> Any:
        """ Loads a model checkpoint

        :param epoch: the epoch number
        :type epoch: int

        :return: the model state
        :rtype: train_state.TrainState
        """
        ckpt = self._orbax_checkpointer.restore(os.path.join(str(self._ckpts_dir), str(epoch)))
        return ckpt['model']

    def _warmup_eval(self, state: train_state.TrainState) -> None:
        """ Warmup models """
        self._eval_model = Transformer(self._flax_model_config_eval)
        dataset = self._dataset_real if self._dataset_real is not None else self._dataset_synth
        self._eval_model.apply(state.params, dataset.get_random_input())

    def train_and_evaluate(self) -> None:
        """ Runs the training loop with evaluation """

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
        # self._best_model_test()
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

        key_dropout, key_params = jax.random.split(rng)

        dataset = self._dataset_real if self._dataset_real is not None else self._dataset_synth
        self.logger.info(model.tabulate({"dropout": key_dropout, "params": key_params},
                                        dataset.get_random_input(),
                                        console_kwargs={'width': 120}))

        if self._pretrained_params is not None:
            params = self._pretrained_params
        else:
            params = model.init({"dropout": key_dropout, "params": key_params}, dataset.get_random_input())

        """ Create lr scheduler """
        steps_per_epoch = 1
        if self._config.dataset_mode in ['real', 'both']:
            steps_per_epoch = len(self._train_dataloader_real)
        elif self._config.dataset_mode == 'synthetic':
            steps_per_epoch = self._config.steps_per_epoch

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

        tx = optax.adamw(self._learning_rate_fn, weight_decay=1e-3)

        """ wrap params, apply_fn and tx in a TrainState, to not keep passing them around """
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @staticmethod
    @jax.jit
    def _compute_metrics_dist(predictions: jnp.ndarray, targets: jnp.ndarray, last_price: jnp.ndarray) -> Dict:
        """ Computes metrics """

        means, stds = predictions

        mae_ = mae(means, targets)
        rmse_ = rmse(means, targets)
        r2_ = r2(means, targets)
        mape_ = mape(means, targets)
        acc_dir_ = acc_dir(means, targets, last_price)

        return {"mae": mae_, "r2": r2_, "rmse": rmse_, "mape": mape_, "acc_dir": acc_dir_}

    @staticmethod
    @jax.jit
    def _compute_metrics_mean(predictions: jnp.ndarray, targets: jnp.ndarray, last_price: jnp.ndarray) -> Dict:
        """ Computes metrics """
        means = predictions

        mae_ = mae(means, targets)
        rmse_ = rmse(means, targets)
        r2_ = r2(means, targets)
        mape_ = mape(means, targets)
        acc_dir_ = acc_dir(means, targets, last_price)

        return {"mae": mae_, "r2": r2_, "rmse": rmse_, "mape": mape_, "acc_dir": acc_dir_}

    @staticmethod
    @jax.jit
    def _compute_metrics_discrete(predictions: jnp.ndarray, targets: jnp.ndarray) -> Dict:
        """ Computes metrics """

        accuracy = jnp.mean(jnp.equal(jnp.argmax(predictions, axis=-1), jnp.argmax(targets, axis=-1)))

        acc_dir_ = acc_dir_discrete(predictions, targets)

        return {"accuracy": accuracy, "acc_dir": acc_dir_}

    @staticmethod
    @jax.jit
    def _model_train_step_mean(state: train_state.TrainState,
                               inputs: Tuple[jnp.ndarray, jnp.ndarray],
                               targets: jnp.ndarray,
                               normalizer: jnp.ndarray,
                               rng: jax.random.PRNGKey):

        def loss_fn(params):
            predictions = state.apply_fn(params, inputs, rngs={"dropout": rng})

            means = predictions

            means_denorm = denormalize(means, normalizer[:, 0:4])
            targets_denorm = denormalize(targets, normalizer[:, 0:4])

            mae_ = mae(means_denorm, targets_denorm)
            r2_ = r2(means_denorm, targets_denorm)
            rmse_ = rmse(means_denorm, targets_denorm)
            mape_ = mape(means_denorm, targets_denorm)
            acc_dir_ = acc_dir(means, targets, inputs[0][:, -1, 3])

            w_mape = 1.0
            w_acc_dir = 1.0
            loss = w_mape * mape_ + w_acc_dir * (1.0 - 0.01 * acc_dir_)

            return loss, (mae_, r2_, rmse_, mape_, acc_dir_)

        (loss, (mae_, r2_, rmse_, mape_, acc_dir_)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = {"mae": mae_, "r2": r2_, "loss": loss, "rmse": rmse_, "mape": mape_, "acc_dir": acc_dir_}
        return state, metrics

    @staticmethod
    @jax.jit
    def _model_train_step_dist(state: train_state.TrainState,
                               inputs: Tuple[jnp.ndarray, jnp.ndarray],
                               targets: jnp.ndarray,
                               normalizer: jnp.ndarray,
                               rng: jax.random.PRNGKey):
        def loss_fn(params):
            predictions = state.apply_fn(params, inputs, rngs={"dropout": rng})

            means, stds = predictions

            means_denorm = denormalize(means, normalizer[:, 0:4])
            targets_denorm = denormalize(targets, normalizer[:, 0:4])

            mae_ = mae(means_denorm, targets_denorm)
            r2_ = r2(means_denorm, targets_denorm)
            rmse_ = rmse(means_denorm, targets_denorm)
            mape_ = mape(means_denorm, targets_denorm)
            acc_dir_ = acc_dir(means, targets, inputs[0][:, -1, 3])  # close price is at 3 idx

            loss = gaussian_negative_log_likelihood(means, stds, targets)

            return loss, (mae_, r2_, rmse_, mape_, acc_dir_)

        (loss, (mae_, r2_, rmse_, mape_, acc_dir_)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = {"mae": mae_, "r2": r2_, "loss": loss, "rmse": rmse_, "mape": mape_, "acc_dir": acc_dir_}
        return state, metrics

    @staticmethod
    @jax.jit
    def _model_train_step_discrete(state: train_state.TrainState,
                                   inputs: Tuple[jnp.ndarray, jnp.ndarray], targets: jnp.ndarray,
                                   rng: jax.random.PRNGKey):

        def loss_fn(params):
            predictions = state.apply_fn(params, inputs, rngs={"dropout": rng})

            loss = binary_cross_entropy(y_pred=predictions, y_true=targets)

            accuracy = jnp.mean(jnp.equal(jnp.argmax(predictions, axis=-1), jnp.argmax(targets, axis=-1)))
            acc_dir_ = acc_dir_discrete(predictions, targets)

            return loss, (accuracy, acc_dir_)

        (loss, (accuracy, acc_dir_)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = {"accuracy": accuracy, "loss": loss, "acc_dir": acc_dir_}
        return state, metrics

    def _model_eval_step(self, state: train_state.TrainState, inputs: Tuple[jnp.ndarray, jnp.ndarray],
                         targets: jnp.ndarray, normalizer: jnp.ndarray,
                         config: TransformerConfig) -> Tuple[train_state.TrainState, Dict]:

        predictions = self._eval_model.apply(state.params, inputs)

        # denormalize both predictions and targets
        targets_denorm = denormalize(targets, normalizer[:, 0:4])
        predictions_denorm = denormalize(predictions, normalizer[:, 0:4])
        last_price_denorm = denormalize(inputs[0][:, -1, 0:4], normalizer[:, 0:4])[:, 3]

        if config.output_mode == 'distribution':
            metrics = self._compute_metrics_dist(predictions_denorm, targets_denorm, last_price_denorm)
        elif config.output_mode == 'mean':
            metrics = self._compute_metrics_mean(predictions_denorm, targets_denorm, last_price_denorm)
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

        do_synth = True if self._config.dataset_mode == 'synthetic' else False
        do_real = True if self._config.dataset_mode == 'real' else False
        do_both = True if self._config.dataset_mode == 'both' else False

        if do_synth:
            for step in range(self._config.steps_per_epoch):
                inputs, targets, normalizer, window_info = next(self._train_dataloader_synth)
                step = state.step
                lr = self._learning_rate_fn(step)

                if self._flax_model_config.output_mode == 'distribution':
                    state, _metrics = self._model_train_step_dist(state, inputs, targets, normalizer, rng)
                elif self._flax_model_config.output_mode == 'mean':
                    state, _metrics = self._model_train_step_mean(state, inputs, targets, normalizer, rng)
                elif self._flax_model_config.output_mode == 'discrete_grid':
                    state, _metrics = self._model_train_step_discrete(state, inputs, targets, rng)
                else:
                    raise NotImplementedError(f"Output mode {self._flax_model_config.output_mode} not implemented")

                for key, value in _metrics.items():
                    metric_list = metrics.get(key, [])
                    metric_list.append(value.item())
                    metrics[key] = metric_list
        if do_real:
            for batch_idx, data in enumerate(self._train_dataloader_real):
                inputs, targets, normalizer, window_info = data
                step = state.step
                lr = self._learning_rate_fn(step)

                if self._flax_model_config.output_mode == 'distribution':
                    state, _metrics = self._model_train_step_dist(state, inputs, targets, normalizer, rng)
                elif self._flax_model_config.output_mode == 'mean':
                    state, _metrics = self._model_train_step_mean(state, inputs, targets, normalizer, rng)
                elif self._flax_model_config.output_mode == 'discrete_grid':
                    state, _metrics = self._model_train_step_discrete(state, inputs, targets, normalizer, rng)
                else:
                    raise NotImplementedError(f"Output mode {self._flax_model_config.output_mode} not implemented")

                for key, value in _metrics.items():
                    metric_list = metrics.get(key, [])
                    metric_list.append(value.item())
                    metrics[key] = metric_list

        if do_both:
            for batch_idx, data in enumerate(self._train_dataloader_real):
                inputs_real, targets_real, normalizer_real, _ = data
                inputs_synth, targets_synth, normalizer_synth, _ = next(self._train_dataloader_synth)

                # mix them randomly along the batch
                randomer = jnp.arange(0, self._config.batch_size)

                rng, key = jax.random.split(rng)
                random_idx = jax.random.permutation(rng, randomer, independent=True)
                inputs_hist = jnp.concatenate((inputs_synth[0], inputs_real[0]), axis=0)
                inputs_hist = inputs_hist[random_idx, ...]
                inputs_extra = jnp.concatenate((inputs_synth[1], inputs_real[1]), axis=0)
                inputs_extra = inputs_extra[random_idx, ...]
                inputs = (inputs_hist, inputs_extra)
                targets = jnp.concatenate((targets_synth, targets_real), axis=0)
                targets = targets[random_idx, ...]
                normalizer = jnp.concatenate((normalizer_synth, normalizer_real), axis=0)
                normalizer = normalizer[random_idx, ...]

                step = state.step
                lr = self._learning_rate_fn(step)

                if self._flax_model_config.output_mode == 'distribution':
                    state, _metrics = self._model_train_step_dist(state, inputs, targets, normalizer, rng)
                elif self._flax_model_config.output_mode == 'mean':
                    state, _metrics = self._model_train_step_mean(state, inputs, targets, normalizer, rng)
                elif self._flax_model_config.output_mode == 'discrete_grid':
                    state, _metrics = self._model_train_step_discrete(state, inputs, targets, normalizer, rng)
                else:
                    raise NotImplementedError(f"Output mode {self._flax_model_config.output_mode} not implemented")

                for key, value in _metrics.items():
                    metric_list = metrics.get(key, [])
                    metric_list.append(value.item())
                    metrics[key] = metric_list

        for key, value in metrics.items():
            metrics[key] = np.mean(value)
            metrics[key] = metrics[key]

        metrics["lr"] = lr.item()

        return state, metrics

    def _evaluate_step(self, state: train_state.TrainState) -> Dict:
        """ Runs an evaluation step """
        metrics = {}

        if self._config.dataset_mode == 'synthetic':
            for step in range(self._config.steps_per_epoch):
                inputs, targets, normalizer, window_info = next(self._test_dataloader_synth)
                state, _metrics = self._model_eval_step(state, inputs, targets, normalizer,
                                                        self._flax_model_config_eval)

                for key, value in _metrics.items():
                    metric_list = metrics.get(key, [])
                    metric_list.append(value.item())
                    metrics[key] = metric_list
        else:  # just test on real dataset when both or only real
            for data in self._test_dataloader_real:
                inputs, targets, normalizer, _ = data
                _, _metrics = self._model_eval_step(state, inputs, targets, normalizer,
                                                    config=self._flax_model_config_eval)

                for key, value in _metrics.items():
                    metric_list = metrics.get(key, [])
                    metric_list.append(value.item())
                    metrics[key] = metric_list

        for key, value in metrics.items():
            metrics[key] = np.mean(value)
            metrics[key] = metrics[key]

        return metrics

    def _best_model_test(self, max_seqs: int = 20):
        """ Generate images from the test set of the best model """

        test_dataloader = DataLoader(self._test_ds, batch_size=1, collate_fn=jax_collate_fn, shuffle=True)
        save_folder = os.path.join(self._log_dir, "best_model_test", self._config.__str__())
        os.makedirs(save_folder, exist_ok=True)

        counter = 0
        for i, data in enumerate(test_dataloader):
            inputs, targets, normalizer, _ = data
            output = self._eval_model.apply(self._best_model_state.params, inputs)
            plot_predictions(input=inputs[0].squeeze(0),
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
