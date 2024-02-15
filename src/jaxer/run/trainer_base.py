import os
from typing import Optional

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from jaxer.utils.dataset import Dataset
from jaxer.utils.logger import get_logger
from jaxer.utils.config import Config
from jaxer.utils.dataset import jax_collate_fn

from flax.training import orbax_utils, train_state
import orbax


class TrainerBase:
    """ Trainer base class. All trainers should inherit from this

    :param config: the configuration for the experiment
    :type config: Config
    """

    def __init__(self, config: Config) -> None:
        self._config = config

        """ Training logs """
        self._log_dir = os.path.join(os.getcwd(), self._config.log_dir, self._config.experiment_name)
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(os.path.join(self._log_dir, "tensorboard"), exist_ok=True)
        self._summary_writer = SummaryWriter(
            os.path.join(self._log_dir, "tensorboard", self._config.__str__()))

        """ Checkpoints """
        self._ckpts_dir = os.path.join(self._log_dir, "ckpt", self._config.__str__())
        os.makedirs(self._ckpts_dir, exist_ok=True)
        self._orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(create=True, max_to_keep=5)
        self._checkpoint_manager = orbax.checkpoint.CheckpointManager(
            self._ckpts_dir, self._orbax_checkpointer, options)

        """ Dataloaders """
        self._dataset = Dataset(dataset_config=self._config.dataset_config)

        self._train_ds, self._test_ds = self._dataset.get_train_test_split(test_size=self._config.test_split)
        self._train_dataloader = DataLoader(self._train_ds, batch_size=self._config.batch_size, shuffle=True,
                                            collate_fn=jax_collate_fn)
        self._test_dataloader = DataLoader(self._test_ds, batch_size=self._config.batch_size, shuffle=True,
                                           collate_fn=jax_collate_fn)

        os.makedirs(os.path.join(self._log_dir, 'configs', self._config.__str__()), exist_ok=True)

        """ Save config """
        self._config.save_config(
            os.path.join(self._log_dir, 'configs', self._config.__str__(), "config.json"))

        """ Best model state: eval purposes """
        self._best_model_state: Optional[train_state.TrainState] = None

        self.logger = get_logger()

    def train_and_evaluate(self) -> None:
        """ Runs a training loop """
        raise NotImplementedError

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

    def _load_model(self, epoch: int) -> train_state.TrainState:
        """ Loads a model checkpoint

        :param epoch: the epoch number
        :type epoch: int

        :return: the model state
        :rtype: train_state.TrainState
        """
        ckpt = self._orbax_checkpointer.restore(os.path.join(self._ckpts_dir, str(epoch)))
        return ckpt['model']
