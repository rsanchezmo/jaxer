import os
from torch.utils.tensorboard import SummaryWriter
from jaxer.utils.logger import get_logger
from jaxer.utils.config import Config
from abc import abstractmethod
from typing import Any


class TrainerBase:
    """ Trainer base class. All trainers should inherit from this (jax, torch, ...)

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

        """ Save config """
        os.makedirs(os.path.join(self._log_dir, 'configs', self._config.__str__()), exist_ok=True)
        self._config.save_config(
            os.path.join(self._log_dir, 'configs', self._config.__str__(), "config.json"))

        self.logger = get_logger()

    @abstractmethod
    def train_and_evaluate(self) -> None:
        """ Runs a training loop """
        raise NotImplementedError

    @abstractmethod
    def _save_model(self, epoch: int, state: Any) -> None:
        """ Saves a model checkpoint

        :param epoch: the epoch number
        :type epoch: int

        :param state: the model state
        :type state: Any
        """
        raise NotImplementedError

    @abstractmethod
    def _load_model(self, epoch: int) -> Any:
        """ Loads a model checkpoint

        :param epoch: the epoch number
        :type epoch: int

        :return: the model state
        :rtype: Any
        """
        raise NotImplementedError
