from jaxer.config.experiment_config import ExperimentConfig
import os
from typing import Tuple, Optional, Callable, Any


class AgentBase:
    """ Agent base class to load a model and perform inference (jax, torch, ...)

    :param experiment: the name of the experiment
    :type experiment: str

    :param model_name: the name of the model to load. If the model is in a subfolder,
            provide a tuple with the subfolder name and the model name
    :type model_name: Tuple[Optional[str], str]

    :raises FileNotFoundError: if the experiment or model does not exist
    """

    def __init__(self, experiment: str, model_name: Tuple[Optional[str], str]) -> None:
        self.experiment_path = experiment
        subfolder, self.model_name = model_name

        self.ckpt_path = os.path.join(self.experiment_path, 'ckpt', subfolder)

        if not os.path.exists(self.experiment_path):
            raise FileNotFoundError(f"Experiment {experiment} does not exist in results folder")

        if not os.path.exists(os.path.join(self.experiment_path, 'configs', subfolder, 'config.json')):
            raise FileNotFoundError(f"Config file for experiment {experiment} does not exist")

        if not os.path.exists(os.path.join(self.ckpt_path, self.model_name)):
            raise FileNotFoundError(f"Model {self.model_name} does not exist in experiment {experiment} "
                                    f"with subfolder {subfolder}")

        self.config = ExperimentConfig.load_config(os.path.join(self.experiment_path, 'configs',
                                                                subfolder, 'config.json'))

        if not self.config.save_weights:
            raise ValueError("Weights were not saved during training, cannot restore model")

        self.model = Callable[[Any], Any]

    def __call__(self, x: Any) -> Any:
        return self.forward(x)

    def forward(self, x: Any) -> Any:
        """ Inference function (you can use __call__ instead)

        :param x: input data
        :type x: Any

        :return: model output
        :rtype: Any
        """
        return self.model(x)
