from typing import Optional, Tuple
import json
import os


def get_best_model(experiment_name: str) -> Tuple[Optional[str], str]:
    """ Returns the best model from the experiment

    :param experiment_name: name of the experiment
    :type experiment_name: str

    :return: subfolder and checkpoint of the best model
    :rtype: Tuple[Optional[str], str]

    :raises FileNotFoundError: if the file is not found
    """
    complete_path = os.path.join(experiment_name, "best_model_train.json")

    if not os.path.exists(complete_path):
        raise FileNotFoundError(f"File {complete_path} not found")

    with open(complete_path, 'r') as f:
        best_model = json.load(f)

    return best_model.get("subfolder", None), str(best_model["ckpt"])
