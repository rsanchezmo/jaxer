from dataclasses import dataclass
import json
from dataclasses import asdict
import os
from typing import Optional, List


@dataclass
class ModelConfig:
    d_model: int
    num_layers: int
    head_layers: int
    n_heads: int
    dim_feedforward: int
    dropout: float
    max_seq_len: int
    input_features: int
    flatten_encoder_output: int
    fe_blocks: int
    use_time2vec: bool
    output_mode: str
    discrete_grid_levels: int
    use_resblocks_in_head: bool
    use_resblocks_in_fe: bool
    average_encoder_output: bool
    norm_encoder_prev: bool

@dataclass
class Config:
    model_config: ModelConfig
    log_dir: str   
    experiment_name: str
    num_epochs: int
    learning_rate: float
    lr_mode: str
    warmup_epochs: int
    dataset_path: str
    dataset_mode: str
    dataset_discrete_levels: Optional[List[float]]
    initial_date: str
    batch_size: int
    test_split: float
    seed: int
    normalizer_mode: str
    save_weights: bool
    early_stopper: int

    def save_config(self, path):
        config = asdict(self)
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)

    def load_config(path):
        with open(path, 'r') as f:
            config = json.load(f)
        return Config(**config)


def get_best_model(experiment_name: str) -> str:
    """ Returns the best model from the experiment """
    complete_path = os.path.join("results", experiment_name, "best_model_train.json")

    if not os.path.exists(complete_path):
        raise FileNotFoundError(f"File {complete_path} not found")
    
    with open(complete_path, 'r') as f:
        best_model = json.load(f)

    return best_model.get("subfolder", None), str(best_model["ckpt"])

