from dataclasses import dataclass, asdict
import json
import os
from typing import Optional, Tuple
from jaxer.utils.dataset import DatasetConfig


@dataclass
class ModelConfig:
    d_model: int
    num_layers: int
    head_layers: int
    n_heads: int
    dim_feedforward: int
    dropout: float
    max_seq_len: int
    flatten_encoder_output: bool
    fe_blocks: int
    use_time2vec: bool
    output_mode: str
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
    dataset_config: DatasetConfig
    batch_size: int
    test_split: float
    seed: int
    save_weights: bool
    early_stopper: int

    def save_config(self, path):
        config = asdict(self)
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load_config(cls, path: str):
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(**config)

    def __str__(self):
        return f"lr_{self.learning_rate}_{self.lr_mode}_bs_{self.batch_size}_ep_{self.num_epochs}_wmp_" \
               f"{self.warmup_epochs}_seed_{self.seed}_" \
               f"dmodel_{self.model_config.d_model}_nlayers_{self.model_config.num_layers}_" \
               f"ndl_{self.model_config.head_layers}_" \
               f"nhds_{self.model_config.n_heads}_dimff_{self.model_config.dim_feedforward}_" \
               f"drpt_{self.model_config.dropout}_" \
               f"maxlen_{self.model_config.max_seq_len}_" \
               f"flat_{self.model_config.flatten_encoder_output}_" \
               f"feblk_{self.model_config.fe_blocks}_t2v_{self.model_config.use_time2vec}_" \
               f"{self.dataset_config.norm_mode}_" \
               f"ind_{True if self.dataset_config.indicators else False}_" \
               f"ds_{self.test_split}_{self.dataset_config.initial_date}_" \
               f"outmd_{self.model_config.output_mode}_" \
               f"reshd_{self.model_config.use_resblocks_in_head}_" \
               f"resfe_{self.model_config.use_resblocks_in_fe}_" \
               f"avout_{self.model_config.average_encoder_output}_" \
               f"nrmpre_{self.model_config.norm_encoder_prev}"


def get_best_model(experiment_name: str) -> Tuple[Optional[str], str]:
    """ Returns the best model from the experiment """
    complete_path = os.path.join("results", experiment_name, "best_model_train.json")

    if not os.path.exists(complete_path):
        raise FileNotFoundError(f"File {complete_path} not found")

    with open(complete_path, 'r') as f:
        best_model = json.load(f)

    return best_model.get("subfolder", None), str(best_model["ckpt"])
