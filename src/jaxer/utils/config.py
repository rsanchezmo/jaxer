from dataclasses import dataclass
import json
from dataclasses import asdict


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

@dataclass
class Config:
    model_config: ModelConfig
    log_dir: str   
    experiment_name: str
    num_epochs: int
    learning_rate: float
    warmup_epochs: int
    dataset_path: str
    batch_size: int
    test_split: float
    seed: int
    normalizer_mode: str

    def save_config(self, path):
        config = asdict(self)
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)

    def load_config(path):
        with open(path, 'r') as f:
            config = json.load(f)
        return Config(**config)


