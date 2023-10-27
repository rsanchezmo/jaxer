from dataclasses import dataclass
import json
from dataclasses import asdict


@dataclass
class ModelConfig:
    d_model: int
    num_layers: int
    n_heads: int
    dim_feedforward: int
    dropout: float
    max_seq_len: int
    input_features: int

@dataclass
class Config:
    model_config: ModelConfig
    log_dir: str   
    experiment_name: str
    num_epochs: int
    learning_rate: float
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


config = ModelConfig(
    d_model=128,
    num_layers=8,
    n_heads=8,
    dim_feedforward=256,
    dropout=0.05,
    max_seq_len=30,
    input_features=6
)

training_config = Config(
    model_config=config,
    log_dir="results",
    experiment_name="v1",
    num_epochs=100,
    learning_rate=1e-5,
    dataset_path="./data/BTCUSD.csv",
    batch_size=128,
    test_split=0.1,
    seed=0,
    normalizer_mode="global"
)