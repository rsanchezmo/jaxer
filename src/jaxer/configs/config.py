from dataclasses import dataclass


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
    seed: int = 0
    

config = ModelConfig(
    d_model=128,
    num_layers=4,
    n_heads=4,
    dim_feedforward=512,
    dropout=0.1,
    max_seq_len=20,
    input_features=6
)

training_config = Config(
    model_config=config,
    log_dir="results",
    experiment_name="v0",
    num_epochs=100,
    learning_rate=1e-4,
    dataset_path="./data/BTCUSD.csv",
    batch_size=64,
    test_split=0.1,
    seed=0
)