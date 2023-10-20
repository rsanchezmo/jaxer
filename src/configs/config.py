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
    num_epochs: int
    learning_rate: float
    train_ds_path: str
    test_ds_path: str
    


config = ModelConfig(
    d_model=512,
    num_layers=6,
    n_heads=8,
    dim_feedforward=2048,
    dropout=0.0,
    max_seq_len=2048,
    input_features=1
)

training_config = Config(
    model_config=config,
    log_dir="logs",
    num_epochs=100,
    learning_rate=1e-4,
    train_ds_path="../data/train.csv",
    test_ds_path="../data/test.csv",
)