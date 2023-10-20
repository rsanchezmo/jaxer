from ..model.transformer import TransformerConfig
import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class Config:
    model_config: TransformerConfig
    log_dir: str   
    num_epochs: int
    learning_rate: float
    


config = TransformerConfig(
    d_model=512,
    num_layers=6,
    n_heads=8,
    dim_feedforward=2048,
    dropout=0.0,
    dtype=jnp.float32
)

training_config = Config(
    model_config=config,
    log_dir="logs",
    num_epochs=100,
    learning_rate=1e-4,
)