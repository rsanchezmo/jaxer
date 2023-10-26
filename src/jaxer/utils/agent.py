from typing import Any
from ..configs.config import Config
from ..model.flax_transformer import Transformer, TransformerConfig
import os
import jax.numpy as jnp
import jax
import orbax.checkpoint


class Agent:
    def __init__(self, experiment: str, model_name: str) -> None:
        self.experiment_path = os.path.join("results", experiment)
        self.model_name = model_name
        self.ckpt_path = os.path.join(self.experiment_path, "ckpt", model_name)

        assert os.path.exists(self.experiment_path), f"Experiment {experiment} does not exist in results folder"
        assert os.path.exists(self.ckpt_path), f"Model {model_name} does not exist in experiment {experiment}"

        self.config = Config.load_config(f"{self.experiment_path}/config.json")

        self._flax_config = TransformerConfig(
            d_model=self.config.model_config["d_model"],
            num_layers=self.config.model_config["num_layers"],
            n_heads=self.config.model_config["n_heads"],
            dim_feedforward=self.config.model_config["dim_feedforward"],
            dropout=self.config.model_config["dropout"],
            max_seq_len=self.config.model_config["max_seq_len"],
            deterministic=True
        )

        """ Create an orbax checkpointer to restore the model"""
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored_state = orbax_checkpointer.restore(self.ckpt_path)


    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Inference function wrapper as __call__ """
        
