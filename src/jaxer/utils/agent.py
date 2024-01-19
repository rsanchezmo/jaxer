from typing import Any
from .config import Config
from ..model.flax_transformer import Transformer, TransformerConfig
import os
import jax.numpy as jnp
import jax
import orbax.checkpoint
from typing import Tuple, Optional


class Agent:
    def __init__(self, experiment: str, model_name: Tuple[Optional[str], str]) -> None:
        self.experiment_path = os.path.join("results", experiment)
        subfolder, self.model_name = model_name


        self.ckpt_path = os.path.join(self.experiment_path, 'ckpt', subfolder, self.model_name, 'default')

        assert os.path.exists(self.experiment_path), f"Experiment {experiment} does not exist in results folder"
        assert os.path.exists(self.ckpt_path), f"Model {self.model_name} does not exist in experiment {experiment} with subfolder {subfolder}"

        self.config = Config.load_config(os.path.join(self.experiment_path,'configs', subfolder, 'config.json'))

        self._flax_config = TransformerConfig(
            d_model=self.config.model_config["d_model"],
            num_layers=self.config.model_config["num_layers"],
            n_heads=self.config.model_config["n_heads"],
            dim_feedforward=self.config.model_config["dim_feedforward"],
            dropout=self.config.model_config["dropout"],
            max_seq_len=self.config.model_config["max_seq_len"],
            deterministic=True,
            input_features=self.config.model_config["input_features"],
            flatten_encoder_output=self.config.model_config["flatten_encoder_output"],
            head_layers=self.config.model_config["head_layers"],
            fe_blocks=self.config.model_config["fe_blocks"],
            use_time2vec=self.config.model_config["use_time2vec"],
            output_mode=self.config.model_config["output_mode"],
            discrete_grid_levels=len(self.config["dataset_grid_levels"])-1 if self.config["dataset_grid_levels"] is not None else None,
            use_resblocks_in_head=self.config.model_config["use_resblocks_in_head"],
            use_resblocks_in_fe=self.config.model_config["use_resblocks_in_fe"],
            average_encoder_output=self.config.model_config["average_encoder_output"],
            norm_encoder_prev=self.config.model_config["norm_encoder_prev"]
        )

        """ Create an orbax checkpointer to restore the model"""
        assert self.config.save_weights, "Weights were not saved during training, cannot restore model"

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored_state = orbax_checkpointer.restore(self.ckpt_path)
        self.params = restored_state["model"]["params"]

        """ Do a warmup of the model """
        self.model = Transformer(self._flax_config)

        self.model.apply(self.params, jnp.ones((1, self.config.model_config["max_seq_len"], self.config.model_config["input_features"])))


    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Inference function wrapper as __call__ """
        return self.model.apply(self.params, x)
        
