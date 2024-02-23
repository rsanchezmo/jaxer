from jaxer.run.agent_base import AgentBase
from jaxer.model.flax_transformer import Transformer, TransformerConfig
import orbax.checkpoint
from typing import Tuple, Optional


class FlaxAgent(AgentBase):
    """ Agent class to load a model and perform inference

    :param experiment: the name of the experiment
    :type experiment: str

    :param model_name: the name of the model to load. If the model is in a subfolder,
        provide a tuple with the subfolder name and the model name
    :type model_name: Tuple[Optional[str], str]
    """

    def __init__(self, experiment: str, model_name: Tuple[Optional[str], str]) -> None:
        super().__init__(experiment, model_name)

        self._flax_config = TransformerConfig(
            d_model=self.config.model_config["d_model"],
            num_layers=self.config.model_config["num_layers"],
            n_heads=self.config.model_config["n_heads"],
            dim_feedforward=self.config.model_config["dim_feedforward"],
            dropout=self.config.model_config["dropout"],
            max_seq_len=self.config.model_config["max_seq_len"],
            deterministic=True,
            flatten_encoder_output=self.config.model_config["flatten_encoder_output"],
            head_layers=self.config.model_config["head_layers"],
            fe_blocks=self.config.model_config["fe_blocks"],
            use_time2vec=self.config.model_config["use_time2vec"],
            output_mode=self.config.model_config["output_mode"],
            discrete_grid_levels=len(
                self.config.dataset_config['discrete_grid_levels']) - 1 if self.config.dataset_config['discrete_grid_levels'] is not None else None,
            use_resblocks_in_head=self.config.model_config["use_resblocks_in_head"],
            use_resblocks_in_fe=self.config.model_config["use_resblocks_in_fe"],
            average_encoder_output=self.config.model_config["average_encoder_output"],
            norm_encoder_prev=self.config.model_config["norm_encoder_prev"]
        )

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored_state = orbax_checkpointer.restore(self.ckpt_path)

        self.model = Transformer(self._flax_config).bind(restored_state["model"]["params"])
