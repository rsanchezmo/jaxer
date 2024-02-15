from dataclasses import dataclass, asdict
import json
import os
from typing import Optional, Tuple
from jaxer.utils.dataset import DatasetConfig


@dataclass
class ModelConfig:
    """ Configuration class for the model

    :param d_model: dimension of the model
    :type d_model: int

    :param num_layers: number of encoder layers in the transformer
    :type num_layers: int

    :param head_layers: number of layers in the prediction head
    :type head_layers: int

    :param n_heads: number of heads in the multihead attention
    :type n_heads: int

    :param dim_feedforward: dimension of the feedforward network
    :type dim_feedforward: int

    :param dropout: dropout rate
    :type dropout: float

    :param max_seq_len: maximum sequence length (context window)
    :type max_seq_len: int

    :param flatten_encoder_output: whether to flatten the encoder output or get the last token
    :type flatten_encoder_output: bool

    :param fe_blocks: number of blocks in the feature extractor
    :type fe_blocks: int

    :param use_time2vec: whether to use time2vec in the feature extractor
    :type use_time2vec: bool

    :param output_mode: output mode of the model (mean, distribution or discrete_grid)
    :type output_mode: str

    :param use_resblocks_in_head: whether to use residual blocks in the head
    :type use_resblocks_in_head: bool

    :param use_resblocks_in_fe: whether to use residual blocks in the feature extractor
    :type use_resblocks_in_fe: bool

    :param average_encoder_output: whether to average the encoder output (if not flattened)
    :type average_encoder_output: bool

    :param norm_encoder_prev: whether to normalize the encoder prev to the attention
    :type norm_encoder_prev: bool

    """
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
    """ Configuration class for a traning experiment

    :param model_config: model configuration (transformer architecture)
    :type model_config: ModelConfig

    :param log_dir: directory to save the logs
    :type log_dir: str

    :param experiment_name: name of the experiment
    :type experiment_name: str

    :param num_epochs: number of epochs to train
    :type num_epochs: int

    :param learning_rate: learning rate
    :type learning_rate: float

    :param lr_mode: learning rate mode (cosine, linear or none)
    :type lr_mode: str

    :param warmup_epochs: number of warmup epochs (for learning rate)
    :type warmup_epochs: int

    :param dataset_config: dataset configuration
    :type dataset_config: DatasetConfig

    :param batch_size: batch size for training
    :type batch_size: int

    :param test_split: test split (between 0 and 1)
    :type test_split: float

    :param seed: seed for reproducibility
    :type seed: int
    """

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
    """ Returns the best model from the experiment

    :param experiment_name: name of the experiment
    :type experiment_name: str

    :return: subfolder and checkpoint of the best model
    :rtype: Tuple[Optional[str], str]

    :raises FileNotFoundError: if the file is not found
    """
    complete_path = os.path.join("results", experiment_name, "best_model_train.json")

    if not os.path.exists(complete_path):
        raise FileNotFoundError(f"File {complete_path} not found")

    with open(complete_path, 'r') as f:
        best_model = json.load(f)

    return best_model.get("subfolder", None), str(best_model["ckpt"])
