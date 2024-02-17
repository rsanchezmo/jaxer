from dataclasses import dataclass, asdict
import json
from jaxer.config.dataset_config import DatasetConfig
from jaxer.config.model_config import ModelConfig


@dataclass
class ExperimentConfig:
    """ Configuration class for a training experiment

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

