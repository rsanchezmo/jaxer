import os
from typing import Optional

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from jaxer.utils.dataset import Dataset
from jaxer.utils.logger import Logger
from jaxer.utils.config import Config
from jaxer.utils.dataset import jax_collate_fn

from flax.training import orbax_utils, train_state
import orbax



class TrainerBase:
    def __init__(self, config: Config) -> None:
        """ Trainer Base Class: All trainers should inherit from this, FLAX or PyTorch [in the future] """
        self._config = config

        """ Training logs """
        self._log_dir = os.path.join(os.getcwd(), self._config.log_dir, self._config.experiment_name)
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(os.path.join(self._log_dir, "tensorboard"), exist_ok=True)
        # get num of experiments inside the folder
        self._summary_writer = SummaryWriter(os.path.join(self._log_dir, "tensorboard", self._config_to_str(self._config)))

        """ Checkpoints """
        self._ckpts_dir = os.path.join(self._log_dir, "ckpt", self._config_to_str(self._config))
        os.makedirs(self._ckpts_dir, exist_ok=True)
        self._orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(create=True, max_to_keep=5)
        self._checkpoint_manager = orbax.checkpoint.CheckpointManager(
            self._ckpts_dir, self._orbax_checkpointer, options)
        

        """ Dataloaders """
        dataset = Dataset(dataset_config=self._config.dataset_config)
        
        self._train_ds, self._test_ds = dataset.get_train_test_split(test_size=self._config.test_split)
        self._train_dataloader = DataLoader(self._train_ds, batch_size=self._config.batch_size, shuffle=True, collate_fn=jax_collate_fn)
        self._test_dataloader = DataLoader(self._test_ds, batch_size=self._config.batch_size, shuffle=True, collate_fn=jax_collate_fn)


        os.makedirs(os.path.join(self._log_dir, 'configs', self._config_to_str(self._config)), exist_ok=True)

        """ Save config """
        self._config.save_config(os.path.join(self._log_dir, 'configs', self._config_to_str(self._config), "config.json"))

        """ Best model state: eval purposes """
        self._best_model_state: Optional[train_state.TrainState] = None

        self.logger = Logger("Trainer")


    @staticmethod
    def to_binary(val: bool):
        return 1 if val else 0

    def _config_to_str(self, config: Config) -> str:
        """ Converts a config to a string """
        return f"{config.learning_rate}_{config.lr_mode}_bs_{config.batch_size}_ep_{config.num_epochs}_wmp_{config.warmup_epochs}_seed_{config.seed}_" \
                f"dmodel_{config.model_config.d_model}_nlayers_{config.model_config.num_layers}_nhdl_{config.model_config.head_layers}_" \
                f"nhds_{config.model_config.n_heads}_dimff_{config.model_config.dim_feedforward}_drpt_{config.model_config.dropout}_" \
                f"maxlen_{config.model_config.max_seq_len}_infeat_{config.model_config.input_features}_flat_{self.to_binary(config.model_config.flatten_encoder_output)}_" \
                f"feblk_{config.model_config.fe_blocks}_t2v_{self.to_binary(config.model_config.use_time2vec)}_{config.dataset_config.norm_mode}_" \
                f"ds_{config.test_split}_{config.dataset_config.initial_date}_" \
                f"outmd_{self.to_binary(config.model_config.output_mode)}_" \
                f"reshd_{self.to_binary(config.model_config.use_resblocks_in_head)}_" \
                f"resfe_{self.to_binary(config.model_config.use_resblocks_in_fe)}_" \
                f"avout_{self.to_binary(config.model_config.average_encoder_output)}_nrmpre_{self.to_binary(config.model_config.norm_encoder_prev)}"


    def train_and_evaluate(self) -> None:
        """ Runs a training loop """
        raise NotImplementedError
    
    def _save_model(self, epoch: int, state: train_state.TrainState) -> None:
        """ Saves a model checkpoint """
        ckpt = {'model': state}
        save_args = orbax_utils.save_args_from_target(ckpt)
        self._checkpoint_manager.save(epoch, ckpt, save_kwargs={'save_args': save_args})

    def _load_model(self, epoch: int) -> train_state.TrainState:
        """ Loads a model checkpoint """
        ckpt = self._orbax_checkpoint.restore(os.path.join(self._ckpts_dir, epoch))
        return ckpt['model']
