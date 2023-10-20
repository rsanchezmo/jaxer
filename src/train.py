from .utils.trainer import Trainer
from .configs.config import training_config


if __name__ == '__main__':

    trainer = Trainer(config=training_config)
    trainer.train()