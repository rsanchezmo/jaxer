from jaxer.utils.trainer import FlaxTrainer as Trainer
from training_config import config


if __name__ == '__main__':

    trainer = Trainer(config=config)
    trainer.train_and_evaluate()
