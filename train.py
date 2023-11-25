from jaxer.utils.trainer import FlaxTrainer as Trainer
from training_config import training_config


if __name__ == '__main__':

    trainer = Trainer(config=training_config)
    trainer.train_and_evaluate()