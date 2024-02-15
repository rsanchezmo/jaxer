import jaxer
from training_config import config


if __name__ == '__main__':

    trainer = jaxer.run.FlaxTrainer(config=config)
    trainer.train_and_evaluate()
