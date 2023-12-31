from jaxer.utils.agent import Agent
import jax.numpy as jnp
import time
from jaxer.utils.dataset import Dataset, jax_collate_fn
from torch.utils.data import DataLoader
from jaxer.utils.plotter import plot_predictions, predict_entire_dataset
from jaxer.utils.config import get_best_model



if __name__ == '__main__':
    """ LOAD THE AGENT """
    experiment = "overfit"
    agent = Agent(experiment=experiment, model_name=get_best_model(experiment))

    """ LOAD SOME DATA """
    x_test = jnp.ones((1, agent.config.model_config["max_seq_len"], agent.config.model_config["input_features"]))

    """ PREDICT """
    init_t = time.time()
    mean, var = agent(x_test)
    print(f"Time taken: {1e3*(time.time() - init_t):.2f} ms")

    # get the dataloaders from training
    """ Dataloaders """
    dataset = Dataset('./data/BTCUSD.csv', agent.config.model_config["max_seq_len"], norm_mode=agent.config.normalizer_mode,
                      initial_date=agent.config.initial_date)
    train_ds, test_ds = dataset.get_train_test_split(test_size=agent.config.test_split)


    # infer entire dataset
    plot_entire_dataset = True
    if plot_entire_dataset:
        predict_entire_dataset(agent, test_ds, mode='test')
        predict_entire_dataset(agent, train_ds, mode='train')

    # infer once
    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=jax_collate_fn)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=jax_collate_fn)

    for batch in test_dataloader:
        input, label, normalizer, initial_date = batch
        mean, var = agent(input)
        plot_predictions(input.squeeze(0), label.squeeze(0), mean.squeeze(0), var.squeeze(0), normalizer=normalizer[0], name='train', initial_date=initial_date[0])
        
    