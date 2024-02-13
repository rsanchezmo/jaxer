from jaxer import Agent
from jaxer import Dataset, jax_collate_fn
from jaxer import plot_predictions, predict_entire_dataset
from jaxer import get_best_model

from torch.utils.data import DataLoader


if __name__ == '__main__':
    """ LOAD THE AGENT """
    experiment = "output_discrete_grid"
    agent = Agent(experiment=experiment, model_name=get_best_model(experiment))

    """ DATALOADERS """
    dataset = Dataset(dataset_config=agent.config.dataset_config)

    train_ds, test_ds = dataset.get_train_test_split(test_size=agent.config.test_split)

    # infer entire dataset
    plot_entire_dataset = False
    if plot_entire_dataset:
        predict_entire_dataset(agent, test_ds, mode='test')
        predict_entire_dataset(agent, train_ds, mode='train')

    # infer once
    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=jax_collate_fn)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=jax_collate_fn)

    for batch in train_dataloader:
        input, label, normalizer, initial_date = batch
        output = agent(input)
        plot_predictions(input.squeeze(0), label.squeeze(0), output, normalizer=normalizer[0], name='train',
                         initial_date=initial_date[0], output_mode=agent.config.model_config["output_mode"],
                         discrete_grid_levels=agent.config.dataset_config.discrete_levels)
