import jaxer
import os
from torch.utils.data import DataLoader

if __name__ == '__main__':
    """ LOAD THE AGENT """
    experiment = "results/synth_tiny_from_pretrained"
    model_name = jaxer.utils.get_best_model(experiment)
    agent = jaxer.run.FlaxAgent(experiment=experiment, model_name=model_name)

    """ plot experiment"""
    experiment_path = os.path.join(experiment, "tensorboard", model_name[0])
    jaxer.utils.plot_tensorboard_experiment(experiment_path)

    """ DATALOADERS """
    if agent.config.dataset_mode == 'synthetic':
        dataset = jaxer.utils.SyntheticDataset(config=jaxer.config.SyntheticDatasetConfig.from_dict(agent.config.synthetic_dataset_config))
        test_dataloader = dataset.generator(batch_size=1, seed=100)
        train_dataloader = dataset.generator(batch_size=1, seed=200)
    else:
        dataset = jaxer.utils.Dataset(dataset_config=jaxer.config.DatasetConfig.from_dict(agent.config.dataset_config))

        train_ds, test_ds = dataset.get_train_test_split(test_size=agent.config.test_split,
                                                         test_tickers=agent.config.test_tickers)
        train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=jaxer.utils.jax_collate_fn)
        test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=jaxer.utils.jax_collate_fn)

    infer_test = False

    if infer_test:
        dataloader = test_dataloader
    else:
        dataloader = train_dataloader

    if agent.config.dataset_mode == 'synthetic':
        for i in range(30):
            x, y_true, normalizer, window_info = next(dataloader)
            y_pred = agent(x)
            jaxer.utils.plot_predictions(x=x, y_true=y_true, y_pred=y_pred, normalizer=normalizer,
                                         window_info=window_info, denormalize_values=True)

    else:
        for batch in dataloader:
            x, y_true, normalizer, window_info = batch
            y_pred = agent(x)
            jaxer.utils.plot_predictions(x=x, y_true=y_true, y_pred=y_pred, normalizer=normalizer,
                                         window_info=window_info[0], denormalize_values=True)
