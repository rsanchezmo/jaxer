import jaxer
import os
from torch.utils.data import DataLoader

if __name__ == '__main__':
    """ LOAD THE AGENT """
    experiment = "hp_search_report_loss_all_30m/mean"
    model_name = jaxer.utils.get_best_model(experiment)
    agent = jaxer.run.FlaxAgent(experiment=experiment, model_name=model_name)

    """ plot experiment"""
    experiment_path = os.path.join(experiment, "tensorboard", model_name[0])
    jaxer.utils.plot_tensorboard_experiment(experiment_path)

    """ DATALOADERS """
    dataset = jaxer.utils.Dataset(dataset_config=jaxer.config.DatasetConfig.from_dict(agent.config.dataset_config))

    train_ds, test_ds = dataset.get_train_test_split(test_size=agent.config.test_split,
                                                     test_tickers=agent.config.test_tickers)

    """ INFERENCE """

    # infer entire dataset
    plot_entire_dataset = False
    if plot_entire_dataset:
        jaxer.utils.predict_entire_dataset(agent, test_ds, mode='test')
        jaxer.utils.predict_entire_dataset(agent, train_ds, mode='train')

    # infer once
    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=jaxer.utils.jax_collate_fn)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=jaxer.utils.jax_collate_fn)

    infer_test = True

    if infer_test:
        dataloader = test_dataloader
    else:
        dataloader = train_dataloader

    for batch in dataloader:
        x, y_true, normalizer, window_info = batch
        y_pred = agent(x)
        jaxer.utils.plot_predictions(x=x, y_true=y_true, y_pred=y_pred, normalizer=normalizer,
                                     window_info=window_info[0])
