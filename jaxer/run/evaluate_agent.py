from jaxer.utils.plotter import plot_metrics
from jaxer.utils.losses import compute_metrics
from jaxer.run.agent import FlaxAgent
from jaxer.utils.synthetic_dataset import SyntheticDataset
from torch.utils.data import DataLoader
from jaxer.utils.dataset import jax_collate_fn
from jaxer.utils.dataset import Dataset
from typing import Union


def evaluate_agent(agent: FlaxAgent, dataset: Union[SyntheticDataset, Dataset], max_samples: int = 50_000, seed: int = 100):

    max_batch_size = 512
    if isinstance(dataset, SyntheticDataset):
        dataloader = dataset.generator(batch_size=max_batch_size, seed=seed)
    else:
        dataloader = DataLoader(dataset, batch_size=max_batch_size, shuffle=False, collate_fn=jax_collate_fn)

    mape = []
    acc_dir = []

    if isinstance(dataset, SyntheticDataset):
        for batch_idx in range(int(max_samples/max_batch_size)):
            x, y_true, normalizer, window_info = next(dataloader)
            y_pred = agent(x)

            metrics = compute_metrics(x, y_pred, y_true, normalizer, denormalize_values=True)

            mape += metrics['mape'][:, 0].tolist()
            acc_dir += metrics['acc_dir'].tolist()
    else:
        for idx, batch in enumerate(dataloader):
            x, y_true, normalizer, window_info = batch
            y_pred = agent(x)

            metrics = compute_metrics(x, y_pred, y_true, normalizer, denormalize_values=True)

            mape += metrics['mape'][:, 0].tolist()
            acc_dir += metrics['acc_dir'].tolist()

    plot_metrics(mape, acc_dir)
