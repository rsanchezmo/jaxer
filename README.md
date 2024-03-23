# Jaxer

Jax and Flax Time Series Prediction Transformer. The goal of this repo is to learn [Jax](https://jax.readthedocs.io/en/latest/) and [Flax](https://flax.readthedocs.io/en/latest/) by implementing a deep learning model. In this case, a **transformer** for time series prediction. I have decided to predict cryptocurrency due to the availability of the data. However, this could be used for any other time series data, such as market stocks or energy demand.

![Jaxer Logo](./docs/images/btc_transformer.png)

You can check out the project [documentation](https://jaxer.readthedocs.io/) for more information about the project and also for results and conclusions.

## Usage

### Installation
I highly recommend to clone and run the repo on linux device because the installation of jax and related libs is easier,
but with a bit of time you may end up running on other compatible platform. To start, just clone the repo:
        
  ```bash
  git clone https://github.com/rsanchezmo/jaxer.git
  cd jaxer
  ```
Create a python venv and source it:
    
  ```bash
    python3 -m venv venv
    source venv/bin/activate
  ```

Install your desired Jax version (more info at https://jax.readthedocs.io/en/latest/installation.html). You must notice
that jax only provides 2 distributions with cuda (12.3 and 11.8). As this repo also depends on torch, you could install 
torch in cpu and jax in gpu, torch is only used for the dataloaders. However, I decided to better install torch+cuda11.8 and jax+cuda11.8.
For example, if already installed CUDA 11.8 on Linux (make sure to have exported to PATH your CUDA version):
    
  ```bash
    (venv) $ pip install --upgrade jax[cuda11_local] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    (venv) $ pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```

If cuda is not installed, you may use pip installation and let the pip install the right version for your gpu. Make sure that
when doing nvcc --version, it says that you should install cuda toolkit (just remove if you have a release already reported on
.bashrc:
     
  ```bash
    (venv) $ pip install --upgrade jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_releases.html
    (venv) $ pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```
Then install the rest of the dependencies (which are in the `requirements.txt` file):
```bash
(venv) $ pip install .
```

### Running the code
You should first unzip the dataset in the `dataset` folder.

```bash
    unzip data.zip
```

#### Training
The training of the model is made really easy, as simple as creating a trainer with the experiment configuration and calling the `train_and_evaluate` method:
    
```python
import jaxer
from training_config import config


if __name__ == '__main__':

    trainer = jaxer.run.FlaxTrainer(config=config)
    trainer.train_and_evaluate()
```

#### Configuration
An example of the configuration file is in the `training_config.py` file. Here is an example of the configuration file for the training of the model:
    
```python
import jaxer

output_mode = 'mean'  # 'mean' or 'distribution' or 'discrete_grid
seq_len = 100
d_model = 128

model_config = jaxer.config.ModelConfig(
    d_model=d_model,
    num_layers=4,
    head_layers=2,
    n_heads=2,
    dim_feedforward=4 * d_model,  # 4 * d_model
    dropout=0.05,
    max_seq_len=seq_len,
    flatten_encoder_output=False,
    fe_blocks=0,  # feature extractor is incremental, for instance input_shape, 128/2, 128 (d_model)
    use_time2vec=False,
    output_mode=output_mode,  # 'mean' or 'distribution' or 'discrete_grid'
    use_resblocks_in_head=False,
    use_resblocks_in_fe=True,
    average_encoder_output=False,
    norm_encoder_prev=True
)

dataset_config = jaxer.config.DatasetConfig(
    datapath='./data/datasets/data/',
    output_mode=output_mode,  # 'mean' or 'distribution' or 'discrete_grid
    discrete_grid_levels=[-9e6, 0.0, 9e6],
    initial_date='2018-01-01',
    norm_mode="window_minmax",
    resolution='30m',
    tickers=['btc_usd', 'eth_usd', 'sol_usd'],
    indicators=None,
    seq_len=seq_len
)

synthetic_dataset_config = jaxer.config.SyntheticDatasetConfig(
    window_size=seq_len,
    output_mode=output_mode,  # 'mean' or 'distribution' or 'discrete_grid
    normalizer_mode='window_minmax',  # 'window_meanstd' or 'window_minmax' or 'window_mean'
    add_noise=False,
    min_amplitude=0.1,
    max_amplitude=1.0,
    min_frequency=0.5,
    max_frequency=30,
    num_sinusoids=5
)

pretrained_folder = "results/exp_synthetic_context"
pretrained_path_subfolder, pretrained_path_ckpt = jaxer.utils.get_best_model(pretrained_folder)
pretrained_model = (pretrained_folder, pretrained_path_subfolder, pretrained_path_ckpt)

config = jaxer.config.ExperimentConfig(
    model_config=model_config,
    pretrained_model=pretrained_model,
    log_dir="results",
    experiment_name="exp_both_pretrained_synthetic",
    num_epochs=1000,
    steps_per_epoch=500,  # for synthetic dataset only
    learning_rate=5e-4,
    lr_mode='cosine',  # 'cosine' 
    warmup_epochs=15,
    dataset_mode='both',  # 'real' or 'synthetic' or 'both'
    dataset_config=dataset_config,
    synthetic_dataset_config=synthetic_dataset_config,
    batch_size=256,
    test_split=0.1,
    test_tickers=['btc_usd'],
    seed=0,
    save_weights=True,
    early_stopper=100
)

```

#### Inference
An agent class has been created, so you can load a trained model and use it to predict any data you want:
        
```python
import jaxer

from torch.utils.data import DataLoader

if __name__ == '__main__':
    # load the agent with best model weights
    experiment = "exp_1"
    agent = jaxer.run.Agent(experiment=experiment, model_name=jaxer.utils.get_best_model(experiment))
    
    # creater dataloaders
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
```

## Contributors

Rodrigo Sánchez Molina

- Email: rsanchezm98@gmail.com
- Linkedin: [rsanchezm98](https://www.linkedin.com/in/rsanchezm98/)
- Github: [rsanchezmo](https://github.com/rsanchezmo)

## Citation
If you find Jaxer useful, please consider citing:

```bibtex
  @misc{2024rsanchezmo,
    title     = {Jaxer: Jax and Flax Time Series Prediction Transformer},
    author    = {Rodrigo Sánchez Molina},
    year      = {2024},
    howpublished = {https://github.com/rsanchezmo/jaxer}
  }
```