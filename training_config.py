import os.path

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
    normalizer_mode='window_minmax',  # 'window_meanstd' or 'window_minmax'
    add_noise=False,
    min_amplitude=0.1,
    max_amplitude=1.0,
    min_frequency=0.5,
    max_frequency=30
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
    dataset_mode='both',  # 'real' or 'synthetic' (in the future may be both, will see)
    dataset_config=dataset_config,
    synthetic_dataset_config=synthetic_dataset_config,
    batch_size=256,
    test_split=0.1,
    test_tickers=['btc_usd'],
    seed=0,
    save_weights=True,
    early_stopper=100
)
