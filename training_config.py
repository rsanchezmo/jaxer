import jaxer

output_mode = 'mean'  # 'mean' or 'distribution' or 'discrete_grid
seq_len = 128
d_model = 256
precision = 'fp32'

model_config = jaxer.config.ModelConfig(
    precision=precision,  # 'fp32' or 'fp16'
    d_model=d_model,
    num_layers=4,
    head_layers=2,
    n_heads=4,
    dim_feedforward=4 * d_model,  # 4 * d_model
    dropout=0.05,
    max_seq_len=seq_len,
    flatten_encoder_output=False,
    fe_blocks=0,  # feature extractor is incremental, for instance input_shape, 128/2, 128 (d_model)
    use_time2vec=False,
    output_mode=output_mode,  # 'mean' or 'distribution' or 'discrete_grid'
    use_resblocks_in_head=True,
    use_resblocks_in_fe=False,
    average_encoder_output=True,  # average the encoder output to get a context embedding
    norm_encoder_prev=True
)

dataset_config = jaxer.config.DatasetConfig(
    datapath='./data/datasets/data/',
    output_mode=output_mode,  # 'mean' or 'distribution' or 'discrete_grid
    discrete_grid_levels=[-9e6, 0.0, 9e6],
    initial_date='2018-01-01',
    norm_mode="window_mean",
    resolution='30m',
    tickers=['btc_usd', 'eth_usd', 'sol_usd'],
    indicators=None,
    seq_len=seq_len
)

synthetic_dataset_config = jaxer.config.SyntheticDatasetConfig(
    window_size=seq_len,
    output_mode=output_mode,  # 'mean' or 'distribution' or 'discrete_grid
    normalizer_mode='window_mean',  # 'window_meanstd' or 'window_minmax' or 'window_mean'
    add_noise=True,
    min_amplitude=0.05,
    max_amplitude=0.2,
    min_frequency=0.1,
    max_frequency=50,
    num_sinusoids=20,
    max_linear_trend=0.5,
    max_exp_trend=0.01,
    precision=precision
)

pretrained_folder = "results/synth_tiny_from_pretrained"
pretrained_path_subfolder, pretrained_path_ckpt = jaxer.utils.get_best_model(pretrained_folder)
pretrained_model = (pretrained_folder, pretrained_path_subfolder, pretrained_path_ckpt)

# pretrained_model = None

config = jaxer.config.ExperimentConfig(
    model_config=model_config,
    pretrained_model=pretrained_model,
    log_dir="results",
    experiment_name="real_tiny_from_pretrained",
    num_epochs=500,
    steps_per_epoch=500,  # for synthetic dataset only
    learning_rate=8e-5,
    lr_mode='linear',  # 'cosine'
    warmup_epochs=20,
    dataset_mode='real',  # 'real' or 'synthetic' or 'both'
    real_proportion=0.3,
    dataset_config=dataset_config,
    synthetic_dataset_config=synthetic_dataset_config,
    batch_size=256,
    test_split=0.1,
    test_tickers=['btc_usd'],
    seed=0,
    save_weights=True,
    early_stopper=100
)
