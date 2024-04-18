import jaxer

output_mode = 'mean'  # 'mean' or 'distribution' or 'discrete_grid
seq_len = 128
d_model = 256
precision = 'fp32'

model_config = jaxer.config.ModelConfig(
    precision=precision,  # 'fp32' or 'fp16'
    d_model=d_model,
    num_layers=8,
    head_layers=1,
    n_heads=8,
    dim_feedforward=4 * d_model,  # 4 * d_model
    dropout=0.2,
    max_seq_len=seq_len,
    flatten_encoder_output=False,
    fe_blocks=1,  # feature extractor is incremental, for instance input_shape, 128/2, 128 (d_model)
    use_time2vec=False,
    output_mode=output_mode,  # 'mean' or 'distribution' or 'discrete_grid'
    use_resblocks_in_head=True,
    use_resblocks_in_fe=True,
    use_extra_tokens=False,
    average_encoder_output=False,  # what about concat the average and the last embedding?
    norm_encoder_prev=True
)

return_mode = False
norm_mode = 'window_meanstd'
close_only = True
ohlc_only = False

dataset_config = jaxer.config.DatasetConfig(
    datapath='./data/datasets/data/',
    output_mode=output_mode,  # 'mean' or 'distribution' or 'discrete_grid
    discrete_grid_levels=None,
    initial_date='2018-01-01',
    norm_mode=norm_mode,
    resolution='all',  # 4h, 1h, 30m, all?
    tickers=['btc_usd', 'eth_usd', 'sol_usd'],
    indicators=None,
    seq_len=seq_len,
    ohlc_only=ohlc_only,
    return_mode=return_mode,
    close_only=close_only
)

synthetic_dataset_config = jaxer.config.SyntheticDatasetConfig(
    window_size=seq_len,
    return_mode=return_mode,
    output_mode=output_mode,  # 'mean' or 'distribution' or 'discrete_grid
    normalizer_mode=norm_mode,  # 'window_meanstd' or 'window_minmax' or 'window_mean' or 'none'
    add_noise=True,
    min_amplitude=0.05,
    max_amplitude=0.1,
    min_frequency=0.1,
    max_frequency=20,
    num_sinusoids=10,
    max_linear_trend=0.6,
    max_exp_trend=0.03,
    precision=precision,
    close_only=close_only
)

# pretrained_folder = "results/synth_tiny_from_pretrained"
# pretrained_path_subfolder, pretrained_path_ckpt = jaxer.utils.get_best_model(pretrained_folder)
# pretrained_model = (pretrained_folder, pretrained_path_subfolder, pretrained_path_ckpt)

pretrained_model = None

config = jaxer.config.ExperimentConfig(
    model_config=model_config,
    pretrained_model=pretrained_model,
    log_dir="results_new",
    experiment_name="mix",
    num_epochs=200,
    steps_per_epoch=500,  # for synthetic dataset only
    learning_rate=5e-4,
    lr_mode='cosine',  # 'cosine'
    weight_decay=1.0e-3,
    warmup_epochs=10,
    dataset_mode='real',  # 'real' or 'synthetic' or 'both'
    real_proportion=0.4,
    dataset_config=dataset_config,
    synthetic_dataset_config=synthetic_dataset_config,
    batch_size=256,
    test_split=0.01,
    test_tickers=['btc_usd'],
    seed=0,
    save_weights=True,
    early_stopper=100
)
