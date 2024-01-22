from jaxer.utils.config import ModelConfig, Config


model_config = ModelConfig(
    d_model=128,
    num_layers=1,
    head_layers=1,
    n_heads=8,
    dim_feedforward=4*128,  # 4 * d_model
    dropout=0.0,
    max_seq_len=32,
    input_features=7,  # [open, high, low, close, volume, trades, open_std, high_std, low_std, close_std, volume_std, trades_std, time_embed]
    flatten_encoder_output=False,
    fe_blocks=1,  # feature extractor is incremental, for instance input_shape, 128/2, 128 (d_model)
    use_time2vec=False,
    output_mode='discrete_grid',  # 'mean' or 'distribution' or 'discrete_grid'
    use_resblocks_in_head=False,
    use_resblocks_in_fe=False,
    average_encoder_output=False,
    norm_encoder_prev=True
)

config = Config(
    model_config=model_config,
    log_dir="results",
    experiment_name="output_discrete_grid",
    num_epochs=1000,
    learning_rate=1e-4,
    lr_mode='linear',  # 'cosine' 
    warmup_epochs=15,
    dataset_path="./data/btc_usd_4h.json",
    dataset_discrete_levels=[-9e6, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 9e6],
    initial_date='2018-01-01',  # the initial date is 2018-01-01
    batch_size=128,
    test_split=0.1,
    seed=0,
    normalizer_mode="global_minmax",
    save_weights=True,
    early_stopper=100
)