from jaxer.utils.config import ModelConfig, Config


model_config = ModelConfig(
    d_model=256,
    num_layers=4,
    head_layers=2,
    n_heads=8,
    dim_feedforward=4*256,  # 4 * d_model
    dropout=0.0,
    max_seq_len=32,
    input_features=8,  # [open, high, low, close, volume, trades, returns_close, time_embed]
    flatten_encoder_output=False,
    fe_blocks=2,  # feature extractor is incremental, for instance input_shape, 128/2, 128 (d_model)
    use_time2vec=True,
    output_distribution=False,
    use_resblocks_in_head=True,
    use_resblocks_in_fe=False,
    average_encoder_output=False,
    norm_encoder_prev=False
)

config = Config(
    model_config=model_config,
    log_dir="results",
    experiment_name="output_mean_search",
    num_epochs=400,
    learning_rate=1e-4,
    lr_mode='linear',  # 'cosine' 
    warmup_epochs=15,
    dataset_path="./data/btc_usd_4h.json",
    initial_date='2019-01-01',  # the initial date is 2018-01-01
    batch_size=128,
    test_split=0.1,
    seed=0,
    normalizer_mode="global_minmax",
    save_weights=True,
    early_stopper=50
)