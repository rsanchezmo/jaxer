from jaxer.utils.config import ModelConfig, Config


model_config = ModelConfig(
    d_model=128,
    num_layers=1,
    head_layers=2,
    n_heads=8,
    dim_feedforward=4*128,  # 4 * d_model
    dropout=0.0,
    max_seq_len=20,
    input_features=7,  # [open, high, low, close, volume, trades, time_embed]
    flatten_encoder_output=False,
    fe_blocks=2,
    use_time2vec=True,
    output_distribution=False,
    use_resblocks_in_head=False,
    use_resblocks_in_fe=False
)

config = Config(
    model_config=model_config,
    log_dir="results",
    experiment_name="output_mean_new",
    num_epochs=200,
    learning_rate=5e-5,
    lr_mode='cosine',  # 'cosine 
    warmup_epochs=10,
    dataset_path="./data/btc_usd_4h.json",
    initial_date='2018-01-01',  # the initial date is 2018-01-01
    batch_size=256,
    test_split=0.1,
    seed=0,
    normalizer_mode="global_minmax",
    save_weights=True,
    early_stopper=20
)