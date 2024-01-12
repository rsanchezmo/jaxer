from jaxer.utils.config import ModelConfig, Config


model_config = ModelConfig(
    d_model=256,
    num_layers=1,
    head_layers=1,
    n_heads=4,
    dim_feedforward=4*256,  # 4 * d_model
    dropout=0.0,
    max_seq_len=5,
    input_features=5,
    flatten_encoder_output=False,
    fe_blocks=1,
    use_time2vec=False,
    output_distribution=False,
    use_resblocks_in_head=False,
    use_resblocks_in_fe=False
)

config = Config(
    model_config=model_config,
    log_dir="results",
    experiment_name="output_mean",
    num_epochs=1500,
    learning_rate=5e-5,
    warmup_epochs=20,
    dataset_path="./data/BTCUSD.csv",
    initial_date='2020-01-01',
    batch_size=128,
    test_split=0.1,
    seed=0,
    normalizer_mode="global_minmax",
)