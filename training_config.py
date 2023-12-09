from jaxer.utils.config import ModelConfig, Config


model_config = ModelConfig(
    d_model=64,
    num_layers=2,
    head_layers=1,
    n_heads=4,
    dim_feedforward=4*64,  # 4 * d_model
    dropout=0.0,
    max_seq_len=15,
    input_features=5,
    flatten_encoder_output=False,
    feature_extractor_residual_blocks=0,
    use_time2vec=False
)

config = Config(
    model_config=model_config,
    log_dir="results",
    experiment_name="overfit",
    num_epochs=300,
    learning_rate=3e-4,
    warmup_epochs=10,
    dataset_path="./data/BTCUSD.csv",
    initial_date='2020-01-01',
    batch_size=64,
    test_split=0.1,
    seed=0,
    normalizer_mode="global_minmax",
)