from jaxer.utils.config import ModelConfig, Config


model_config = ModelConfig(
    d_model=256,
    num_layers=4,
    head_layers=2,
    n_heads=4,
    dim_feedforward=4*256,  # 4 * d_model
    dropout=0.0,
    max_seq_len=10,
    input_features=5,
    flatten_encoder_output=False,
    feature_extractor_residual_blocks=2,
    use_time2vec=False
)

config = Config(
    model_config=model_config,
    log_dir="results",
    experiment_name="step_by_step_x2",
    num_epochs=100,
    learning_rate=3e-4,
    warmup_epochs=10,
    dataset_path="./data/BTCUSD.csv",
    initial_date='2018-01-01',
    batch_size=64,
    test_split=0.1,
    seed=0,
    normalizer_mode="global_minmax",
)