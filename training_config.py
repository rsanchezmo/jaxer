from jaxer.utils.config import ModelConfig, Config


model_config = ModelConfig(
    d_model=128,
    num_layers=1,
    head_layers=1,
    n_heads=4,
    dim_feedforward=4*128,  # 4 * d_model
    dropout=0.0,
    max_seq_len=10,
    input_features=5,
    flatten_encoder_output=False,
    feature_extractor_residual_blocks=0,
    use_time2vec=False,
    output_distribution=False
)

config = Config(
    model_config=model_config,
    log_dir="results",
    experiment_name="overfit",
    num_epochs=1000,
    learning_rate=1e-4,
    warmup_epochs=20,
    dataset_path="./data/BTCUSD.csv",
    initial_date='2021-01-01',
    batch_size=64,
    test_split=0.05,
    seed=0,
    normalizer_mode="global_minmax",
)